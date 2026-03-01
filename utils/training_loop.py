from utils.pcl_deberta import PCLDeBERTa
from utils.optim import get_cosine_schedule_with_warmup
from utils.early_stopping import EarlyStopping
from utils.eval import evaluate, find_best_threshold
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import bitsandbytes as bnb
from logging import getLogger
import optuna
from collections.abc import Callable

LOG = getLogger(__name__)


def train_model(
    model: PCLDeBERTa,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dev_loader: DataLoader,
    pos_weight: torch.Tensor,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    warmup_fraction: float,
    patience: int,
    head_lr_multiplier: float = 3.0,
    label_smoothing: float = 0.0,
    eval_every_n_steps: int = 50,
    accumulate_grad_batches: int = 1,
    use_8bit_adam: bool = False,
    trial: optuna.trial.Trial | None = None,
) -> dict:
    """
    Train the model with early stopping, cosine annealing with warmup,
    and weighted BCE loss.

    Args:
        head_lr_multiplier: The head LR is lr * head_lr_multiplier.
        label_smoothing: Smooths binary targets towards 0.5.
            y_smooth = y * (1 - label_smoothing) + 0.5 * label_smoothing
        accumulate_grad_batches: Accumulate gradients over this many micro-batches
            before each optimizer step. Effective batch size = batch_size × accumulate_grad_batches.
            eval_every_n_steps refers to optimizer steps, not micro-batches.
        use_8bit_adam: Use bitsandbytes 8-bit AdamW (saves ~2.4 GB for large models).
            Falls back to standard AdamW if bitsandbytes is not installed.

    Returns dict with keys: best_val_f1, best_threshold, dev_metrics, train_losses.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential LR: head learns head_lr_multiplier times faster than backbone
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    param_groups = [
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_multiplier},
    ]
    if use_8bit_adam:
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=weight_decay)
        LOG.info("Using bitsandbytes 8-bit AdamW")
    else:
        optimizer = AdamW(param_groups, weight_decay=weight_decay)

    optimizer_steps_per_epoch = max(1, len(train_loader) // accumulate_grad_batches)
    total_steps = optimizer_steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Scale patience from epochs to eval rounds so early stopping works in epoch units
    evals_per_epoch = max(1, optimizer_steps_per_epoch // eval_every_n_steps)
    patience_in_evals = patience * evals_per_epoch
    early_stopper = EarlyStopping(patience=patience_in_evals)
    LOG.info(f"Early stopping: patience={patience} epochs = {patience_in_evals} eval rounds "
             f"({evals_per_epoch} evals/epoch, eval every {eval_every_n_steps} steps)")
    if accumulate_grad_batches > 1:
        LOG.info(f"Gradient accumulation: {accumulate_grad_batches} micro-steps per optimizer step")

    model.train()
    global_step = 0
    micro_step = 0
    train_losses = []
    best_val_f1 = 0.0
    best_state_dict = None
    running_loss = 0.0
    running_micro_steps = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        LOG.info(f"Epoch {epoch+1}/{num_epochs}")
        for batch_id, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            extra_features = batch.get("extra_features", None)

            # Label smoothing: push targets towards 0.5
            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            unnormalised_scores = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features).squeeze(-1)
            loss = criterion(unnormalised_scores, labels)
            (loss / accumulate_grad_batches).backward()

            running_loss += loss.item()
            running_micro_steps += 1
            micro_step += 1

            is_update_step = (micro_step % accumulate_grad_batches == 0)
            is_last_batch = (batch_id + 1 == len(train_loader))

            if is_update_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_every_n_steps == 0:
                    avg_train_loss = running_loss / running_micro_steps
                    train_losses.append(avg_train_loss)
                    running_loss = 0.0
                    running_micro_steps = 0

                    val_metrics = evaluate(model, device, val_loader, criterion=criterion)
                    val_f1 = val_metrics["f1"]
                    val_loss = val_metrics["loss"]

                    LOG.info(
                        f"Step {global_step} | Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val F1: {val_f1:.4f} | Val P: {val_metrics['precision']:.4f} | "
                        f"Val R: {val_metrics['recall']:.4f}"
                    )

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        # Store on CPU to save GPU memory
                        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                    # Optuna pruning
                    if trial is not None:
                        trial.report(val_f1, global_step)
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()

                    if early_stopper.step(val_f1):
                        LOG.info(f"Early stopping at step {global_step}")
                        break

        if early_stopper.should_stop:
            break

    # Restore best model and evaluate on dev set
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Find optimal threshold on val set, apply to dev evaluation
    best_thresh, thresh_val_f1 = find_best_threshold(model, device, val_loader)
    LOG.info(f"Optimal threshold: {best_thresh:.3f} (val F1 with threshold: {thresh_val_f1:.4f}, "
             f"val F1 @0.5: {best_val_f1:.4f})")


    dev_metrics = evaluate(model, device, dev_loader, criterion=criterion, threshold=best_thresh)
    dev_loss = dev_metrics["loss"]
    LOG.info(
        f"Dev Loss: {dev_loss:.4f} | "
        f"Dev F1: {dev_metrics['f1']:.4f} | "
        f"Dev P: {dev_metrics['precision']:.4f} | Dev R: {dev_metrics['recall']:.4f} | "
        f"Threshold: {best_thresh:.3f}"
    )

    return {
        "best_val_f1": thresh_val_f1,  # Use threshold-optimised val F1
        "best_threshold": best_thresh,
        "dev_metrics": dev_metrics,
        "train_losses": train_losses
    }


# ---------------------------------------------------------------------------
# Category multi-task training helpers (binary BCE + per-category BCE)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_category_binary(
    model: PCLDeBERTa,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the binary output of PCLDeBERTa on val/dev batches.

    Model returns logits of shape (B, 1 + n_categories); only logits[:, 0]
    is used for binary evaluation here.
    """
    was_training = model.training
    model.eval()
    all_scores, all_labels = [], []
    total_loss, num_batches = 0.0, 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        binary_logits = logits[:, 0]  # (B,)
        all_scores.append(binary_logits.cpu())
        all_labels.append(labels.cpu())
        total_loss += criterion(binary_logits, labels).item()
        num_batches += 1

    model.train(was_training)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    loss = total_loss / max(1, num_batches)
    probs = torch.sigmoid(all_scores)
    preds = (probs >= threshold).long().numpy()
    labels_np = all_labels.long().numpy()
    return {
        "f1": f1_score(labels_np, preds, zero_division=0),
        "precision": precision_score(labels_np, preds, zero_division=0),
        "recall": recall_score(labels_np, preds, zero_division=0),
        "loss": loss, "preds": preds, "labels": labels_np, "threshold": threshold,
    }


@torch.no_grad()
def _find_best_threshold_category(
    model: PCLDeBERTa,
    device: torch.device,
    loader: DataLoader,
) -> tuple[float, float]:
    """Find the binary threshold maximising F1 on a validation set."""
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        all_scores.append(logits[:, 0].cpu())  # binary logit only
        all_labels.append(labels.cpu())

    probs = torch.sigmoid(torch.cat(all_scores))
    labels_np = torch.cat(all_labels).long().numpy()
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.25, 0.70, 0.025):
        preds = (probs >= thresh).long().numpy()
        f1 = float(f1_score(labels_np, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh, best_f1


def train_category_model(
    model: PCLDeBERTa,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dev_loader: DataLoader,
    pos_weight: torch.Tensor,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    warmup_fraction: float,
    patience: int,
    category_weight: float = 0.5,
    head_lr_multiplier: float = 3.0,
    label_smoothing: float = 0.0,
    eval_every_n_steps: int = 50,
    trial: optuna.trial.Trial | None = None,
) -> dict:
    """
    Train a PCLDeBERTa model with joint loss:

      Loss = BCEWithLogits(binary_logits, binary_labels)
             + category_weight × BCEWithLogits(category_logits, category_labels)

    binary_labels  : float (B,) — 0/1 overall PCL
    category_labels: float (B, n_categories) — multi-hot per-category labels

    Batches from make_category_dataloaders must provide 'labels' and
    'category_labels'. Early stopping and threshold tuning are on binary F1.
    Returns dict with keys: best_val_f1, best_threshold, dev_metrics, train_losses.
    """
    # Single BCEWithLogitsLoss with per-output pos_weight:
    #   index 0 (binary): pos_weight as computed from class imbalance
    #   index 1..n_categories (categories): pos_weight=1 (no per-class weighting)
    n_categories = model.n_out - 1
    full_pos_weight = torch.cat([
        pos_weight.to(device),
        torch.ones(n_categories, device=device),
    ])  # shape (1 + n_categories,)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=full_pos_weight)
    # Separate binary criterion for eval helpers (scalar loss, no categories)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params,     "lr": lr * head_lr_multiplier},
    ], weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    evals_per_epoch = max(1, len(train_loader) // eval_every_n_steps)
    patience_in_evals = patience * evals_per_epoch
    early_stopper = EarlyStopping(patience=patience_in_evals)
    LOG.info(f"[CategoryModel] Early stopping: patience={patience} epochs = {patience_in_evals} eval rounds")

    model.train()
    global_step = 0
    train_losses = []
    best_val_f1 = 0.0
    best_state_dict = None
    running_loss = 0.0

    for epoch in range(num_epochs):
        LOG.info(f"[CategoryModel] Epoch {epoch+1}/{num_epochs}")
        for batch in train_loader:
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            labels          = batch["labels"].to(device)
            category_labels = batch["category_labels"].to(device)

            if label_smoothing > 0:
                labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            # Combined labels: (B, 1 + n_categories)
            full_labels = torch.cat([labels.unsqueeze(1), category_labels], dim=1)
            per_output_loss = criterion(logits, full_labels)  # (B, 1 + n_categories)
            binary_loss   = per_output_loss[:, 0].mean()
            category_loss = per_output_loss[:, 1:].mean()
            loss = binary_loss + category_weight * category_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % eval_every_n_steps == 0:
                avg_train_loss = running_loss / eval_every_n_steps
                train_losses.append(avg_train_loss)
                running_loss = 0.0

                val_metrics = _eval_category_binary(model, device, val_loader, binary_criterion)
                val_f1 = val_metrics["f1"]
                LOG.info(
                    f"Step {global_step} | Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_f1:.4f} | Val P: {val_metrics['precision']:.4f} | "
                    f"Val R: {val_metrics['recall']:.4f}"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if trial is not None:
                    trial.report(val_f1, global_step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if early_stopper.step(val_f1):
                    LOG.info(f"[CategoryModel] Early stopping at step {global_step}")
                    break

        if early_stopper.should_stop:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    best_thresh, thresh_val_f1 = _find_best_threshold_category(model, device, val_loader)
    LOG.info(f"[CategoryModel] Optimal threshold: {best_thresh:.3f} (val F1: {thresh_val_f1:.4f})")

    dev_metrics = _eval_category_binary(
        model, device, dev_loader, binary_criterion, threshold=best_thresh
    )
    LOG.info(
        f"[CategoryModel] Dev F1: {dev_metrics['f1']:.4f} | "
        f"Dev P: {dev_metrics['precision']:.4f} | Dev R: {dev_metrics['recall']:.4f}"
    )

    return {
        "best_val_f1": thresh_val_f1,
        "best_threshold": best_thresh,
        "dev_metrics": dev_metrics,
        "train_losses": train_losses,
    }