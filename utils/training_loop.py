from utils.pcl_deberta import PCLDeBERTa
from utils.optim import get_cosine_schedule_with_warmup
from utils.early_stopping import EarlyStopping
from utils.eval import evaluate, find_best_threshold
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
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
    trial: optuna.trial.Trial | None = None,
) -> dict:
    """
    Train the model with early stopping, cosine annealing with warmup,
    and weighted BCE loss.

    Args:
        head_lr_multiplier: The head LR is lr * head_lr_multiplier.
        label_smoothing: Smooths binary targets towards 0.5.
            y_smooth = y * (1 - label_smoothing) + 0.5 * label_smoothing

    Returns dict with keys: best_val_f1, best_threshold, dev_metrics, train_losses.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential LR: head learns head_lr_multiplier times faster than backbone
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * head_lr_multiplier}
    ], weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_fraction)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Scale patience from epochs to eval rounds so early stopping works in epoch units
    evals_per_epoch = max(1, len(train_loader) // eval_every_n_steps)
    patience_in_evals = patience * evals_per_epoch
    early_stopper = EarlyStopping(patience=patience_in_evals)
    LOG.info(f"Early stopping: patience={patience} epochs = {patience_in_evals} eval rounds "
             f"({evals_per_epoch} evals/epoch, eval every {eval_every_n_steps} steps)")

    model.train()
    global_step = 0
    train_losses = []
    best_val_f1 = 0.0
    best_state_dict = None
    running_loss = 0.0

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

            optimizer.zero_grad()
            
            unnormalised_scores = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features).squeeze(-1)
            loss = criterion(unnormalised_scores, labels)
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

                val_metrics = evaluate(model, device, val_loader)
                val_f1 = val_metrics["f1"]

                LOG.info(
                    f"Step {global_step} | Train Loss: {avg_train_loss:.4f} | "
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

    dev_metrics = evaluate(model, device, dev_loader, threshold=best_thresh)
    LOG.info(
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