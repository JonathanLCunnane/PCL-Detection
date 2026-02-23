from torch import nn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from collections.abc import Callable

@torch.no_grad()
def evaluate(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate the model on a DataLoader.
    Returns dict with keys: f1, precision, recall, loss, preds, labels, threshold.
    """
    was_training = model.training
    model.eval()
    all_scores = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        extra_features = batch.get("extra_features", None)

        unnormalised_scores = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features).squeeze(-1)
        all_scores.append(unnormalised_scores.cpu())
        all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)

    loss = nn.functional.binary_cross_entropy_with_logits(all_scores, all_labels).item()

    probs = torch.sigmoid(all_scores)
    thresh_preds = (probs >= threshold).long().numpy()
    labels_np = all_labels.long().numpy()

    f1 = f1_score(labels_np, thresh_preds, zero_division=0)
    precision = precision_score(labels_np, thresh_preds, zero_division=0)
    recall = recall_score(labels_np, thresh_preds, zero_division=0)

    model.train(was_training)
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "loss": loss, "preds": thresh_preds, "labels": labels_np,
        "threshold": threshold
    }


@torch.no_grad()
def find_best_threshold(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
) -> tuple[float, float]:
    """
    Sweep thresholds on a validation set to find the one maximising F1.
    Returns (best_threshold, best_f1).
    """
    model.eval()
    all_scores = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        extra_features = batch.get("extra_features", None)

        unnormalised_scores = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features).squeeze(-1)
        all_scores.append(unnormalised_scores.cpu())
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