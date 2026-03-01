import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with logits.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy negatives so the model concentrates gradient
    on hard/borderline examples — useful for heavily imbalanced tasks.

    Integrates with pos_weight the same way as BCEWithLogitsLoss:
    positive examples are upweighted by pos_weight before focal scaling.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Element-wise BCE (unreduced), with optional pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        # p_t = p if y=1 else (1-p)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
