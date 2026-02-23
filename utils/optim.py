import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from logging import getLogger

LOG = getLogger(__name__)


def compute_pos_weight(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss to handle class imbalance.
    Uses sqrt(num_negatives / num_positives) instead of the raw ratio
    to reduce gradient variance with small batch sizes.
    """
    n_pos = df["binary_label"].sum()
    n_neg = len(df) - n_pos
    raw_ratio = n_neg / n_pos
    weight = torch.tensor([math.sqrt(raw_ratio)], dtype=torch.float32).to(device)
    LOG.info(f"pos_weight = {weight.item():.2f} (raw ratio={raw_ratio:.2f}, neg={n_neg}, pos={n_pos})")
    return weight


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> LambdaLR:
    """
    Cosine annealing LR schedule with linear warmup.
    - step < num_warmup_steps: LR increases linearly from 0 to base_lr.
    - step >= num_warmup_steps: LR decays following a cosine curve to 0.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)