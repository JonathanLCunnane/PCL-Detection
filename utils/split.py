import pandas as pd
from sklearn.model_selection import train_test_split
from logging import getLogger

LOG = getLogger(__name__)

def split_train_val(train_df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """Stratified split of training data into train_sub and val_sub."""
    train_sub, val_sub = train_test_split(
        train_df, test_size=val_frac, random_state=seed,
        stratify=train_df["binary_label"]
    )
    LOG.info(f"Train/val split: {len(train_sub)} train, {len(val_sub)} val (val_frac={val_frac})")
    LOG.info(f"Train, val positive count: {train_sub['binary_label'].sum()}, {val_sub['binary_label'].sum()}")
    return train_sub, val_sub
