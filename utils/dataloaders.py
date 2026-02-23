from collections.abc import Callable
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from transformers import SentencePieceBackend, TokenizersBackend
from utils.feature import FeatureType, extract_ner_features, extract_pos_features
from utils.fightin_words import extract_zscore_features
from utils.pcl_dataset import PCLDataset
import numpy as np

def make_dataloaders(
    train_sub_df: pd.DataFrame,
    val_sub_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    tokeniser: TokenizersBackend | SentencePieceBackend,
    extra_feature_factory: Callable[[list[str]], torch.Tensor] | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train-sub, val-sub, and dev sets."""
    dataset_kwargs = lambda df: {
        "texts": df["text"].tolist(),
        "labels": df["binary_label"].tolist(),
        "max_length": max_length,
        "tokeniser": tokeniser,
        "extra_features": extra_feature_factory(df["text"].tolist()) if extra_feature_factory is not None else None
    }
    train_ds = PCLDataset(**dataset_kwargs(train_sub_df))
    val_ds = PCLDataset(**dataset_kwargs(val_sub_df))
    dev_ds = PCLDataset(**dataset_kwargs(dev_df))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dev_loader
