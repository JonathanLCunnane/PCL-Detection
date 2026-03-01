from collections.abc import Callable
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import SentencePieceBackend, TokenizersBackend
from utils.feature import FeatureType, extract_ner_features, extract_pos_features, extract_keyword_feature
from utils.fightin_words import extract_zscore_features
from utils.pcl_dataset import PCLDataset, PCLCategoryDataset
from utils.data import PCL_CATEGORIES
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


def make_category_dataloaders(
    train_sub_df: pd.DataFrame,
    val_sub_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    tokeniser: TokenizersBackend | SentencePieceBackend,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders with per-category multi-hot labels for Exp C.

    DataFrames must have 'text', 'binary_label', and one column per
    PCL_CATEGORIES entry (produced by load_data_categories).

    Each batch item contains:
      'labels'          : binary float (PCL vs not-PCL)
      'category_labels' : float tensor (n_categories,) — multi-hot
    """
    def dataset_for(df: pd.DataFrame) -> PCLCategoryDataset:
        return PCLCategoryDataset(
            texts=df["text"].tolist(),
            labels=df["binary_label"].tolist(),
            category_labels=df[PCL_CATEGORIES].values,
            max_length=max_length,
            tokeniser=tokeniser,
        )

    train_ds = dataset_for(train_sub_df)
    val_ds   = dataset_for(val_sub_df)
    dev_ds   = dataset_for(dev_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dev_loader


def make_weighted_dataloaders(
    train_sub_df: pd.DataFrame,
    val_sub_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    tokeniser: TokenizersBackend | SentencePieceBackend,
    target_pos_frac: float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Training DataLoader uses WeightedRandomSampler to oversample the PCL
    minority class. target_pos_frac controls the expected fraction of positive
    samples per batch (0.5 = balanced, ~0.095 = natural).

    When using this loader, set pos_weight=1.0 in the BCE loss — the sampler
    already handles the imbalance; adding pos_weight would double-compensate.
    """
    labels = np.array(train_sub_df["binary_label"].tolist())
    pos_count = int(labels.sum())
    neg_count = len(labels) - pos_count
    pos_w = target_pos_frac / pos_count
    neg_w = (1.0 - target_pos_frac) / neg_count
    sample_weights = np.where(labels == 1, pos_w, neg_w).tolist()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)

    train_ds = PCLDataset(texts=train_sub_df["text"].tolist(),
                          labels=train_sub_df["binary_label"].tolist(),
                          max_length=max_length, tokeniser=tokeniser)
    val_ds   = PCLDataset(texts=val_sub_df["text"].tolist(),
                          labels=val_sub_df["binary_label"].tolist(),
                          max_length=max_length, tokeniser=tokeniser)
    dev_ds   = PCLDataset(texts=dev_df["text"].tolist(),
                          labels=dev_df["binary_label"].tolist(),
                          max_length=max_length, tokeniser=tokeniser)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dev_loader


def make_keyword_dataloaders(
    train_sub_df: pd.DataFrame,
    val_sub_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    tokeniser: TokenizersBackend | SentencePieceBackend,
    keyword_to_idx: dict,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders with keyword one-hot extra features (FeatureType.KEYWORD).

    DataFrames must have 'text', 'binary_label', and 'keyword' columns
    (produced by load_data_with_keyword).

    The keyword_to_idx vocabulary should be built from train_sub_df only
    to avoid leakage. Unknown keywords in val/dev get a zero vector.
    """
    def get_kw_tensor(df: pd.DataFrame) -> torch.Tensor:
        feats = np.array(
            [extract_keyword_feature(kw, keyword_to_idx) for kw in df["keyword"]],
            dtype=np.float32,
        )
        return torch.tensor(feats).to(device)

    def dataset_for(df: pd.DataFrame) -> PCLDataset:
        return PCLDataset(
            texts=df["text"].tolist(),
            labels=df["binary_label"].tolist(),
            max_length=max_length,
            tokeniser=tokeniser,
            extra_features=get_kw_tensor(df),
        )

    train_ds = dataset_for(train_sub_df)
    val_ds   = dataset_for(val_sub_df)
    dev_ds   = dataset_for(dev_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dev_loader
