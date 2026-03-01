from transformers import SentencePieceBackend, TokenizersBackend
from torch.utils.data import Dataset
import torch
import numpy as np
from logging import getLogger

LOG = getLogger(__name__)

class PCLDataset(Dataset):
    """
    Pre-tokenising all of the text to prevent doing it each time it is needed.
    """

    def __init__(self, *, texts: list[str], labels: list[int] | None = None, 
                 max_length: int, tokeniser: TokenizersBackend | SentencePieceBackend,
                 extra_features: torch.Tensor | None = None):
        self.extra_features = extra_features
        self.encodings = tokeniser(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]
        return item


class PCLCategoryDataset(Dataset):
    """
    PCLDataset extended with per-category multi-hot labels for Exp C.

    Each batch item contains:
      - 'labels'          : binary float (0/1) — overall PCL classification
      - 'category_labels' : float tensor of shape (n_categories,) — multi-hot
                            vector indicating which PCL categories are present.
                            Non-PCL paragraphs have all-zero category labels.
    """

    def __init__(self, *, texts: list[str], labels: list[int],
                 category_labels,  # shape (N, n_categories), array-like
                 max_length: int, tokeniser: TokenizersBackend | SentencePieceBackend):
        self.encodings = tokeniser(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.category_labels = torch.tensor(category_labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        item["category_labels"] = self.category_labels[idx]
        return item