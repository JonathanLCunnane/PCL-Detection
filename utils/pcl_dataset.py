from transformers import SentencePieceBackend, TokenizersBackend, PreTrainedTokenizerBase
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


class PCLVerbalizerDataset(Dataset):
    """
    Dataset for the verbalizer approach.

    Wraps each text in a template containing a single [MASK] token,
    tokenises the combined string, and records the position of the [MASK]
    so the model can extract MLM logits there.

    Template: "{text} . It was [MASK] ."  (default)

    Each batch item contains:
      - input_ids, attention_mask, token_type_ids  (standard)
      - mask_token_indices : LongTensor scalar — position of the [MASK] token
      - labels             : float scalar (0/1)

    If the [MASK] token is truncated away (very long inputs), a warning is
    logged and position 0 is used as a fallback.
    """

    def __init__(self, *, texts: list[str], labels: list[int] | None = None,
                 max_length: int,
                 tokeniser: PreTrainedTokenizerBase,
                 template: str = "{text} . It was {mask} ."):
        mask_token = tokeniser.mask_token
        mask_token_id = tokeniser.mask_token_id

        templated_texts = [template.format(text=t, mask=mask_token) for t in texts]

        self.encodings = tokeniser(
            templated_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

        # Pre-compute mask position for every sample
        input_ids = self.encodings["input_ids"]  # (N, seq_len)
        mask_positions = []
        for i in range(len(input_ids)):
            positions = (input_ids[i] == mask_token_id).nonzero(as_tuple=False).squeeze(-1)
            if positions.numel() == 0:
                LOG.warning(
                    f"Sample {i}: [MASK] token was truncated "
                    f"(text may be too long for max_length={max_length}). "
                    f"Falling back to position 0."
                )
                mask_positions.append(torch.tensor(0, dtype=torch.long))
            else:
                mask_positions.append(positions[0])  # first (and only) [MASK]

        self.mask_token_indices = torch.stack(mask_positions)  # (N,)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["mask_token_indices"] = self.mask_token_indices[idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item