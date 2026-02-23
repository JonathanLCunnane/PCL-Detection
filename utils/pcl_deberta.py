from enum import Enum
from torch import nn
import torch
from transformers import AutoModel
from logging import getLogger

LOG = getLogger(__name__)


class PoolingStrategy(Enum):
    """Pooling strategies for extracting a fixed-size representation from the backbone."""
    CLS = "cls"           # [CLS] token (index 0)
    MEAN = "mean"         # Attention-mask-weighted mean
    MAX = "max"           # Element-wise max over non-padding tokens
    CLS_MEAN = "cls_mean" # Concatenation of CLS and mean (2x hidden size)


class PCLClassifierHead(nn.Module):
    """
    Custom classifier head for PCL detection.

    If hidden_dim > 0: Linear -> GELU -> Dropout -> Linear (2-layer MLP)
    If hidden_dim == 0: Linear only (single layer, like SimpleTransformers)

    The input accepts (cls_dim + n_extra_features), allowing future
    experiments to concatenate additional features.
    """

    def __init__(self, cls_dim: int = 768, hidden_dim: int = 256,
                 dropout_rate: float = 0.1, n_extra_features: int = 0):
        super().__init__()
        input_dim = cls_dim + n_extra_features
        if hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Single linear layer — matches SimpleTransformers / HuggingFace default
            self.head = nn.Linear(input_dim, 1)

    def forward(self, cls_embedding: torch.Tensor,
                extra_features: torch.Tensor | None = None) -> torch.Tensor:
        if extra_features is not None:
            x = torch.cat([cls_embedding, extra_features], dim=-1)
        else:
            x = cls_embedding
        return self.head(x)
    

class PCLDeBERTa(nn.Module):
    """
    Full model: DeBERTa backbone + PCLClassifierHead.

    Pooling strategies (searched as a hyperparameter via PoolingStrategy enum):
      - CLS:      Use the [CLS] token representation (index 0).
      - MEAN:     Attention-mask-weighted mean over all tokens.
      - MAX:      Element-wise max over non-padding tokens.
      - CLS_MEAN: Concatenate [CLS] and mean pooled (doubles input dim to 1536).

    DeBERTa-v3 was pretrained with RTD (not NSP), so [CLS] has no specially
    learned representation — mean/max pooling may outperform it, but we
    search over all strategies to let the data decide.
    """

    def __init__(self, hidden_dim: int = 256, dropout_rate: float = 0.1,
                 n_extra_features: int = 0, pooling: PoolingStrategy = PoolingStrategy.MEAN,
                 model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        self.pooling = pooling

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone = self.backbone.float()
        self.backbone.gradient_checkpointing_enable()
        LOG.info(f"Backbone model loaded: {self.backbone.__class__.__name__}, dtype {self.backbone.dtype}, "
                 f"gradient_checkpointing={self.backbone.is_gradient_checkpointing}")

        base_dim = self.backbone.config.hidden_size  # 768
        # cls_mean concatenates two representations -> 2x hidden size
        cls_dim = base_dim * 2 if pooling == PoolingStrategy.CLS_MEAN else base_dim

        self.classifier = PCLClassifierHead(
            cls_dim=cls_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            n_extra_features=n_extra_features
        )

    def _pool(self, last_hidden: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply the configured pooling strategy."""
        if self.pooling == PoolingStrategy.CLS:
            return last_hidden[:, 0, :]  # [CLS] is always index 0

        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)

        if self.pooling == PoolingStrategy.MEAN:
            sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / count

        if self.pooling == PoolingStrategy.MAX:
            # Replace padding positions with -inf so they are never selected
            last_hidden = last_hidden.masked_fill(mask_expanded == 0, -1e9)
            return last_hidden.max(dim=1).values

        # PoolingStrategy.CLS_MEAN
        cls_repr = last_hidden[:, 0, :]
        sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_repr = sum_hidden / count
        return torch.cat([cls_repr, mean_repr], dim=-1)  # (B, 2*hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None,
                extra_features: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden)

        pooled = self._pool(last_hidden, attention_mask)
        scores = self.classifier(pooled, extra_features)
        return scores
