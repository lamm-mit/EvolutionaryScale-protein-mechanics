"""========================  THE EDITABLE ASSET  ========================
Grouped silk autoresearch model.

Input is one sample per fiber/property idv. Each sample contains a set of independently embedded
spidroin sequences:

  seq_embeddings : (B, Smax, embed_dim) float32   ESMC mean embedding per sequence
  seq_mask       : (B, Smax) bool                 True = real sequence, False = padding
  category_ids   : (B, Smax) int64                0 = padding, >0 = sequence category
  seq_lengths    : (B, Smax) int64                amino-acid length per sequence

Return:
  (B, 4) predictions for [toughness, E, strength, strain]

Autoresearch agents may fundamentally change the set aggregation architecture here: attention,
DeepSets, set transformers, category-aware pooling, mixture-of-experts by category, motif/statistical
features, per-target heads, etc. Keep the function signature stable unless you also update the fixed
harness intentionally.
====================================================================="""
from __future__ import annotations

import torch
import torch.nn as nn

DESCRIPTION = "baseline: ESMC per-sequence mean embeddings -> masked mean set pool -> MLP"


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class SetMeanMLP(nn.Module):
    def __init__(self, embed_dim, n_targets, hidden=256, dropout=0.3, n_categories=None, category_dim=0):
        super().__init__()
        self.use_categories = bool(category_dim and n_categories)
        in_dim = embed_dim
        if self.use_categories:
            self.category_emb = nn.Embedding(n_categories, category_dim, padding_idx=0)
            in_dim += category_dim
        self.seq_norm = nn.LayerNorm(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, seq_embeddings, seq_mask, category_ids=None, seq_lengths=None):
        x = seq_embeddings
        if self.use_categories:
            if category_ids is None:
                raise ValueError("category_ids required when category_dim > 0")
            x = torch.cat([x, self.category_emb(category_ids)], dim=-1)
        pooled = masked_mean(self.seq_norm(x), seq_mask)
        return self.head(pooled)


def build_model(embed_dim, n_targets, cfg, n_categories=None):
    return SetMeanMLP(
        embed_dim,
        n_targets,
        hidden=cfg.get("hidden", 256),
        dropout=cfg.get("dropout", 0.3),
        n_categories=n_categories,
        category_dim=cfg.get("category_dim", 0),
    )
