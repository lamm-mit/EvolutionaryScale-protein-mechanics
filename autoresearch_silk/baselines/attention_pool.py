"""Attention pooling: a learned query attends over residues (mask-aware) -> MLP head.
Drop in as model.py."""
import torch
import torch.nn as nn

DESCRIPTION = "attention pooling (learned query) -> MLP"


class AttnPool(nn.Module):
    def __init__(self, embed_dim, n_targets, hidden=256, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Sequential(nn.Linear(embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x, mask):
        x = self.norm(x)
        s = self.score(x).squeeze(-1)                      # (B, L)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)          # (B, L, 1)
        pooled = (a * x).sum(1)                            # (B, d)
        return self.head(pooled)


def build_model(embed_dim, n_targets, cfg):
    return AttnPool(embed_dim, n_targets, cfg.get("hidden", 256), cfg.get("dropout", 0.3))
