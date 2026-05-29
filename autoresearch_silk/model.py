"""Reference baseline: masked mean-pool -> 2-layer MLP. (Same as the seeded model.py.)"""
import torch.nn as nn

DESCRIPTION = "baseline: masked mean-pool -> 2-layer MLP"


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class MeanPoolMLP(nn.Module):
    def __init__(self, embed_dim, n_targets, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )

    def forward(self, x, mask):
        return self.net(masked_mean(x, mask))


def build_model(embed_dim, n_targets, cfg):
    return MeanPoolMLP(embed_dim, n_targets, cfg.get("hidden", 256), cfg.get("dropout", 0.3))
