"""Multi-scale Conv1D motif detectors over residue embeddings (silk is repetitive), then
mask-aware mean+max pooling -> MLP. Drop in as model.py."""
import torch
import torch.nn as nn

DESCRIPTION = "multi-scale Conv1D motif detectors -> mean+max pool -> MLP"


class ConvMotif(nn.Module):
    def __init__(self, embed_dim, n_targets, channels=256, kernels=(3, 7, 15), dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, channels, k, padding=k // 2) for k in kernels])
        c = channels * len(kernels)
        self.head = nn.Sequential(
            nn.Linear(2 * c, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channels, n_targets),
        )

    def forward(self, x, mask):
        x = self.norm(x).transpose(1, 2)                   # (B, d, L)
        h = torch.cat([torch.relu(c(x)) for c in self.convs], dim=1)  # (B, c, L)
        m = mask.unsqueeze(1).float()                      # (B, 1, L)
        h = h * m
        mean = h.sum(2) / m.sum(2).clamp(min=1.0)
        mx = h.masked_fill(m == 0, float("-inf")).max(2).values
        return self.head(torch.cat([mean, mx], dim=1))


def build_model(embed_dim, n_targets, cfg):
    return ConvMotif(embed_dim, n_targets, cfg.get("channels", 256), dropout=cfg.get("dropout", 0.3))
