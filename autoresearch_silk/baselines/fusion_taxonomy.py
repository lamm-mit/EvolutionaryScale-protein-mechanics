"""FUSION / MULTI-TASK example (the user's idea): a shared trunk over residue embeddings feeds
BOTH the 4-target property head AND auxiliary taxonomy classifiers (family / genus / silk-type),
supervised jointly. The taxonomy heads shape the shared representation; at eval ONLY the property
head is used, so it stays a sequence-only predictor (taxonomy is predicted from sequence, never fed
in). Drop in as model.py; tune `aux_weight` in config.json.

Demonstrates the optional contract run_experiment.py supports:
  AUX_COLS                              : list of meta columns to supervise (from data/*.parquet)
  build_aux_heads(self, class_counts)   : called once with {col: n_classes}
  auxiliary_loss(self, x, mask, labels) : returns a scalar added to the loss during TRAINING only
"""
import torch
import torch.nn as nn

DESCRIPTION = "fusion: shared trunk + property head + auxiliary taxonomy classifiers (multi-task)"


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class Fusion(nn.Module):
    AUX_COLS = ["family", "genus", "category1"]      # taxonomy targets (predicted from sequence)

    def __init__(self, embed_dim, n_targets, hidden=256, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.prop = nn.Linear(hidden, n_targets)
        self.aux = nn.ModuleDict()
        self.ce = nn.CrossEntropyLoss()

    def build_aux_heads(self, class_counts):         # called by run_experiment before training
        h = self.prop.in_features
        for col, n in class_counts.items():
            self.aux[col] = nn.Linear(h, n)

    def _trunk(self, x, mask):
        return self.trunk(masked_mean(x, mask))

    def forward(self, x, mask):                      # eval/metric uses only this
        return self.prop(self._trunk(x, mask))

    def auxiliary_loss(self, x, mask, aux_labels):   # training-only multi-task signal
        h = self._trunk(x, mask)
        return sum(self.ce(self.aux[c](h), lab) for c, lab in aux_labels.items())


def build_model(embed_dim, n_targets, cfg):
    return Fusion(embed_dim, n_targets, cfg.get("hidden", 256), cfg.get("dropout", 0.3))
