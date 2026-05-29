"""========================  THE EDITABLE ASSET  ========================
This is the ONE file the autoresearch agent rewrites to improve the score.
Define how per-residue ESMC embeddings of a silk sequence map to the 4 mechanical
targets [toughness, E, strength, strain]. Then run:  python run_experiment.py

CONTRACT (keep these stable so run_experiment.py works):
  build_model(embed_dim: int, n_targets: int, cfg: dict) -> torch.nn.Module
  the returned module's forward(x, mask) -> (B, n_targets), where
      x    : (B, Lmax, embed_dim) float32   padded per-residue ESMC embeddings
      mask : (B, Lmax) bool                 True = real residue, False = padding
  Optionally set DESCRIPTION (a short string) — it is logged to the ledger.

You may use `mask` to pool however you like (mean/max/attention), or ignore the
sequence dimension entirely. You may also read hyper-parameters from `cfg`
(config.json) — add your own keys there.

OPTIONAL multi-task / FUSION hook (supported by run_experiment.py): to also train
auxiliary classifiers from the sequence (e.g. taxonomy) and fuse them, declare on
your module:
  AUX_COLS = ["family", "genus", "category1"]        # meta columns from data/*.parquet
  def build_aux_heads(self, class_counts): ...        # called once with {col: n_classes}
  def auxiliary_loss(self, x, mask, aux_labels): ...   # scalar added to loss in TRAINING only
Eval/metric stay forward()->4 targets, so it remains sequence-only. See
`baselines/fusion_taxonomy.py`. Tune `aux_weight` in config.json.

IDEAS TO EXPLORE (go beyond these — the agent chooses what, if anything, to try):
  * pooling: masked-mean (baseline), masked-max, attention/gated pooling, [CLS]-style query,
    GeM pooling, mean+max+std concat, learned soft-bins over the sequence.
  * sequence models over residues: Conv1D stacks (multi-scale motif detectors — silk is
    repetitive!), dilated convs, a small Transformer/Performer, BiLSTM, set-transformer.
  * heads: shared trunk + per-target heads vs one multi-task head; deeper/wider; residual MLP.
  * targets: predict log(toughness)/log(E) (skewed); uncertainty weighting of the 4-task loss.
  * fusion / multi-task: taxonomy aux heads (hook above); two-stage (pretrain classifier, fuse).
  * sequence patterns: `dataio.sequence_motif_features(seqs)` (composition + silk-motif counts);
    or GPT/ESM surprisal to flag key residues; fuse with the embedding pool.
  * regularization: dropout, weight decay, mixup over pooled features.
  * bigger backbone: edit config.json `esmc_model` (600M/6B) then re-run setup.py.
  * harder mode: LoRA-fine-tune ESMC end-to-end (see baselines/lora_finetune.py).

REMEMBER THE CHALLENGE: many distinct sequences share ONE fiber measurement, so the signal is
weak/indirect (mean-pool/Ridge/category-mean all sit at R²≈0 — see program.md). Any clearly
positive, repeatable mean test R² is a real result.
====================================================================="""
import torch
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
    return MeanPoolMLP(embed_dim, n_targets,
                       hidden=cfg.get("hidden", 256),
                       dropout=cfg.get("dropout", 0.3))
