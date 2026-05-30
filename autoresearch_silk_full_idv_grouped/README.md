# `autoresearch_silk_full_idv_grouped`

Autonomous architecture search for predicting dragline silk fiber mechanics from **sets of spidroin
sequences** associated with one Silkome `idv`.

Default dataset:

```text
lamm-mit/silkome-full-idv-grouped
```

Each sample is:

```python
{
    "sequences": [seq_1, seq_2, ...],
    "sequence_categories": [cat_1, cat_2, ...],
    "targets": [toughness, E, strength, strain],
}
```

The setup embeds each sequence independently with ESMC, caches the mean embedding for every sequence,
and lets `model.py` aggregate the set.

## Why This Problem

The original row-level task asks one spidroin sequence to predict a fiber-level measurement. That is a
weakly supervised proxy. This grouped version is biologically cleaner: a measured dragline fiber is
associated with multiple curated spidroin sequences from the same `idv`, so the model can learn how
sequence components combine.

Important caveat: `silkome-full-idv-grouped` provides transcriptome-associated spidroin context. It
is not guaranteed to be the exact proteomic composition physically incorporated into each dragline
fiber.

## Quick Start

```bash
cd autoresearch_silk_full_idv_grouped
conda activate esm
python setup.py --smoke-test
python setup.py
python run_experiment.py --tag baseline
```

Then start a coding-agent loop using the instructions in `program.md`.

## Kick Off Codex

Open a Codex session in this folder:

```bash
cd  autoresearch_silk_full_idv_grouped
codex --yolo
```

Paste this prompt:

```text
You are an autonomous ML research agent; this folder is your entire workspace.

GOAL: maximize the mean test R2 for predicting four dragline silk fiber properties
[toughness, E, strength, strain] from a set of spidroin sequences associated with one Silkome idv.

FIRST, read program.md, then journal.md and leaderboard.md. Understand the fixed benchmark before
editing anything.

DATA/INPUT: setup.py embeds each protein sequence independently with ESMC. The model receives:
  seq_embeddings: (B, Smax, d)
  seq_mask:       (B, Smax)
  category_ids:   (B, Smax)
  seq_lengths:    (B, Smax)
Do not feed sequence_concat_x25, sequence, FASTA text, or any concatenated multi-protein string into
ESMC. The correct formulation is set-of-sequences -> fiber properties.

ONE-TIME: if cache/ lacks the grouped ESMC arrays, run:
  python setup.py --smoke-test
  python setup.py
Then establish the starting score:
  python run_experiment.py --tag baseline

THEN loop:
  1. Form ONE hypothesis likely to improve set aggregation or prediction.
  2. Edit only model.py and optionally config.json. Keep the forward contract:
       forward(seq_embeddings, seq_mask, category_ids, seq_lengths) -> (B, 4)
  3. Commit before running:
       git add model.py config.json && git commit -m "exp NNN: <idea>"
       git tag autoresearch-exp/grouped-NNN
  4. Run:
       python run_experiment.py --tag "<idea>"
  5. If mean test R2 improves, record why in journal.md and keep it.
     If it does not improve, revert model.py/config.json to the last good architecture, record the
     negative result in journal.md, and continue.

Explore category-aware pooling, attention pooling, DeepSets, set transformers, per-category experts,
length-aware pooling, mean+max+std pooling, per-target heads, or stronger ESMC backbones. Keep the
dataset, splits, target definition, and metric fixed unless explicitly asked.
```

## Input Format

Do **not** feed `sequence_concat_x25` or `sequence` from the grouped dataset into ESMC as one long
string. Those fields exist only for simple text baselines and inspection.

Correct ESMC workflow:

```text
ESMC(seq_1) -> embedding_1
ESMC(seq_2) -> embedding_2
...
aggregate {embedding_i} -> property head
```

This is what `setup.py` does. The cached arrays are:

```text
cache/<dataset>__<model>_{train,test}_seq_mean.npy      # (N, Smax, d)
cache/<dataset>__<model>_{train,test}_seq_mask.npy      # (N, Smax)
cache/<dataset>__<model>_{train,test}_category_ids.npy  # (N, Smax)
cache/<dataset>__<model>_{train,test}_seq_lengths.npy   # (N, Smax)
```

## Targets

Default target mode is raw physical values:

```text
toughness, E, strength, strain
```

The harness standardizes targets on the training split before MSE optimization and reports R2 in the
chosen target space. Use raw targets for physical interpretability. `target_mode: "norm"` is available
for experiments using `toughnessNorm`, `ENorm`, `strengthNorm`, and `strainNorm`.

## Files

| file | role |
|---|---|
| `program.md` | autonomous-agent brief |
| `setup.py` | load grouped HF dataset and cache independent ESMC sequence embeddings |
| `dataio.py` | grouped data loader, target scaler, grouped val split, R2 |
| `model.py` | editable set aggregation model |
| `run_experiment.py` | fixed train/eval/ledger harness |
| `config.json` | dataset, backbone, target mode, hyperparameters |
| `ledger.jsonl`, `leaderboard.md`, `journal.md` | experiment trace |

## Starting Baseline

`model.py` starts as:

```text
per-sequence ESMC mean embeddings -> masked mean over sequences -> MLP
```

Useful directions for autoresearch:

- category-aware pooling with learned `sequence_categories` embeddings
- attention pooling / DeepSets / set transformer over sequence embeddings
- per-category experts for MaSp, MiSp, AcSp, Flag, PySp, etc.
- length-aware pooling
- per-target heads
- stronger ESMC backbone via `config.json`
