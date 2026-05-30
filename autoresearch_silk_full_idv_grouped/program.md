# Research Goal - Predict Fiber Mechanics From Sets Of Spidroin Sequences

You are an autonomous ML research agent. Maximize the mean test R2 for predicting four dragline silk
fiber properties from all available spidroin sequences associated with one Silkome `idv`:

```text
targets = [toughness, E, strength, strain]
metric = mean test R2 over the four targets
```

This is a grouped problem:

```text
{sequence_1, sequence_2, ..., sequence_n} -> fiber-level properties
```

## Set Up

```bash
python setup.py --smoke-test
python setup.py
python run_experiment.py --tag baseline
```

The default dataset is `lamm-mit/silkome-full-idv-grouped`.

## Critical Input Rule

Do not use `sequence_concat_x25` as the ESMC input. Each protein sequence must be embedded
independently. The setup already does this and caches:

```text
seq_embeddings: (N_groups, max_sequences, embed_dim)
seq_mask:       (N_groups, max_sequences)
category_ids:   (N_groups, max_sequences)
seq_lengths:    (N_groups, max_sequences)
```

`model.py` receives batches in this form:

```python
forward(seq_embeddings, seq_mask, category_ids, seq_lengths) -> (B, 4)
```

## The Loop

1. Read `model.py`, `leaderboard.md`, and `journal.md`.
2. Form one hypothesis likely to improve grouped set aggregation.
3. Edit `model.py` and optionally `config.json`.
4. Commit before running so the logged commit points at the tested code.
5. Run:

   ```bash
   python run_experiment.py --tag "<idea>"
   ```

6. If it improves the best mean test R2, keep it and record why in `journal.md`.
7. If not, revert `model.py` / `config.json` to the last good architecture and record the negative
   result.
8. Repeat.

## Rules

- Keep the dataset, splits, target definitions, and metric fixed unless explicitly asked.
- Prefer editing only `model.py` and `config.json`.
- Do not edit `setup.py`, `dataio.py`, `run_experiment.py`, `data/`, or `cache/` during comparable
  architecture search.
- No direct test-set tuning beyond the one scalar reported by the harness.

## What To Explore

The starting baseline is:

```text
ESMC mean embedding per sequence -> masked mean over sequences -> MLP
```

Promising directions:

- learned attention pooling over the set of sequence embeddings
- DeepSets or a small set transformer
- category-aware models using `category_ids`
- per-category experts or gates for MaSp, MiSp, AcSp, Flag, PySp, etc.
- length-aware pooling using `seq_lengths`
- mean + max + std pooling over the sequence set
- per-target heads or uncertainty-weighted multitask losses
- target transforms through `config.json`
- stronger ESMC backbone, then rerun `setup.py`

## Target Mode

Default is `target_mode: "raw"`:

```text
toughness, E, strength, strain
```

The harness standardizes targets internally before training. Use raw targets for the main benchmark.
`target_mode: "norm"` is available only if you intentionally want to work in the Silkome normalized
target space.

## Caveat

`silkome-full-idv-grouped` is broader sequence context, not verified fiber proteomics. It can help the
model learn component integration, but the labels remain fiber-level measurements affected by spinning,
environment, structure, and other unobserved factors.
