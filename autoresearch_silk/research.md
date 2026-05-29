# Research goal — predict silk fiber mechanics from sequence

You are an autonomous ML research agent. **Maximize the mean test R²** for predicting four
mechanical properties of silk fibers **directly from the protein sequence**:

> targets = [`toughness`, `E` (modulus), `strength`, `strain`]   →   metric = **mean of the 4 test R²**

This follows Karpathy's *autoresearch* loop: **propose → run → measure → ratchet**.

## The loop (repeat indefinitely)
1. **Read** `model.py` (the architecture) and `leaderboard.md` / `journal.md` (what's been tried).
2. **Form a hypothesis** for a change likely to raise mean test R².
3. **Edit `model.py`** (and/or add keys to `config.json`). Keep the contract in `model.py`'s header.
4. **Run** `python run_experiment.py --tag "<short idea>"`. It prints `SCALAR  mean test R² = …`.
5. **Ratchet:** if the scalar improved over the best in `leaderboard.md`, keep the change and append a
   note to `journal.md` (what you tried + result + why you think it helped). If it did **not** improve,
   revert `model.py` and record the negative result in `journal.md` (negative results are valuable —
   don't repeat them).
6. Go to 1.

## Rules
- **Only edit `model.py` and `config.json`.** Do **not** modify `dataio.py`, `run_experiment.py`, the
  data, or the cache — the metric and splits must stay fixed and comparable.
- **No test peeking** beyond the single scalar the harness reports. The harness already early-stops on
  a *grouped* validation split (see "the catch"); do your model selection via that, not the test set.
- Keep each run reasonably fast (aim ≤ a few minutes) so you can iterate a lot. If an idea needs the
  backbone fine-tuned, see `baselines/lora_finetune.py` (slower, separate path).
- One change per run when possible, so you know what moved the metric.

## How the data/harness works
- `setup.py` cached **ESMC per-residue embeddings** for every sequence (so experiments are fast). Your
  model receives padded per-residue embeddings `x:(B,Lmax,d)` + `mask:(B,Lmax)` and outputs `(B,4)`.
- Targets are standardized for a balanced 4-task MSE; R² is computed on raw units (scale-invariant).
- **Switch backbone** (stronger features) by editing `config.json`'s `esmc_model` and re-running
  `python setup.py --model <hf_id> [--device cuda]` (e.g. `biohub/ESMC-600M` or `biohub/ESMC-6B` on a GPU).

## The catch (why this is hard — read this)
There are ~3170 distinct sequences but only ~268 distinct measured-property tuples: **many sequences
share one fiber measurement** (properties are per-fiber/species; sequences are individual spidroins).
So the sequence→mechanics signal is weak and indirect. The validation split is **grouped by property
tuple** so identical-label fibers don't leak across train/val. Architectures that capture **repeat /
motif structure** (silk is highly repetitive: poly-A crystallites, GPGXX/GGX, GAGAGS) or that pool
the sequence cleverly are more likely to generalize than naive mean-pooling. Beware: it is easy to
"improve" by overfitting the small test set — prefer changes that also raise the grouped val R².

## Ideas (starting points — invent your own!)
Pooling (attention/gated/GeM/mean+max+std) · Conv1D / dilated convs over residues (motif detectors) ·
small Transformer / set-transformer · per-target heads vs shared trunk · log-transform skewed targets
(toughness, E) · uncertainty-weighted multi-task loss · concat composition/length/motif-count features ·
stronger backbone (600M/6B) · SAE features · LoRA fine-tuning (`baselines/`). Drop-in alternatives to
copy into `model.py` are in `baselines/`.

## Reference measurements (calibration — read before optimizing)
Measured on this exact split with ESMC-300M **mean-pooled** features:

| approach | mean test R² |
|----------|----:|
| predict the global mean | ~0.00 (by definition) |
| **mean-pool → MLP (seeded baseline)** | **~0.01** |
| Ridge (mean-pool features) | −0.12 … −0.00 (only ≈0 at very high regularization) |
| RandomForest (mean-pool) | −0.04 |
| `category1` (silk-type) mean predictor | −0.015 |
| Ridge, grouped 5-fold CV (honest generalization) | −0.32 |

**So the bar is brutal: nothing beats predicting the mean yet.** Note 173/175 test property-tuples
also appear in train, yet classical models still go *negative* — the single-sequence→fiber-mechanics
signal is extremely weak, and the train/test split looks distribution-shifted. **Any clearly positive
mean test R² is a genuine result.** Don't be fooled by tiny positive numbers (~0.01) — they're just
"predict the mean" noise; aim for a robust, repeatable gain that also shows up in grouped-val R².

**Most promising levers** (in rough order of expected payoff):
1. **LoRA fine-tuning** of ESMC end-to-end (`baselines/lora_finetune.py`) — lets the backbone
   specialize; the cached-embedding head search may be capped near 0.
2. **Stronger backbone**: `ESMC-600M` / `ESMC-6B` (edit `config.json` + re-run `setup.py`,
   ideally on a GPU). Bigger representations may expose signal mean-pool 300M can't.
3. **Per-residue sequence models** (Conv/attention/transformer) that read repeat/motif structure.
4. **Reframing**: per-target heads, log-targets, predicting family/category as an auxiliary task,
   or per-family residuals; robuster losses; ensembling.

Beat the top row of `leaderboard.md`. If after a real effort nothing clearly beats ~0, that
*itself* is a finding (fiber mechanics may need more than one spidroin sequence) — record it.
