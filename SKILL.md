---
name: esm-protein-mechanics
description: >
  Local command-line tools for structural-protein analysis with EvolutionaryScale / Biohub
  models (ESMC, ESMFold2, ESMC-6B sparse autoencoders). Fold sequences to PDB/mmCIF, compute
  embeddings, run zero-shot deep mutational scans, extract interpretable SAE features, and train /
  apply a lightweight property classifier on frozen ESMC embeddings. Use when a task involves
  protein structure prediction, protein language-model embeddings, variant-effect scoring, SAE
  feature interpretation, or training a predictor on protein sequences — especially for silk and
  other structural/biomaterials proteins. Everything runs locally; no API key.
---

# ESM Protein-Mechanics CLI toolkit

A set of self-contained command-line tools wrapping the **ESMC** protein language model,
**ESMFold2** structure predictor, and the **ESMC-6B sparse autoencoder (SAE)**. All run **locally**
(no Biohub API key / no Forge). Tools live in `cli/` and share `cli/esm_common.py`.

## 1 · Installation (required before any tool runs)

These models need a **patched `transformers`** that ships with EvolutionaryScale's `esm` SDK. Install
into a dedicated conda env so it never collides with a stock `transformers`:

```bash
conda create -n esm python=3.12 -y
conda activate esm
pip install "esm @ git+https://github.com/Biohub/esm.git@c94ed8d" \
            jupyter ipykernel py3Dmol scikit-learn matplotlib pandas joblib datasets pyarrow peft
```

Verify:
```bash
python -c "import torch, esm, transformers; print('ok', transformers.__version__)"
```

Run every tool **inside this env** (`conda activate esm`). For non-interactive/agent use, call the
env interpreter directly, e.g. `"$(conda info --base)/envs/esm/bin/python" cli/esm_fold.py ...`.

## 2 · Environment notes (read before running)

- **Device** is auto-detected: CUDA → Apple-Silicon MPS → CPU. Light models (`ESMC-300M/600M`) use
  MPS/GPU. **Heavy paths (ESMFold2, ESMC-6B, the SAE) run on CPU in float32** on Apple Silicon —
  MPS lacks a couple of required ops. Pass `--device cuda` on a GPU box for big speedups.
- **No API key.** Feature *descriptions* in `esm_sae` come from a keyless public endpoint
  (`biohub.ai/.../features/<id>`); skip them with `--no-describe` if offline.
- **First-run downloads (cached afterwards):** ESMC-300M ≈ 1.3 GB, ESMC-6B ≈ 25 GB,
  ESMFold2-Fast ≈ 0.8 GB (+ its 6B backbone), SAE layer ≈ a few hundred MB. Ensure disk + RAM.
- Each tool has `-h/--help`.

## 3 · Tools

| Tool | Purpose | Model | Typical time |
|------|---------|-------|--------------|
| `esm_fold.py` | sequence → 3-D structure (PDB/mmCIF) + pLDDT/pTM | ESMFold2 (+6B) | ~10–20 s/protein (CPU) |
| `esm_embed.py` | sequence → embedding vectors (.npy/.csv) | ESMC-300M | <1 s/protein |
| `esm_mutscan.py` | zero-shot deep mutational scan (LLR matrix) | ESMC-300M | seconds/window |
| `esm_sae.py` | interpretable SAE features + descriptions | ESMC-6B + SAE | ~1 min load, ~2 s/protein |
| `make_dataset.py` | build/push the structural-protein dataset | — (UniProt) | seconds |
| `esm_train_head.py` | train a family/property head on frozen embeddings | ESMC-300M | ~1 min |
| `esm_predict.py` | apply a trained head to new sequences | ESMC-300M | <1 s/protein |

### esm_fold.py — structure prediction
```bash
python cli/esm_fold.py --seq MQIFVKTLTGKT... --out folds --formats pdb,cif
python cli/esm_fold.py --fasta proteins.fasta --out folds --steps 50 --loops 3
```
Writes `<out>/<name>.pdb` and/or `.cif` per record + `<out>/fold_summary.csv` (pLDDT, pTM, length).
Options: `--model {biohub/ESMFold2-Fast,biohub/ESMFold2}`, `--steps`, `--loops`, `--seed`, `--device`.

### esm_embed.py — embeddings
```bash
python cli/esm_embed.py --fasta proteins.fasta --pool mean --out emb   # emb.npy (N×d) + emb.csv
python cli/esm_embed.py --seq MQIF... --pool none --out perres         # per-residue (L×d) .npy
```
`--model {biohub/ESMC-300M,biohub/ESMC-600M,biohub/ESMC-6B}`.

### esm_mutscan.py — zero-shot deep mutational scan
```bash
python cli/esm_mutscan.py --seq GGAGQGG...AAAAAAAA --out scan --plot scan.png
python cli/esm_mutscan.py --fasta one.fasta --start 20 --end 45
```
CSV of `log P(mut) − log P(wt)` (20 AA × window); negative = disfavored. `--start/--end` set the window.

### esm_sae.py — interpretable features (ESMC-6B SAE, local)
```bash
python cli/esm_sae.py --seq GGAGQGG... --topk 10 --out sae          # sae_<name>.csv with descriptions
python cli/esm_sae.py --fasta proteins.fasta --rank prevalence --save-matrix --no-describe
```
CSV per sequence: `feature_id, max_activation, prevalence, label, category, summary`. `--save-matrix`
also dumps the `(L × 16384)` activations. `--rank {max,prevalence}`, `--layer 60`.

### make_dataset.py — build / push the dataset
```bash
python cli/make_dataset.py --per-family 40 --out data/structural_protein_families.csv
python cli/make_dataset.py --push --repo lamm-mit/structural-protein-families   # needs HF login
```
Fetches structural-protein families from UniProt, windows them, writes a CSV locally, and (with
`--push`) uploads via the `datasets` library as **parquet** so the HF dataset viewer works. Public
dataset (loadable with `datasets.load_dataset`):
<https://huggingface.co/datasets/lamm-mit/structural-protein-families>.

### esm_train_head.py — train a predictor for ANY property on frozen embeddings
```bash
python cli/esm_train_head.py --out-dir head_model                       # default HF dataset, 5-fold CV
python cli/esm_train_head.py --train train.csv --test test.csv --seq-column sequence --target family
python cli/esm_train_head.py --train data.csv --seq-column seq --target tm   # numeric -> regression
python cli/esm_train_head.py --train myorg/my-dataset --target solubility --head mlp
python cli/esm_train_head.py --method lora --target family --lora-r 16 --epochs 4   # LoRA fine-tune
```
Predicts **any** target: the task is auto-detected from the target column's dtype —
**categorical → classification**, **numeric → regression** (override with `--task`). Inputs: a
sequence column (`--seq-column`) and a target column (`--target`/`--label`); one dataset
(cross-validated via `--cv`) or an explicit `--train`/`--test` split, each a **local CSV or a HF
dataset repo id** (use `repo:split`; a HF `--train` with a test/validation split auto-supplies the
test set). Two methods via **`--method`**:
- **`head`** (default) — ESMC **frozen**; a small head on mean-pooled embeddings
  (`--head {linear,mlp}`; linear = LogisticRegression / Ridge). Saves `head.joblib`/`head.pt`.
- **`lora`** — **LoRA adapters** added to ESMC and fine-tuned end-to-end (PEFT, needs `peft`).
  Tune `--lora-r/--lora-alpha/--lora-dropout/--lora-target-modules` (ESMC-card defaults
  `layernorm_qkv.1,out_proj,ffn.1,ffn.3`), `--epochs`, `--lr`, `--max-length`. Saves the adapter +
  tokenizer; only ~0.4 % of weights train.

Both save `meta.json` to `--out-dir` and print CV/held-out accuracy (classification) or R²/MAE
(regression). `esm_predict.py` auto-detects the method/task from `meta.json`.

### esm_predict.py — apply a trained head
```bash
python cli/esm_predict.py --model-dir head_model --seq GPGGYGPGQQ... 
python cli/esm_predict.py --model-dir head_model --fasta unknowns.fasta --out preds.csv
```
Loads `meta.json`, re-embeds with the same ESMC model, and prints — depending on the trained task —
either the predicted **label + top-k probabilities** (classification) or the predicted **numeric
value** (regression). `--out` writes a predictions CSV.

## 4 · Recipes

- **Fold a FASTA and rank by confidence:** `esm_fold.py --fasta x.fasta` → sort `fold_summary.csv` by pTM.
- **Score a designed mutation:** `esm_mutscan.py` over the window; read the mutant row/column LLR.
- **Interpret what the model "sees":** `esm_sae.py --topk 15` → inspect `label`/`category` per feature.
- **Train then classify unknowns:** `esm_train_head.py` → `esm_predict.py --model-dir ...`.

## 5 · Gotchas
- Always `conda activate esm` first; the stock `transformers` will NOT load `esmc`/`esmfold2`.
- ESMFold2 / SAE will be slow on CPU and need RAM (the 6B backbone is ~25 GB). Use a GPU if available.
- Sequences must be standard 20 amino acids; tools validate and error on others. ESMC context ≤ 2048.
