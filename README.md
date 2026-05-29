# Protein Language Models for Structural Proteins — ESMC, ESMFold2 & SAEs

Example notebooks that use the **EvolutionaryScale / Biohub** protein models to explore
**silk and other structural / biomaterials proteins** (spider dragline, silkworm fibroin, collagen,
elastin, resilin, keratin), with well-folded globular proteins as controls.

| Notebook | What it does | Open in Colab |
|----------|--------------|---------------|
| **`ESM_structural_proteins.ipynb`** | ESMC embeddings, masked-language modeling, in-silico mutational scanning, attention, **ESMFold2** structure prediction, a transfer-learning head, and an interactive relatedness graph | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lamm-mit/EvolutionaryScale-protein-mechanics/blob/main/ESM_structural_proteins.ipynb) |
| **`ESM_sae_structural_proteins.ipynb`** | **Sparse-autoencoder (SAE)** feature interpretation of ESMC-6B — *what concepts* the model uses for each material | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lamm-mit/EvolutionaryScale-protein-mechanics/blob/main/ESM_sae_structural_proteins.ipynb) |

---

## 1 · The models

| Model | Type | Size | Role here |
|-------|------|-----:|-----------|
| **ESMC** (`biohub/ESMC-300M / 600M / 6B`) | Masked **protein language model** (transformer; rotary, SwiGLU, pre-LN) | 0.3–6 B | Per-residue & whole-protein **embeddings**, masked-residue prediction, zero-shot variant scoring |
| **ESMFold2 / ESMFold2-Fast** (`biohub/ESMFold2*`) | **Structure prediction** (diffusion trunk on a frozen ESMC-6B backbone) | 0.2 B + 6 B backbone | All-atom 3-D structure + confidence (**pLDDT, pTM**) from a single sequence |
| **ESMC-6B SAE** (`biohub/ESMC-6B-sae-…`) | **Sparse autoencoder** on layer-60 embeddings | 16,384 features, k=64 | Decomposes embeddings into **interpretable features** with text descriptions |

**Protein language models (pLMs)** are trained by masking residues and predicting them across
hundreds of millions of natural sequences. In doing so they internalize the statistical "grammar" of
proteins — which residues co-occur, which substitutions are tolerated, what folds look like — and
expose it as a dense per-residue **embedding**. **ESMFold2** turns sequence into 3-D coordinates;
its per-residue **pLDDT** confidence (0–1 here) is itself informative. A **sparse autoencoder**
re-expresses each polysemantic embedding as a handful of approximately *monosemantic* features, many
of which correspond to recognizable biological concepts.

---

## 2 · The science: why structural proteins?

Silk, collagen, elastin, and resilin are **biomaterials**: their mechanics emerge from short sequence
**motifs** tiled into long repeats. That makes them an ideal probe for *what a protein language model
has actually learned*.

| Material | Signature motif | Structural role |
|----------|-----------------|-----------------|
| Spider dragline (MaSp1) | poly-Ala `(A)n` + `GGX` | β-sheet nanocrystals + amorphous matrix |
| Spider dragline (MaSp2) | `GPGGY`/`GPGQQ` + poly-Ala | elastic β-spirals + crystals |
| Silkworm fibroin | `GAGAGS` | antiparallel β-sheet crystallites |
| Collagen | `Gly-X-Y` (often `GPP`) | triple helix |
| Elastin | `VPGVG` | elastomeric β-turns |
| Resilin | `PSDSYGAP` | near-perfect rubber elasticity |

A recurring theme: silks are **intrinsically disordered in solution** and only order into β-sheet
nanocrystals upon spinning — so a single-chain structure predictor *should* report low confidence,
and an SAE *should* flag them as low-complexity/disordered. Both happen (see below).

---

## 3 · Setup

A dedicated conda environment (keeps the patched `transformers` away from your other envs):

```bash
conda create -n esm python=3.12 -y
conda activate esm
pip install "esm @ git+https://github.com/Biohub/esm.git@c94ed8d" jupyter ipykernel py3Dmol scikit-learn matplotlib
python -m ipykernel install --user --name esm --display-name "Python (esm — ESMC/ESMFold2)"
```

Then open either notebook and select the **Python (esm — ESMC/ESMFold2)** kernel.

- The `esm` SDK installs a **patched `transformers`** that registers the `esmc` and `esmfold2`
  architectures (stock transformers does not recognize them).
- **Device:** code auto-detects CUDA → Apple-Silicon MPS → CPU. ESMC-300M runs anywhere. **ESMFold2
  and the ESMC-6B SAE run on CPU in float32** on Apple Silicon (MPS lacks a couple of required ops);
  this is fine with enough RAM (developed on a 128 GB Mac).
- **Downloads (cached after first use):** ESMC-300M ≈ 1.3 GB, ESMC-6B ≈ 25 GB, ESMFold2-Fast ≈ 0.8 GB,
  SAE layer-60 ≈ a few hundred MB.

---

## 4 · Notebook 1 — `ESM_structural_proteins.ipynb`

Sections: embeddings → embedding-space map → masked-LM → naturalness → mutational scan → attention →
**ESMFold2 folding** → transfer-learning head → interactive graph. Every figure is saved to
[`results/`](results/) as PNG **and** SVG.

**Embedding space (PCA).** Mean ESMC embeddings of ~30 proteins; silks, other-structural, and
globular proteins separate without any supervision.

![PCA of embeddings](results/04_embedding_pca.png)

**Which family does ESMC model best?** Masked-residue recovery across families — repetitive spider
silk is almost perfectly predictable (**top-1 ≈ 0.98**), diverse collagen the least (**≈ 0.62**).

![Masked-LM family benchmark](results/05_masked_family_benchmark.png)

**In-silico deep mutational scan.** Zero-shot `log P(mut) − log P(wt)` across a MaSp1 window — the
poly-alanine crystalline block is strongly constrained.

![Mutational scan](results/07_mutational_scan_masp1.png)

**Attention — long-range structure.** Averaged last-layer attention is mostly local; *individual
heads in a middle layer* on a folded protein (ubiquitin) reveal off-diagonal (long-range) contacts.

![Attention heads](results/08b_attention_heads_ubiquitin.png)

**ESMFold2 structure confidence.** Ubiquitin folds confidently (**pLDDT ≈ 0.82**); a silk peptide
does not (**≈ 0.41**) — a *feature of the biology*, since dragline silk is disordered until spun.

![Per-residue pLDDT](results/09_plddt_per_residue.png)

**Transfer learning.** A new head trained on *frozen* ESMC embeddings classifies material family;
evaluated honestly with 5-fold cross-validation on UniProt data augmented by sliding windows
(**CV accuracy ≈ 0.94 ± 0.02**, chance = 0.14).

![Transfer-learning head](results/10_transfer_head.png)

**Interactive relatedness graph.** Proteins linked by ESMC-embedding similarity; live threshold /
k-NN / highlight controls in Jupyter.

![Sequence relatedness graph](results/11_sequence_graph.png)

*(Also in `results/`: cosine-similarity heatmap `04_…`, per-residue surprisal `06_…`, and the
all-heads/last-layer attention overview `08_…`.)*

---

## 5 · Notebook 2 — `ESM_sae_structural_proteins.ipynb`

Decomposes ESMC-6B's layer-60 embeddings into **16,384 sparse features** (top-64 active per residue)
and asks what each structural protein "uses". Adapted from Biohub's SAE cookbook tutorial but rebuilt
to run **locally** (the original uses the Forge API). Figures in [`results_sae/`](results_sae/).

**Motif-localized features.** The strongest features for a spider-silk sequence, plotted along the
chain — sharp peaks track specific motifs (poly-Ala crystals vs. `GPGXX` turns).

![Motif features](results_sae/A_motif_features_masp2.png)

**Order vs. disorder.** Activation mass carried by *disorder / low-complexity / repeat* features per
protein — silks, elastin, and resilin score high; folded globular proteins score low. The SAE
recovers a biologically meaningful order/disorder axis.

![Disorder score](results_sae/B_disorder_score.png)

**Feature fingerprints across families.** Proteins clustered by the cosine similarity of their
16,384-feature fingerprints; silks group together and separate from globular controls.

![Fingerprint similarity](results_sae/D_fingerprint_similarity.png)

**Highlights from the auto-generated feature descriptions** (treat as hypotheses):
- Spider silk top features → *"Gly/Ser-rich disordered low-complexity regions"*, *"Secreted
  low-complexity adhesive repeats"*.
- **Collagen's unique feature → *"Collagen Gly–X–Y repeat detector"*** — the SAE literally has a
  collagen-repeat detector.
- Elastin's unique feature → *"Solenoid repeat register detector"*; ubiquitin → only generic features.

The notebook also **maps a feature onto a 3-D ESMFold2 structure** (local) and provides an interactive
protein × feature explorer.

---

## 6 · Repository layout

```
EvolutionaryScale-protein-mechanics/      # (local working folder: ESM_samples/)
├── ESM_structural_proteins.ipynb        # Notebook 1 (executed, with outputs)
├── ESM_sae_structural_proteins.ipynb    # Notebook 2 (executed, with outputs)
├── build_notebook.py                     # generator for Notebook 1 (clean-slate; overwrites the .ipynb)
├── build_sae_notebook.py                 # generator for Notebook 2
├── results/                              # Notebook 1 figures (PNG + SVG)
├── results_sae/                          # Notebook 2 figures (PNG + SVG)
└── README.md
```

> The `build_*.py` scripts regenerate the notebooks from scratch (no personal edits). If you've edited
> a notebook by hand, don't re-run its generator — it will overwrite your changes.

---

## 7 · Caveats and Notes

- pLMs/SAE outputs are **in-silico hypotheses**, not wet-lab annotations; SAE feature descriptions are
  auto-generated and can be wrong, especially for rare biology.
- ESMC has a **2048-residue** context window; long proteins (spidroins, fibroin) are sliced.
- The transfer-learning demo uses *stratified* CV; for a stricter test, split by source protein before
  windowing (group-aware CV) so windows from one protein don't straddle train/test.

## 8 · References

- ESMC / ESMFold2 model cards: <https://huggingface.co/biohub/ESMC-6B>, <https://huggingface.co/biohub/ESMFold2>
- ESMC SAE cards & ESM Atlas (feature descriptions): <https://huggingface.co/biohub>
- Candido et al., *Language Modeling Materializes a World Model of Protein Biology* (2026).
- Biohub `esm` SDK: <https://github.com/Biohub/esm>
