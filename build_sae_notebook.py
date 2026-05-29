"""Builds ESM_sae_structural_proteins.ipynb — interpreting ESMC-6B Sparse-Autoencoder (SAE)
features for silk and other structural proteins, fully LOCAL (no Biohub API key).
Adapted from the Biohub `esmc_sae_feature_interpretation` cookbook tutorial."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):  cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

# ----------------------------------------------------------------------------- title
md(r"""# Interpreting ESMC Features in Structural Proteins (Sparse Autoencoders)

*Companion to `ESM_structural_proteins.ipynb`. Focus: silk & other structural/biomaterials proteins.
Runs fully **locally** — no Biohub API key.*

A protein language model packs a lot into each residue's embedding, but those 2560 dimensions are
**polysemantic** (each mixes many concepts). A **Sparse Autoencoder (SAE)** re-expresses every
residue embedding as a sparse combination of **16,384 interpretable features**, of which only the
**top k = 64** are active per residue. Many features turn out to be (approximately) *monosemantic* —
they fire for one recognizable concept: a structural motif, a compositional bias, a binding site, a
fold/family signal…

We use the **ESMC-6B SAE trained on layer 60** (`biohub/ESMC-6B-sae-...`) and ask, specifically for
**structural proteins**:

1. **Motif-localized features** — which features fire sharply on silk's `poly-Ala` crystals, `GPGXX`
   turns, collagen `GXY`, elastin `VPGVG`?
2. **Order vs. disorder / domain features** — which broad features separate crystalline from
   amorphous/disordered regions, or mark whole domains?
3. **Map features onto 3-D** — project activations onto an **ESMFold2**-predicted structure (local).
4. **Compare across families** — feature *fingerprints*: what is shared vs. unique across
   silk / collagen / elastin / keratin / globular, and how do proteins cluster by features?

> The official tutorial calls the Biohub **Forge API** (`ESMCForgeInferenceClient`, needs a token).
> Here we use the **local Hugging Face path**: load ESMC-6B + the SAE weights and run on this machine.
> Feature *descriptions* are fetched from a **keyless** public endpoint (`biohub.ai/.../features/<id>`).""")

# ----------------------------------------------------------------------------- setup
md(r"""## 0 · Setup

Uses the dedicated **`esm`** environment (same as the companion notebook). ESMC-6B is large
(~25 GB) and the model is written for CUDA; with no GPU we run it on **CPU in float32** — fine here
(plenty of RAM), just not instant. A forward pass for a ~150-residue protein takes a few seconds.""")

code(r"""import os, json, urllib.request, urllib.parse, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path

# ESMC-6B + diffusion-free SAE run on CPU (fp32). MPS lacks some ops for the 6B/SAE path.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| torch", torch.__version__)

RESULTS = Path("results_sae"); RESULTS.mkdir(exist_ok=True)
def savefig(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    print("saved:", RESULTS / f"{name}.png")""")

# ----------------------------------------------------------------------------- load models
md(r"""## 1 · Load ESMC-6B and the layer-60 SAE (local)

The SAE was trained to reconstruct ESMC-6B's **layer-60** hidden states. We download just that one
layer's weights (`layer_60.safetensors`, ~few hundred MB) from the multi-layer SAE repo, attach it to
the model, and from then on every forward pass also returns sparse feature activations.""")

code(r'''from transformers import AutoModel, AutoTokenizer

ESMC = "biohub/ESMC-6B"
SAE_REPO = "biohub/ESMC-6B-sae-k64-codebook16384"   # multi-layer; we grab layer 60 only
SAE_LAYER = 60

tokenizer = AutoTokenizer.from_pretrained(ESMC)
model = AutoModel.from_pretrained(ESMC).to(DEVICE).float().eval()

sae = AutoModel.from_pretrained(
    SAE_REPO,
    allow_patterns=["config.json", f"layer_{SAE_LAYER}.safetensors"],
    device=DEVICE,
)
sae.initialize_layers([SAE_LAYER])
model.add_sae_models([sae.layers[str(SAE_LAYER)]])

CODEBOOK = sae.layers[str(SAE_LAYER)].config.codebook_dim if hasattr(
    sae.layers[str(SAE_LAYER)], "config") else 16384
print(f"ESMC-6B + SAE layer {SAE_LAYER} ready on {DEVICE}: "
      f"{CODEBOOK} features, top-k=64 active per residue")''')

# ----------------------------------------------------------------------------- proteins
md(r"""## 2 · A structural-protein panel

Consensus repeat constructs make each material's motif explicit; we add real (sliced) sequences and
globular controls. We keep lengths modest so the 6B forward passes stay quick.""")

code(r'''def fetch_uniprot(acc, maxlen=180):
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "esm-class-notebook"})
    full = "".join(urllib.request.urlopen(req, timeout=30).read().decode().splitlines()[1:])
    mid = len(full) // 2
    return full[max(0, mid - maxlen // 2): mid + maxlen // 2]

PROTEINS = {
    # consensus repeats (motif grammar is explicit)
    "MaSp1 (silk consensus)":  "GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA" * 2,
    "MaSp2 (silk consensus)":  "GPGGYGPGQQGPGGYGPGQQGPSGPGSAAAAAAAA" * 2,
    "Fibroin (GAGAGS)":        "GAGAGS" * 16,
    "Collagen (GPP)":          "GPP" * 30,
    "Elastin (VPGVG)":         "VPGVG" * 18,
    "Resilin":                 "GGRPSDSYGAPGGGN" * 6,
    # globular controls
    "Ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
}
# add a couple of real ones (fetched, sliced)
for acc, name in {"P05790": "Fibroin H (real)", "P02144": "Myoglobin (real)"}.items():
    try: PROTEINS[name] = fetch_uniprot(acc)
    except Exception as e: print("fetch failed", name, e)

for n, s in PROTEINS.items():
    print(f"{n:26} {len(s):3d} aa")''')

# ----------------------------------------------------------------------------- extract
md(r"""## 3 · Extract SAE features

Each forward pass returns `sae_outputs["layer60"]`, a sparse `(L+2, 16384)` tensor (the +2 are the
BOS/EOS tokens, which we drop). Each residue keeps only its **top-64** features, so the matrix is
~99.6% zeros.""")

code(r'''@torch.inference_mode()
def sae_features(seq):
    """Return dense SAE feature matrix (L, 16384) for one sequence."""
    inputs = {k: v.to(DEVICE) for k, v in tokenizer(seq, return_tensors="pt").items()}
    out = model(**inputs)
    feats = out["sae_outputs"][f"layer{SAE_LAYER}"].to_dense().float().cpu().numpy()
    return feats[1:-1]                                   # drop BOS/EOS -> (L, 16384)

FEATS = {}
for name, seq in PROTEINS.items():
    t0 = time.time(); F = sae_features(seq); FEATS[name] = F
    print(f"{name:26} {F.shape}  active/res={ (F>0).sum(1).mean():.0f}  "
          f"sparsity={100*(F==0).mean():.1f}%  ({time.time()-t0:.1f}s)")''')

# ----------------------------------------------------------------------------- descriptions
md(r"""## 4 · Feature descriptions (keyless)

Each of the 16,384 features has an **automatically generated** description (label / summary /
category) produced by an agent that examined where the feature fires across millions of proteins.
We fetch these from a public endpoint. *Treat them as hypotheses, not ground truth.*""")

code(r'''from functools import lru_cache

@lru_cache(maxsize=8192)
def feature_info(idx):
    """Description metadata for an SAE feature (keyless public endpoint)."""
    url = f"https://biohub.ai/esm/protein/api/v1alpha1/features/{int(idx)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "esm-class-notebook"})
        return json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        return {"label": "(unavailable)", "category": None, "summary": str(e)[:60]}

def label_of(idx):
    d = feature_info(idx); return f"{d.get('label','?')}  [{d.get('category')}]"

# sanity check
for idx in list(FEATS.values())[0].max(0).argsort()[::-1][:3]:
    print(f"feature {idx:5d}: {label_of(int(idx))}")''')

# ----------------------------------------------------------------------------- A: motif-localized
md(r"""## 5 · Analysis A — motif-localized features

Features that fire as **sharp peaks** tend to mark short, specific motifs. For a spider-silk protein
we find the most strongly activating features and plot them along the sequence; the peaks should line
up with the **poly-alanine crystalline blocks** and the **glycine-rich** repeats.""")

code(r'''def plot_features_on_sequence(name, feature_ids, fname=None):
    seq = PROTEINS[name]; F = FEATS[name]
    fig, axes = plt.subplots(len(feature_ids), 1, figsize=(14, 1.9 * len(feature_ids)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, fid in zip(axes, feature_ids):
        ax.bar(np.arange(len(seq)), F[:, fid], width=1.0, color="steelblue")
        ax.set_ylabel(f"f{fid}", fontsize=9)
        ax.set_title(f"feature {fid}: {label_of(int(fid))}", fontsize=9, loc="left")
    axes[-1].set_xlabel("residue position")
    # annotate the sequence motif track on the bottom axis
    axes[-1].set_xticks(range(0, len(seq), 5))
    fig.suptitle(f"{name}  —  {seq[:40]}…", y=1.002)
    plt.tight_layout()
    if fname: savefig(fig, fname)
    plt.show()

silk = "MaSp2 (silk consensus)"
top_local = FEATS[silk].max(0).argsort()[::-1][:4]
print("Top features by max activation in", silk, "->", top_local.tolist())
plot_features_on_sequence(silk, top_local, fname="A_motif_features_masp2")''')

md(r"""Compare the peak positions with the sequence: in `…GPGGYGPGQQ…AAAAAAAA`, do some features
fire specifically on the **poly-Ala** stretch and others on the **GPGXX** turns? Re-run with
`silk = "Fibroin (GAGAGS)"` or `"Collagen (GPP)"` to see motif-specific features for other materials.""")

# ----------------------------------------------------------------------------- B: order/disorder
md(r"""## 6 · Analysis B — order/disorder & domain-level features

Features that are **prevalent** (active across many residues) tend to capture *broad* properties:
low-complexity/disorder, repeat character, or whole-domain/fold signals. We rank features by
prevalence (vs. max activation), and contrast a **disordered, repetitive silk** with a **folded
globular** protein.""")

code(r'''def top_features(name, by="prevalence", k=8, thresh=0.0):
    F = FEATS[name]
    score = (F > thresh).sum(0) if by == "prevalence" else F.max(0)
    ids = np.argsort(score)[::-1][:k]
    return [(int(i), float(score[i])) for i in ids]

for name in [silk, "Ubiquitin"]:
    print(f"\n### {name} — most PREVALENT features (broad signals)")
    for fid, sc in top_features(name, by="prevalence", k=5):
        print(f"  f{fid:5d} active@{int(sc):3d}/{len(PROTEINS[name])} res | {label_of(fid)}")''')

code(r'''# A "disorder/repeat-ness" score per protein: fraction of residues where any of the protein's
# top prevalence features is a known low-complexity/disorder/repeat feature.
def category_fraction(name, keywords=("disorder", "low-complexity", "repeat", "compositional")):
    F = FEATS[name]
    ids = np.argsort((F > 0).sum(0))[::-1][:20]          # 20 most prevalent features
    hits = [i for i in ids if any(k in (str(feature_info(int(i)).get("label","")).lower()
            + str(feature_info(int(i)).get("category","")).lower()) for k in keywords)]
    # mean activation mass carried by those features
    return float(F[:, hits].sum() / (F.sum() + 1e-9)) if hits else 0.0

names = list(PROTEINS)
frac = [category_fraction(n) for n in names]
order = np.argsort(frac)[::-1]
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar([names[i] for i in order], [frac[i] for i in order], color="indianred")
ax.set_ylabel("activation mass in\ndisorder/low-complexity/repeat features")
ax.set_title("How 'low-complexity / disordered' does the SAE find each protein?")
ax.tick_params(axis="x", rotation=35)
plt.tight_layout(); savefig(fig, "B_disorder_score"); plt.show()''')

md(r"""Silks, elastin, and resilin should score high (they *are* low-complexity / disordered until
assembled), while ubiquitin/myoglobin (folded globular) should score low — the SAE recovers a
biologically meaningful order/disorder axis. *Discuss: where does collagen fall, and why?*""")

# ----------------------------------------------------------------------------- C: 3D mapping
md(r"""## 7 · Analysis C — map a feature onto a 3-D structure (ESMFold2, local)

The tutorial colored a crystal structure from the PDB. Silk has no good folded PDB entry (it's
disordered), so we **predict** a structure locally with ESMFold2 and paint each residue by a chosen
SAE feature's activation. *(Loads ESMFold2's own 6B backbone — a second large model; guarded.)*""")

code(r'''MAP_3D = True   # set False to skip (loads ESMFold2 ~ another large model)

if MAP_3D:
    from contextlib import nullcontext
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

    map_name = "MaSp1 (silk consensus)"           # which protein to fold + paint
    map_seq = PROTEINS[map_name]
    feat_to_map = int(FEATS[map_name].max(0).argmax())   # its single strongest feature

    fold = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast")
    fold = (fold.float() if DEVICE != "cuda" else fold).to(DEVICE).eval()
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=map_seq)])
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if DEVICE == "cuda" else nullcontext())
    with torch.inference_mode(), amp:
        res = ESMFold2InputBuilder().fold(fold, spi, num_loops=2, num_sampling_steps=20,
                                          num_diffusion_samples=1, seed=0)
    print(f"folded {map_name}: pLDDT={float(res.plddt.mean()):.2f} | "
          f"painting feature {feat_to_map}: {label_of(feat_to_map)}")''')

code(r'''if MAP_3D:
    import py3Dmol
    cif = res.complex.to_mmcif()
    act = FEATS[map_name][:, feat_to_map]
    act = act / (act.max() + 1e-9)                       # normalize 0..1
    def feat_color(a):                                   # white -> red by activation
        g = int(255 * (1 - a)); return f"#ff{g:02x}{g:02x}"
    view = py3Dmol.view(width=620, height=460)
    view.addModel(cif, "mmcif"); view.setStyle({"cartoon": {"color": "white"}})
    for i, a in enumerate(act):
        view.setStyle({"resi": str(i + 1)}, {"cartoon": {"color": feat_color(float(a))}})
    view.zoomTo()
    print(f"{map_name}: residues colored by feature {feat_to_map} activation (red = high)")
    view.show()''')

# ----------------------------------------------------------------------------- D: cross-family
md(r"""## 8 · Analysis D — feature *fingerprints*: compare & cluster families

Summarize each protein by its **feature fingerprint** = the max activation of every feature across
its residues (a 16,384-vector). Proteins with similar fingerprints use similar internal concepts.
We (a) cluster proteins by fingerprint similarity, and (b) pull out the features that best
**discriminate** the materials.""")

code(r'''# fingerprints (max activation per feature), L2-normalized
fp = np.stack([FEATS[n].max(0) for n in names])          # (n_proteins, 16384)
fpn = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9)
sim = fpn @ fpn.T

from scipy.cluster.hierarchy import linkage, leaves_list
Z = linkage(fpn, method="average", metric="cosine")
order = leaves_list(Z)

fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(sim[np.ix_(order, order)], cmap="viridis")
ax.set_xticks(range(len(names))); ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels([names[i] for i in order], fontsize=8)
ax.set_title("Protein similarity by SAE feature fingerprint (cosine)")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); savefig(fig, "D_fingerprint_similarity"); plt.show()''')

code(r'''# Which features are SHARED across all proteins vs UNIQUE to one?
present = np.stack([(FEATS[n].max(0) > 1.0) for n in names])     # (n_proteins, 16384) boolean
ubiquity = present.sum(0)                                        # in how many proteins is it strong?
shared = np.where(ubiquity == len(names))[0]
unique_to = {n: np.where(present[i] & (ubiquity == 1))[0] for i, n in enumerate(names)}

print(f"features strong in ALL {len(names)} proteins (generic): {len(shared)}")
for fid in shared[:3]:
    print(f"    f{int(fid)}: {label_of(int(fid))}")
print("\nrepresentative UNIQUE features per protein:")
for n in names:
    u = unique_to[n]
    if len(u):
        fid = int(u[np.argmax(fp[names.index(n), u])])           # its strongest unique feature
        print(f"  {n:26} f{fid:5d}: {label_of(fid)}")''')

md(r"""**Read it as:** *shared* features are generic (length, common residues), while *unique* features
are candidate **family signatures** — e.g. a poly-Ala/β-crystal feature for silk, a `GXY`/collagen
feature for collagen. The clustering should group the silks together and separate the globular
controls.""")

# ----------------------------------------------------------------------------- interactive
md(r"""## 9 · Interactive: explore any protein × feature

Pick a protein and one of its top features to see where it fires and what it (allegedly) means.
*(Needs a live kernel.)*""")

code(r'''try:
    import ipywidgets as widgets

    def explore(protein, rank_by, rank):
        F = FEATS[protein]; seq = PROTEINS[protein]
        score = F.max(0) if rank_by == "max activation" else (F > 0).sum(0)
        fid = int(np.argsort(score)[::-1][rank])
        d = feature_info(fid)
        fig, ax = plt.subplots(figsize=(13, 2.6))
        ax.bar(np.arange(len(seq)), F[:, fid], width=1.0, color="darkorange")
        ax.set_title(f"{protein} | feature {fid} (rank {rank} by {rank_by})\n"
                     f"{d.get('label')}  [{d.get('category')}]", fontsize=10, loc="left")
        ax.set_xlabel("residue position"); ax.set_ylabel("activation")
        plt.tight_layout(); plt.show()
        print("summary:", d.get("summary"))

    widgets.interact(
        explore,
        protein=widgets.Dropdown(options=list(PROTEINS), value=silk),
        rank_by=widgets.Dropdown(options=["max activation", "prevalence"], value="max activation"),
        rank=widgets.IntSlider(min=0, max=9, value=0),
    )
except Exception as e:
    print("Interactive view needs a live Jupyter kernel. (", e, ")")''')

# ----------------------------------------------------------------------------- wrap up
md(r"""## 10 · Interpreting & caveats

**Activation shapes** (from the tutorial, applied to biomaterials):
- *Sharp peaks* → short motifs (poly-Ala crystallite edges, `GPGXX` turns, `GXY` register).
- *Broad activation* → secondary-structure runs, domains, low-complexity/repeat character.
- *Global activation* → fold class, ordered-vs-disordered, family/taxonomic signal.

**Caveats (important for class):**
- SAE features are *interpretability hypotheses*, not validated annotations; descriptions are
  auto-generated and can be wrong, especially for rare biology.
- Rankings depend on the aggregation (max vs. prevalence) and we used **no TF-IDF normalization**
  (the Forge API offers it); common features may dominate.
- We used **one SAE (layer 60, 16,384, k=64)**. Other layers/codebook sizes expose different concepts.

### Exercises
1. Find a feature that fires **only on poly-Ala** (mask/inspect MaSp1) and check its description.
2. Re-run §7 on collagen vs. ubiquitin — does the painted feature localize to a structural element?
3. Replace the SAE with `…codebook65536` and see whether features get more specific.
4. Build a feature-fingerprint **nearest-neighbor** search: given a new sequence, find the panel
   protein with the most similar fingerprint.

### References
- Biohub cookbook: *Understanding Proteins with SAE Features* (tutorial this is adapted from).
- ESMC SAE model cards: `biohub/ESMC-6B-sae-...`; feature descriptions via the ESM Atlas.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "esm", "display_name": "Python (esm — ESMC/ESMFold2)", "language": "python"}
nb["metadata"]["language_info"] = {"name": "python"}
out = "/Users/mbuehler/ESM_samples/ESM_sae_structural_proteins.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
