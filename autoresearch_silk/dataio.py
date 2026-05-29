"""Data plumbing for the silk autoresearch loop. DO NOT EDIT for experiments — the metric and
splits must stay fixed so results are comparable. (model.py + config.json are what you change.)

Loads cached ESMC embeddings (built by setup.py) + the 4 targets, provides a target standardizer,
grouped train/val splitting, padded per-residue batches, and the R² metric.
"""
import json, os, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["toughness", "E", "strength", "strain"]


def _cfg():
    return json.load(open(os.path.join(HERE, "config.json")))


def _slug():
    return _cfg()["esmc_model"].split("/")[-1]


class SilkData:
    """Holds one split: per-residue embeddings (list of (Li,d)), mean (N,d), targets (N,4), groups."""
    def __init__(self, split):
        import pandas as pd
        sl = _slug()
        meta_path = os.path.join(HERE, "cache", f"{sl}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No cache for '{_cfg()['esmc_model']}'. Run:  python setup.py  first.")
        self.dim = json.load(open(meta_path))["dim"]
        df = pd.read_parquet(os.path.join(HERE, "data", f"{split}.parquet"))
        self.mean = np.load(os.path.join(HERE, "cache", f"{sl}_{split}_mean.npy"))
        npz = np.load(os.path.join(HERE, "cache", f"{sl}_{split}_resid.npz"))
        flat, lengths = npz["resid"], npz["lengths"]
        offs = np.concatenate([[0], np.cumsum(lengths)])
        self.resid = [flat[offs[i]:offs[i + 1]] for i in range(len(lengths))]   # list of (Li,d) fp16
        self.y = df[TARGETS].to_numpy("float32")
        self.idv = df["idv"].to_numpy() if "idv" in df else np.arange(len(df))
        # group id = identical measured-property tuple (many sequences share one fiber measurement)
        keys = [hashlib.md5(np.round(row, 5).tobytes()).hexdigest() for row in self.y]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.groups = np.array([uniq[k] for k in keys])

    def __len__(self):
        return len(self.y)


class TargetScaler:
    """Standardize each target (fit on train). R² is scale-invariant, but standardizing makes the
    multi-target MSE loss balanced across the 4 properties."""
    def __init__(self, y):
        self.mu = y.mean(0); self.sd = y.std(0) + 1e-8

    def transform(self, y):
        return (y - self.mu) / self.sd

    def inverse(self, z):
        return z * self.sd + self.mu


def grouped_train_val_split(groups, val_frac, seed):
    """Split indices so a property-group is entirely in train OR val (no fiber leaks across)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(round(val_frac * len(uniq))))
    val_groups = set(uniq[:n_val].tolist())
    val = np.array([i for i, g in enumerate(groups) if g in val_groups])
    tr = np.array([i for i, g in enumerate(groups) if g not in val_groups])
    return tr, val


def r2_per_target(y_true, y_pred):
    """Per-target R² (coefficient of determination) and their mean."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0) + 1e-12
    per = 1.0 - ss_res / ss_tot
    return {t: float(per[i]) for i, t in enumerate(TARGETS)}, float(per.mean())


def make_batches(data: "SilkData", idx, batch_size, shuffle, seed=0, device="cpu"):
    """Yield (X (B,Lmax,d) float32, mask (B,Lmax) bool, y (B,4) float32) torch tensors.
    Per-residue embeddings are padded to the batch's max length; `mask` marks real residues."""
    import torch
    idx = np.array(idx)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sub = idx[i:i + batch_size]
        seqs = [data.resid[j] for j in sub]
        Lmax = max(s.shape[0] for s in seqs)
        X = np.zeros((len(sub), Lmax, data.dim), dtype="float32")
        mask = np.zeros((len(sub), Lmax), dtype=bool)
        for r, s in enumerate(seqs):
            X[r, :s.shape[0]] = s
            mask[r, :s.shape[0]] = True
        yield (torch.from_numpy(X).to(device),
               torch.from_numpy(mask).to(device),
               torch.from_numpy(data.y[sub]).to(device))
