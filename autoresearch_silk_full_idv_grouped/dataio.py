"""Data plumbing for grouped silk autoresearch.

Each sample is one fiber/property id (`idv`) with a set of independently embedded spidroin
sequences. Models receive `(B, Smax, d)` sequence-level ESMC embeddings plus a sequence mask and
optional category ids/lengths, then predict four fiber-level targets.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_TARGETS = ["toughness", "E", "strength", "strain"]
NORM_TARGETS = ["toughnessNorm", "ENorm", "strengthNorm", "strainNorm"]


def _cfg():
    return json.load(open(os.path.join(HERE, "config.json")))


def target_columns():
    mode = _cfg().get("target_mode", "raw")
    if mode == "raw":
        return RAW_TARGETS
    if mode == "norm":
        return NORM_TARGETS
    raise ValueError(f"Unsupported target_mode: {mode}")


def _dataset():
    return _cfg().get("dataset", "lamm-mit/silkome-full-idv-grouped")


def _dataset_slug():
    return _dataset().split("/")[-1]


def _cache_key():
    return f"{_dataset_slug()}__{_cfg()['esmc_model'].split('/')[-1]}"


class GroupedSilkData:
    """One split of grouped sequence-set samples."""

    def __init__(self, split):
        import pandas as pd

        key, dsl = _cache_key(), _dataset_slug()
        meta_path = os.path.join(HERE, "cache", f"{key}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No cache for dataset '{_dataset()}' + model '{_cfg()['esmc_model']}'. "
                "Run: python setup.py first."
            )
        self.meta = json.load(open(meta_path))
        if self.meta.get("data_only"):
            raise FileNotFoundError("Cache metadata is data-only; run python setup.py without --data-only.")
        self.dim = int(self.meta["dim"])
        self.category_classes = list(self.meta.get("category_classes", []))
        self.n_categories = len(self.category_classes) + 1  # + padding id 0
        self.df = pd.read_parquet(os.path.join(HERE, "data", f"{dsl}_{split}.parquet"))

        self.seq_mean = np.load(os.path.join(HERE, "cache", f"{key}_{split}_seq_mean.npy"))
        self.seq_mask = np.load(os.path.join(HERE, "cache", f"{key}_{split}_seq_mask.npy"))
        self.category_ids = np.load(os.path.join(HERE, "cache", f"{key}_{split}_category_ids.npy"))
        self.seq_lengths = np.load(os.path.join(HERE, "cache", f"{key}_{split}_seq_lengths.npy"))

        targets = target_columns()
        self.target_names = RAW_TARGETS
        self.y = self.df[targets].to_numpy("float32")
        self.idv = self.df["idv"].to_numpy() if "idv" in self.df else np.arange(len(self.df))
        if "property_tuple_key" in self.df:
            keys = self.df["property_tuple_key"].astype(str).tolist()
        else:
            keys = [hashlib.md5(np.round(row, 5).tobytes()).hexdigest() for row in self.y]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.groups = np.array([uniq[k] for k in keys])

    def __len__(self):
        return len(self.y)


class TargetScaler:
    """Standardize each target using the training split."""

    def __init__(self, y):
        self.mu = y.mean(0)
        self.sd = y.std(0) + 1e-8

    def transform(self, y):
        return (y - self.mu) / self.sd

    def inverse(self, z):
        return z * self.sd + self.mu


def grouped_train_val_split(groups, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(round(val_frac * len(uniq))))
    val_groups = set(uniq[:n_val].tolist())
    val = np.array([i for i, g in enumerate(groups) if g in val_groups])
    tr = np.array([i for i, g in enumerate(groups) if g not in val_groups])
    return tr, val


def r2_per_target(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0) + 1e-12
    per = 1.0 - ss_res / ss_tot
    return {t: float(per[i]) for i, t in enumerate(RAW_TARGETS)}, float(per.mean())


def make_batches(data: GroupedSilkData, idx, batch_size, shuffle, seed=0, device="cpu", return_idx=False):
    import torch

    idx = np.array(idx)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sub = idx[i : i + batch_size]
        out = (
            torch.from_numpy(data.seq_mean[sub]).to(device),
            torch.from_numpy(data.seq_mask[sub]).to(device),
            torch.from_numpy(data.category_ids[sub]).to(device),
            torch.from_numpy(data.seq_lengths[sub]).to(device),
            torch.from_numpy(data.y[sub]).to(device),
        )
        yield (out + (sub,)) if return_idx else out
