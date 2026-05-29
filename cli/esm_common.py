"""Shared helpers for the ESM CLI tools (device selection, model loading, FASTA I/O,
batched embeddings). Import from the sibling CLI scripts.

All tools run LOCALLY (no Biohub API key) in the dedicated `esm` conda environment.
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


# --------------------------------------------------------------------------- devices
def pick_device(prefer: str = "auto", heavy: bool = False) -> str:
    """Choose a torch device.

    heavy=True (ESMC-6B, ESMFold2, SAE) -> CPU unless CUDA, because Apple-Silicon MPS
    lacks a few ops these paths need. Light models (ESMC-300M/600M) use MPS when present.
    """
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if heavy:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- FASTA / inputs
def read_fasta(path: str) -> dict[str, str]:
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0] or f"seq{len(seqs)}"
                buf = []
            elif line:
                buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(s: str) -> str:
    s = "".join(s.split()).upper()
    bad = sorted(set(s) - VALID_AA)
    if bad:
        raise ValueError(f"invalid amino-acid characters: {bad}")
    return s


def collect_inputs(seq: str | None, fasta: str | None) -> dict[str, str]:
    """Return {name: sequence} from a --seq string and/or a --fasta file."""
    out: dict[str, str] = {}
    if fasta:
        out.update({k: clean_seq(v) for k, v in read_fasta(fasta).items()})
    if seq:
        out["query"] = clean_seq(seq)
    if not out:
        raise SystemExit("Provide --seq SEQUENCE and/or --fasta FILE")
    return out


# --------------------------------------------------------------------------- ESMC loading
def load_esmc(model_name: str = "biohub/ESMC-300M", device: str | None = None,
              for_masked_lm: bool = True):
    """Load an ESMC tokenizer + model. for_masked_lm=True gives logits + hidden states."""
    from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
    heavy = "6B" in model_name
    device = device or pick_device(heavy=heavy)
    tok = AutoTokenizer.from_pretrained(model_name)
    Cls = AutoModelForMaskedLM if for_masked_lm else AutoModel
    model = Cls.from_pretrained(model_name)
    model = (model.float() if device != "cuda" and heavy else model).to(device).eval()
    return tok, model, device


@torch.inference_mode()
def embed_sequences(tok, model, seqs: list[str], device: str, pool: str = "mean",
                    batch_size: int = 8) -> list[np.ndarray]:
    """Mean-pooled (or per-residue) final-layer ESMC embeddings, batched & pad-masked.

    pool='mean' -> one (d,) vector per sequence; pool='none' -> (L, d) per sequence.
    """
    results: list[np.ndarray] = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]                       # (B, L, d)
        mask = enc["attention_mask"].clone().float()
        mask[:, 0] = 0                                  # drop <cls>
        lengths = enc["attention_mask"].sum(1)
        for r, L in enumerate(lengths):
            mask[r, int(L) - 1] = 0                      # drop <eos>
        if pool == "mean":
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            for v in pooled:
                results.append(v.float().cpu().numpy())
        else:
            for r, L in enumerate(lengths):
                results.append(h[r, 1:int(L) - 1].float().cpu().numpy())
    return results
