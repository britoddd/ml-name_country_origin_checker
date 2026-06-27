"""
common.py — Shared utilities for the Name Origin-Country Checker classifier.

Everything that MUST stay byte-identical between preprocessing (01), training (03)
and inference (05) lives here:
  - normalize_name / name_to_indices  (featurization)
  - the character vocabulary + constants
  - Config, NameClassifier (CharCNN + BiLSTM + Attention), FocalLoss

CLAUDE.md requires the normalization + featurization helpers to be identical across
scripts; importing them from one module is how we guarantee that. This filename has
no leading digit so it is importable normally (the 0X_*.py scripts are loaded by path).
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np


# --------------------------------------------------------------------------------------
# Console / UTF-8 guard + tqdm fallback (shared by every script)
# --------------------------------------------------------------------------------------
def force_utf8_stdout() -> None:
    """Force UTF-8 stdout so emoji status prints don't crash a Windows cp1252 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


try:  # graceful no-op fallback if tqdm is missing
    from tqdm import tqdm  # noqa: F401
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):  # type: ignore
        if iterable is None:
            class _Dummy:
                def update(self, *_a, **_k):
                    pass

                def close(self):
                    pass

                def set_postfix(self, *_a, **_k):
                    pass

            return _Dummy()
        return iterable


# --------------------------------------------------------------------------------------
# Character vocabulary  (KEEP IDENTICAL ACROSS 01 / 03 / 05)
# --------------------------------------------------------------------------------------
# 1-indexed: 0 = PAD. 'a'..'z' -> 1..26, ' ' -> 27.  VOCAB_SIZE leaves room for UNK(28).
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}  # a->1 ... ' '->27
PAD_IDX = 0
UNK_IDX = len(ALPHABET) + 1  # 28 (kept reserved; normalization rarely emits it)
VOCAB_SIZE = len(ALPHABET) + 2  # 29  (PAD + 27 real chars + UNK)

MAX_LEN = 30  # fixed name length (truncate / pad)


def normalize_name(name: str) -> str:
    """NFD decompose -> drop combining marks (accents) -> lowercase -> keep [a-z ] -> collapse.

    Returns a cleaned lowercase string of a-z and single spaces, or '' if nothing survives.
    MUST be identical in preprocessing and inference.
    """
    if not isinstance(name, str):
        return ""
    # decompose accents and drop the combining marks
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = stripped.lower()
    out_chars = [c if c in CHAR_TO_IDX else " " for c in stripped]
    # collapse runs of whitespace, strip ends
    cleaned = "".join(out_chars)
    cleaned = " ".join(cleaned.split())
    return cleaned


def name_to_indices(name: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Map a (raw or normalized) name to a fixed-length int16 index vector. 0 = PAD."""
    norm = normalize_name(name)
    idxs = [CHAR_TO_IDX.get(c, UNK_IDX) for c in norm[:max_len]]
    if len(idxs) < max_len:
        idxs.extend([PAD_IDX] * (max_len - len(idxs)))
    return np.asarray(idxs, dtype=np.int16)


# --------------------------------------------------------------------------------------
# Model config + architecture
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    max_len: int = MAX_LEN
    embed_dim: int = 128
    cnn_kernels: tuple = (2, 3, 4)
    cnn_filters: int = 192
    lstm_hidden: int = 384
    lstm_layers: int = 2
    clf_hidden: int = 768
    dropout: float = 0.4
    num_classes: int = 0  # set at build time
    pad_idx: int = PAD_IDX

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cnn_kernels"] = list(self.cnn_kernels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "cnn_kernels" in d:
            d["cnn_kernels"] = tuple(d["cnn_kernels"])
        # ignore unknown keys defensively
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def build_model(cfg: "Config"):
    """Import torch lazily so 01/02 (no torch needed) stay importable."""
    import torch
    import torch.nn as nn

    class BahdanauAttention(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.W = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, 1, bias=False)

        def forward(self, seq, mask):
            # seq: (B, T, D); mask: (B, T) with True for real tokens
            score = self.v(torch.tanh(self.W(seq))).squeeze(-1)  # (B, T)
            score = score.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(score, dim=1).unsqueeze(-1)  # (B, T, 1)
            context = (seq * attn).sum(dim=1)  # (B, D)
            return context, attn.squeeze(-1)

    class NameClassifier(nn.Module):
        def __init__(self, c: Config):
            super().__init__()
            self.cfg = c
            self.embedding = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=c.pad_idx)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(c.embed_dim, c.cnn_filters, kernel_size=k, padding=k // 2)
                    for k in c.cnn_kernels
                ]
            )
            cnn_out = c.cnn_filters * len(c.cnn_kernels)
            self.lstm = nn.LSTM(
                cnn_out,
                c.lstm_hidden,
                num_layers=c.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=c.dropout if c.lstm_layers > 1 else 0.0,
            )
            lstm_out = c.lstm_hidden * 2
            self.attn = BahdanauAttention(lstm_out)
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out, c.clf_hidden),
                nn.LayerNorm(c.clf_hidden),
                nn.ReLU(),
                nn.Dropout(c.dropout),
                nn.Linear(c.clf_hidden, c.num_classes),
            )

        def forward(self, x):
            # x: (B, T) int indices
            mask = x != self.cfg.pad_idx  # (B, T)
            emb = self.embedding(x)  # (B, T, E)
            emb_t = emb.transpose(1, 2)  # (B, E, T)
            conv_feats = [torch.relu(conv(emb_t)) for conv in self.convs]
            # align time dim (padding=k//2 can produce T or T+1 for even kernels) -> trim to T
            T = x.size(1)
            conv_feats = [f[..., :T] for f in conv_feats]
            cnn = torch.cat(conv_feats, dim=1)  # (B, F*len, T)
            cnn = cnn.transpose(1, 2)  # (B, T, F*len)
            lstm_out, _ = self.lstm(cnn)  # (B, T, 2H)
            context, _ = self.attn(lstm_out, mask)  # (B, 2H)
            return self.classifier(context)

    return NameClassifier(cfg)


def build_focal_loss(alpha, gamma: float = 1.5):
    """Focal loss with per-class alpha weights. alpha is a 1-D tensor on the train device.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t),  computed from log_softmax for stability.
    """
    import torch
    import torch.nn as nn

    class FocalLoss(nn.Module):
        def __init__(self, alpha_t, g: float):
            super().__init__()
            self.register_buffer("alpha", alpha_t)
            self.gamma = g

        def forward(self, logits, target):
            logp = torch.log_softmax(logits, dim=1)
            logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)  # (B,)
            p_t = logp_t.exp()
            a_t = self.alpha[target]
            loss = -a_t * (1.0 - p_t).pow(self.gamma) * logp_t
            return loss.mean()

    return FocalLoss(alpha, gamma)


def tempered_class_weights(counts: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Inverse-frequency weights tempered by `power`, renormalized to mean 1.0."""
    counts = np.asarray(counts, dtype=np.float64)
    inv = 1.0 / np.clip(counts, 1.0, None)
    w = inv ** power
    w = w / w.mean()
    return w.astype(np.float32)
