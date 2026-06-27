"""
common.py — Shared featurization for the Name Origin-Country Checker.

The normalization + featurization must stay byte-identical between preprocessing (01)
and inference (05):
  - normalize_name / name_to_indices  (featurization)
  - the character vocabulary + constants

CLAUDE.md requires the normalization + featurization helpers to be identical across
scripts; importing them from one module is how we guarantee that. This filename has
no leading digit so it is importable normally (the 0X_*.py scripts are loaded by path).
"""

from __future__ import annotations

import sys
import unicodedata

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
# Character vocabulary  (KEEP IDENTICAL ACROSS 01 / 05)
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
