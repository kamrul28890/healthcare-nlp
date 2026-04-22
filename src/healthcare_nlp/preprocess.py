from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text


def preprocess_series(texts):
    return texts.astype(str).map(normalize_text)
