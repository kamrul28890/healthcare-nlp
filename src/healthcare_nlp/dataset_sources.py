from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_ade_corpus_v2(
    output_csv_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict:
    """Download and normalize ADE Corpus V2 classification split into project schema."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2",
        "Ade_corpus_v2_classification",
        split="train",
    )

    df = dataset.to_pandas()
    df = df[["text", "label"]].dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    # This benchmark comes from biomedical case report text, so mark as clinical domain.
    df["domain"] = "clinical"

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0 when provided")
        sample_size = min(sample_size, len(df))
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    class_balance = df["label"].value_counts().to_dict()
    return {
        "output_path": str(output_path),
        "rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in class_balance.items()},
    }
