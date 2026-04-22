from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["text", "label", "domain"]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    df = df[REQUIRED_COLUMNS].dropna(subset=["text", "label", "domain"]).copy()
    df["label"] = df["label"].astype(int)
    df["domain"] = df["domain"].astype(str).str.lower().str.strip()
    return df


def split_train_val_test(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(train_size + val_size + test_size, 5) != 1.0:
        raise ValueError("Train/val/test proportions must sum to 1.0")

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df["label"],
        random_state=random_state,
    )

    val_fraction_of_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
