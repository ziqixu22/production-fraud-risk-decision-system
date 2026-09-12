from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_and_merge(transaction_path: Path, identity_path: Path | None = None) -> pd.DataFrame:
    tx = pd.read_csv(transaction_path)
    if identity_path and identity_path.exists():
        ident = pd.read_csv(identity_path)
        df = tx.merge(ident, on="TransactionID", how="left", validate="one_to_one")
    else:
        df = tx

    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique")

    return df.sort_values("TransactionDT").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_fraction: float = 0.70, valid_fraction: float = 0.15) -> Splits:
    if train_fraction + valid_fraction >= 1:
        raise ValueError("train_fraction + valid_fraction must be < 1")
    n = len(df)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + valid_fraction))
    return Splits(df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy())
