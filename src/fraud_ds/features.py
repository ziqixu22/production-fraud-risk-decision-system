from __future__ import annotations
import numpy as np
import pandas as pd

BASE_NUMERIC = [
    "TransactionAmt", "TransactionDT",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "C1", "C2", "C4", "C11", "C13", "C14",
    "D1", "D2", "D10", "D15"
]

BASE_CATEGORICAL = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain", "DeviceType"
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["log_amount"] = np.log1p(x["TransactionAmt"].clip(lower=0))
    seconds_day = 24 * 60 * 60
    x["day_index"] = (x["TransactionDT"] // seconds_day).astype(int)
    x["hour"] = ((x["TransactionDT"] % seconds_day) // 3600).astype(int)
    x["is_night"] = x["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    return x


def selected_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in BASE_NUMERIC + ["log_amount", "day_index", "hour", "is_night"] if c in df.columns]
    categorical = [c for c in BASE_CATEGORICAL if c in df.columns]
    return numeric, categorical
