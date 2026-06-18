from __future__ import annotations
import pandas as pd

def to_utc_date_index(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df = df.dropna(subset=[col]).set_index(col)
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = df.index.tz_convert("UTC").normalize()
    return df.sort_index()
