from __future__ import annotations
import pandas as pd
import numpy as np

def build_target_weight_panel(
    weights: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Forward-fills target weights between rebalance dates.
    Returns MultiIndex DataFrame (date, ticker) -> target_weight
    """
    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"], utc=True)
    w = w.set_index(["date", "ticker"]).sort_index()

    idx = pd.MultiIndex.from_product(
        [calendar, w.index.get_level_values("ticker").unique()],
        names=["date", "ticker"]
    )

    w_full = w.reindex(idx)
    w_full["target_weight"] = w_full["target_weight"].groupby("ticker").ffill().fillna(0.0)
    return w_full

def compute_weight_tracking_error(
    target: pd.DataFrame,
    executed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns daily tracking error time series.
    """
    ex = executed.set_index(["date", "ticker"])
    df = target.join(ex, how="left").fillna(0.0)

    df["sq_err"] = (df["exec_weight"] - df["target_weight"]) ** 2
    te = df.groupby("date")["sq_err"].sum().pow(0.5)
    return te.rename("weight_tracking_error")
