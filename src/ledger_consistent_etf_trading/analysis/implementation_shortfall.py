from __future__ import annotations
import pandas as pd

def compute_implementation_shortfall(
    equity_exec: pd.Series,
    equity_ideal: pd.Series,
) -> dict:
    """
    Returns IS in $ and bps.
    """
    equity_exec = equity_exec.dropna()
    equity_ideal = equity_ideal.reindex(equity_exec.index).dropna()

    start = equity_exec.index[0]
    end = equity_exec.index[-1]

    exec_final = float(equity_exec.loc[end])
    ideal_final = float(equity_ideal.loc[end])

    is_dollars = ideal_final - exec_final
    is_bps = (is_dollars / ideal_final) * 10_000 if ideal_final > 0 else 0.0

    return {
        "impl_shortfall_$": float(is_dollars),
        "impl_shortfall_bps": float(is_bps),
    }
