from __future__ import annotations
import pandas as pd

def equal_weight_long_only(
    panel: pd.DataFrame,
    rebalance_freq: str = "M",  # monthly
) -> pd.DataFrame:

    # Generate equal-weight long-only weights from an aligned panel.

    dates = panel.index.get_level_values("date").unique()
    tickers = panel.index.get_level_values("ticker").unique()

    # Rebalance dates
    rebal_dates = (
        pd.Series(dates)
        .dt.to_period(rebalance_freq)
        .drop_duplicates()
        .index
    )
    rebal_dates = dates[rebal_dates]

    w = 1.0 / len(tickers)

    rows = []
    for d in rebal_dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t, "target_weight": w})

    return pd.DataFrame(rows)
