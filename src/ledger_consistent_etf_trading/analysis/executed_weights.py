from __future__ import annotations
import pandas as pd

def compute_executed_weights(
    positions_df: pd.DataFrame,
    panel: pd.DataFrame,
    equity_df: pd.DataFrame,
) -> pd.DataFrame:

    # Normalize equity index
    eq = equity_df.copy()
    if "date" in eq.columns:
        eq["date"] = pd.to_datetime(eq["date"], utc=True)
        eq = eq.set_index("date")
    equity = eq["equity"]


    rows = []
    for d, pos in positions_df.iterrows():
        d = pd.to_datetime(d, utc=True)
        if d not in equity.index:
            continue
        eq_d = float(equity.loc[d])
        if eq_d <= 0:
            continue

        prices = panel.xs(d, level="date")["close"]
        for t, sh in pos.items():
            if sh == 0:
                continue
            px = float(prices.get(t, 0.0))
            if px <= 0:
                continue
            w = (sh * px) / eq_d
            rows.append({"date": d, "ticker": t, "exec_weight": w})

    return pd.DataFrame(rows)
