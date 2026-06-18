from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ARTIFACTS_DIR

def pick_closest(df: pd.DataFrame, target: float) -> pd.Series:
    i = (df["pct_clipped"] - target).abs().idxmin()
    return df.loc[i]

def main():
    df = pd.read_csv(ARTIFACTS_DIR / "capacity_stress_weekly_with_te_is.csv")
    df = df[df["max_participation"] == 0.01].copy()

    targets = [0.00, 0.05, 0.25, 0.85]  
    rows = [pick_closest(df, t) for t in targets]

    out = pd.DataFrame(rows)[[
        "initial_cash","max_trade_notional","pct_clipped",
        "avg_weight_te","p95_weight_te","impl_shortfall_bps",
        "end_equity","sharpe","max_drawdown","total_cost_$","fills"
    ]].reset_index(drop=True)

    out_path = ARTIFACTS_DIR / "table_regimes_weekly.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(out)

if __name__ == "__main__":
    main()
