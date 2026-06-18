from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.risk.scorecard import build_scorecard

def main():
    ensure_dirs()

    panel = pd.read_parquet(PROCESSED_DIR / "panel_1d_aligned.parquet")
    equity = pd.read_csv(ARTIFACTS_DIR / "equity_equal_weight.csv")
    fills = pd.read_csv(ARTIFACTS_DIR / "fills_equal_weight.csv")

    score = build_scorecard(equity_df=equity, fills_df=fills, panel=panel, benchmark="SPY")
    out = ARTIFACTS_DIR / "scorecard_equal_weight.csv"
    score.to_csv(out, index=False)

    print(f"Saved scorecard: {out}")
    print(score.T)

if __name__ == "__main__":
    main()
