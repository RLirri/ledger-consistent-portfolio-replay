from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.portfolio.baseline_weights import equal_weight_long_only

def main():
    panel = pd.read_parquet(PROCESSED_DIR / "panel_1d_aligned.parquet")

    weights = equal_weight_long_only(panel, rebalance_freq="M")

    out = ARTIFACTS_DIR / "weights_equal_weight.csv"
    weights.to_csv(out, index=False)

    print(f"Saved dummy weights: {out}")

if __name__ == "__main__":
    main()
