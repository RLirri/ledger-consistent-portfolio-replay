from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.execution.capacity import CapacityConfig
from ledger_consistent_etf_trading.execution.cost_model import CostConfig
from ledger_consistent_etf_trading.replay.replay_engine import ReplayConfig, run_replay

def main():
    ensure_dirs()

    panel = pd.read_parquet(PROCESSED_DIR / "panel_1d_aligned.parquet")
    weights = pd.read_csv(ARTIFACTS_DIR / "weights_equal_weight.csv")

    cap_cfg = CapacityConfig(
        max_participation=0.05,
        max_trade_notional=250_000,
        carry_unfilled=True,
    )

    cost_cfg = CostConfig(
        fees_bps=0.0,
        half_spread_bps=0.5,
        slippage_a_bps=15.0,
        slippage_b=0.7,
        slippage_c=1.0,
        vol_ref=0.01,
    )

    replay_cfg = ReplayConfig(
        initial_cash=1_000_000.0,
        price_convention="next_open",
        use_close_for_mtm=True,
    )

    fills_df, equity_df, positions_df = run_replay(panel, weights, cap_cfg, cost_cfg, replay_cfg)

    fills_path = ARTIFACTS_DIR / "fills_equal_weight.csv"
    equity_path = ARTIFACTS_DIR / "equity_equal_weight.csv"
    pos_path = ARTIFACTS_DIR / "positions_equal_weight.csv"

    fills_df.to_csv(fills_path, index=False)
    equity_df.to_csv(equity_path)
    positions_df.to_csv(pos_path)

    print("Saved:")
    print(" ", fills_path)
    print(" ", equity_path)
    print(" ", pos_path)

if __name__ == "__main__":
    main()
