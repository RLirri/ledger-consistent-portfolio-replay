from __future__ import annotations
import itertools
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.execution.capacity import CapacityConfig
from ledger_consistent_etf_trading.execution.cost_model import CostConfig
from ledger_consistent_etf_trading.replay.replay_engine import ReplayConfig, run_replay
from ledger_consistent_etf_trading.risk.scorecard import build_scorecard

def main():
    ensure_dirs()

    panel = pd.read_parquet(PROCESSED_DIR / "panel_1d_aligned.parquet")
    weights = pd.read_csv(ARTIFACTS_DIR / "weights_equal_weight_weekly.csv")

    max_participation_grid = [0.01, 0.05, 0.10]
    half_spread_grid = [0.5, 1.0, 2.0]
    slippage_a_grid = [5.0, 15.0, 30.0]

    rows = []
    for mp, hs, a in itertools.product(max_participation_grid, half_spread_grid, slippage_a_grid):
        cap_cfg = CapacityConfig(
            max_participation=mp,
            max_trade_notional=250_000,
            carry_unfilled=True,
        )
        cost_cfg = CostConfig(
            fees_bps=0.0,
            half_spread_bps=hs,
            slippage_a_bps=a,
            slippage_b=0.7,
            slippage_c=1.0,
            vol_ref=0.01,
        )
        replay_cfg = ReplayConfig(
            initial_cash=1_000_000.0,
            price_convention="next_open",
            use_close_for_mtm=True,
            strict_cash=True,
            min_trade_notional=1_000.0,
        )

        fills_df, equity_df, _ = run_replay(panel, weights, cap_cfg, cost_cfg, replay_cfg)

        # scorecard expects equity CSV
        eq_tmp = equity_df.reset_index().rename(columns={"index": "date"})
        score = build_scorecard(eq_tmp, fills_df, panel, benchmark="SPY").iloc[0].to_dict()

        score.update({
            "max_participation": mp,
            "half_spread_bps": hs,
            "slippage_a_bps": a,
            "fills": int(len(fills_df)),
        })
        rows.append(score)
        print(f"Done mp={mp} hs={hs} a={a}")

    out = pd.DataFrame(rows)
    out_path = ARTIFACTS_DIR / "sensitivity_scorecards_weekly.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
