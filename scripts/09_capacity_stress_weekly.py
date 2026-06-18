from __future__ import annotations

import itertools
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.execution.capacity import CapacityConfig
from ledger_consistent_etf_trading.execution.cost_model import CostConfig
from ledger_consistent_etf_trading.replay.replay_engine import ReplayConfig, run_replay
from ledger_consistent_etf_trading.risk.scorecard import build_scorecard

from ledger_consistent_etf_trading.analysis.executed_weights import compute_executed_weights
from ledger_consistent_etf_trading.analysis.tracking_error import (
    build_target_weight_panel,
    compute_weight_tracking_error,
)
from ledger_consistent_etf_trading.analysis.implementation_shortfall import (
    compute_implementation_shortfall,
)


def clip_stats(fills: pd.DataFrame) -> dict:
    if len(fills) == 0:
        return {"pct_clipped": 0.0, "avg_remaining_$": 0.0, "p95_remaining_$": 0.0}

    rem = fills.get("remaining_notional_$", pd.Series([0.0] * len(fills))).abs()
    was = fills.get("was_clipped", pd.Series([False] * len(fills))).astype(bool)

    pct = float(was.mean())
    avg_rem = float(rem[was].mean()) if was.any() else 0.0
    p95_rem = float(rem[was].quantile(0.95)) if was.any() else 0.0

    return {"pct_clipped": pct, "avg_remaining_$": avg_rem, "p95_remaining_$": p95_rem}


def main():
    ensure_dirs()

    panel = pd.read_parquet(PROCESSED_DIR / "panel_1d_aligned.parquet")
    weights = pd.read_csv(ARTIFACTS_DIR / "weights_equal_weight_weekly.csv")

    # Calendar for target-weight forward-fill
    calendar = panel.index.get_level_values("date").unique().sort_values()
    target_panel = build_target_weight_panel(weights, calendar)

    # Baseline execution costs
    base_cost = CostConfig(
        fees_bps=0.0,
        half_spread_bps=0.5,
        slippage_a_bps=15.0,
        slippage_b=0.7,
        slippage_c=1.0,
        vol_ref=0.01,
    )

    # Configs for implementation shortfall
    ideal_cap = CapacityConfig(
        max_participation=1.0,
        max_trade_notional=1e12,
        carry_unfilled=False,
    )
    ideal_cost = CostConfig(
        fees_bps=0.0,
        half_spread_bps=0.0,
        slippage_a_bps=0.0,
        slippage_b=0.7,
        slippage_c=1.0,
        vol_ref=0.01,
    )

    # Capacity stress grid to trigger clipping
    initial_cash_grid = [1_000_000, 5_000_000, 10_000_000, 25_000_000]
    max_trade_notional_grid = [25_000, 50_000, 100_000, 250_000]
    max_participation_grid = [0.01, 0.05]  # for ETFs, ADV cap usually not binding here

    rows = []

    for init_cash, max_notional, mp in itertools.product(
        initial_cash_grid, max_trade_notional_grid, max_participation_grid
    ):
        # Executed run with caps + costs
        cap_cfg = CapacityConfig(
            max_participation=mp,
            max_trade_notional=max_notional,
            carry_unfilled=True,
        )
        replay_cfg = ReplayConfig(
            initial_cash=float(init_cash),
            price_convention="next_open",
            use_close_for_mtm=True,
            strict_cash=True,
            min_trade_notional=1_000.0,
        )

        fills_df, equity_df, positions_df = run_replay(panel, weights, cap_cfg, base_cost, replay_cfg)

        # Scorecard
        eq_tmp = equity_df.reset_index().rename(columns={"index": "date"})
        score = build_scorecard(eq_tmp, fills_df, panel, benchmark="SPY").iloc[0].to_dict()

        # Clipping stats
        cs = clip_stats(fills_df)

        # Tracking error for executed weights vs target weights
        exec_w = compute_executed_weights(
            positions_df=positions_df,
            panel=panel,
            equity_df=eq_tmp,
        )
        te = compute_weight_tracking_error(target_panel, exec_w)

        te_stats = {
            "avg_weight_te": float(te.mean()) if len(te) else 0.0,
            "p95_weight_te": float(te.quantile(0.95)) if len(te) else 0.0,
            "max_weight_te": float(te.max()) if len(te) else 0.0,
        }

        # Run for implementation shortfall
        ideal_replay_cfg = ReplayConfig(
            initial_cash=float(init_cash),
            price_convention="next_open",
            use_close_for_mtm=True,
            strict_cash=False,
            min_trade_notional=0.0,
        )
        _, equity_ideal_df, _ = run_replay(panel, weights, ideal_cap, ideal_cost, ideal_replay_cfg)

        is_stats = compute_implementation_shortfall(
            equity_exec=equity_df["equity"],
            equity_ideal=equity_ideal_df["equity"],
        )

        # Combine
        score.update(
            {
                "initial_cash": init_cash,
                "max_trade_notional": max_notional,
                "max_participation": mp,
                "fills": int(len(fills_df)),
                **cs,
                **te_stats,
                **is_stats,
            }
        )
        rows.append(score)

        print(
            f"Done cash={init_cash} maxN={max_notional} mp={mp} "
            f"clipped={cs['pct_clipped']:.3f} te_avg={te_stats['avg_weight_te']:.6f} "
            f"is_bps={is_stats['impl_shortfall_bps']:.2f}"
        )

    out = pd.DataFrame(rows)

    out_path = ARTIFACTS_DIR / "capacity_stress_weekly_with_te_is.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
