"""Characterization of the untouched legacy engine (commit 852d46f). These tests pin the behaviours
the V2 design replaces; they are expected to keep passing for as long as the legacy code is preserved."""
import pathlib
import subprocess
import unittest

import numpy as np
import pandas as pd

import ledger_consistent_etf_trading as pkg
from ledger_consistent_etf_trading.execution.capacity import CapacityConfig
from ledger_consistent_etf_trading.execution.cost_model import CostConfig
from ledger_consistent_etf_trading.replay.replay_engine import ReplayConfig, run_replay

REPO = pathlib.Path(__file__).resolve().parents[1]
PIN = "852d46f824dc7c8a395992fc5fccdccff5e4e468"
DAYS = pd.date_range("2031-03-03", periods=14, freq="B", tz="UTC")
ZERO = CostConfig(fees_bps=0.0, half_spread_bps=0.0, slippage_a_bps=0.0)
WIDE = CapacityConfig(max_participation=1.0, max_trade_notional=1e12, carry_unfilled=True)


def panel(tickers, over=None, price=50.0, adv=1e9):
    over = over or {}
    rows = [dict(date=d, ticker=t, open=over.get((t, i), (price, price))[0], close=over.get((t, i), (price, price))[1], adv_notional=adv, vol=0.01)
            for i, d in enumerate(DAYS) for t in tickers]
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def tape(spec):
    return pd.DataFrame([dict(date=DAYS[i], ticker=t, target_weight=w) for i, ws in spec.items() for t, w in ws.items()])


class LegacyPreserved(unittest.TestCase):
    def test_package_is_this_checkout(self):
        self.assertTrue(pathlib.Path(pkg.__file__).resolve().is_relative_to(REPO), "tests must import the V2 checkout, not the installed public repository")

    def test_legacy_modules_unchanged_since_pin(self):
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", PIN, "--", "src/ledger_consistent_etf_trading", ":!src/ledger_consistent_etf_trading/v2",
                            "scripts/0*", "scripts/1*", "visualization"])
        self.assertEqual(r.returncode, 0, "legacy sources differ from the pinned commit")


class LegacyBehaviour(unittest.TestCase):
    def test_carry_is_added_to_recomputed_target_difference(self):
        _, _, pos = run_replay(panel(["Q"]), tape({0: {"Q": 0.4}, 4: {"Q": 0.4}, 8: {"Q": 0.4}}),
                               CapacityConfig(max_participation=1.0, max_trade_notional=5_000.0, carry_unfilled=True), ZERO,
                               ReplayConfig(initial_cash=80_000.0, min_trade_notional=0.0))
        self.assertGreater(pos["Q"].max(), 640, "target is 0.4 x 80,000 / 50 = 640 units; legacy overshoots")

    def test_buy_before_funding_sale_is_dropped(self):
        for held, target, expect in (("M", "N", 1600), ("N", "M", 0)):
            _, eq, pos = run_replay(panel(["M", "N"]), tape({0: {held: 1.0, target: 0.0}, 5: {held: 0.0, target: 1.0}}), WIDE, ZERO,
                                    ReplayConfig(initial_cash=80_000.0, min_trade_notional=0.0))
            got = int(pos[target].iloc[8]) if target in pos else 0
            self.assertEqual(got, expect)

    def test_same_session_close_changes_open_priced_order(self):
        bought = []
        for close6 in (50.0, 65.0):
            fills, _, _ = run_replay(panel(["Q"], {("Q", 6): (50.0, close6)}), tape({0: {"Q": 0.5}, 5: {"Q": 0.7}}), WIDE, ZERO,
                                     ReplayConfig(initial_cash=80_000.0, min_trade_notional=0.0))
            bought.append(int(fills[fills["date"] == DAYS[6]]["shares_delta"].sum()))
        self.assertNotEqual(bought[0], bought[1])

    def test_ideal_comparator_settings_can_borrow(self):
        _, eq, _ = run_replay(panel(["Q"], {("Q", 6): (50.0, 65.0)}), tape({0: {"Q": 0.5}, 5: {"Q": 1.0}}),
                              CapacityConfig(max_participation=1.0, max_trade_notional=1e12, carry_unfilled=False), ZERO,
                              ReplayConfig(initial_cash=80_000.0, strict_cash=False, min_trade_notional=0.0))
        self.assertLess(eq["cash"].min(), -1.0)

    def test_missing_adv_fails_open(self):
        p = panel(["Q"]); p["adv_notional"] = np.nan
        fills, eq, _ = run_replay(p, tape({0: {"Q": 1.0}}), CapacityConfig(max_participation=0.05, max_trade_notional=1e12), CostConfig(), ReplayConfig(initial_cash=80_000.0))
        self.assertEqual(len(fills), 1, "the fill proceeds with no participation limit")
        self.assertTrue(np.isnan(fills["total_cost_$"].iloc[0]) and np.isnan(eq["cash"].iloc[-1]), "NaN cost propagates into cash")


if __name__ == "__main__":
    unittest.main()
