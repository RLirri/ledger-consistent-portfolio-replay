"""Sprint 3C regression tests: demand lineage, completion bounds, minimum-trade floor.
Expected values are worked out by hand in the comments, not read from the implementation."""
import unittest

import numpy as np

from ledger_consistent_etf_trading.v2 import engine, metrics
from ledger_consistent_etf_trading.v2.config import CostParams
from ledger_consistent_etf_trading.v2.data import Features
from tests.helpers import NOCOST, cfg, synth, weights

POLICIES = ("replace", "cancel", "legacy_additive")


def flat(n=1, T=40, price=100.0):
    m = synth(T=T, tickers=tuple(f"A{i}" for i in range(n)), flat=True)
    m.open[:] = price; m.close[:] = price
    return m


def feats(m, cap_dollars_per_day, vol=0.01):
    """Constant features; with participation 0.05 the daily fill limit is cap_dollars_per_day."""
    return Features(np.full_like(m.close, vol), np.full_like(m.close, cap_dollars_per_day / 0.05), "lagged", "mean_dollar")


def run(m, targets, cap, policy="replace", cash=20_000.0, **kw):
    return engine.run(m, cfg(m, initial_cash=cash, cost=NOCOST, residual_policy=policy, **kw), targets=targets, feats=feats(m, cap))


def lineage(res):
    return metrics.demand_summary(res.demand), metrics.order_summary(res.parents)


class CompletionAccounting(unittest.TestCase):
    def assertLineage(self, res, originated, executed, withdrawn, outstanding):
        d, o = lineage(res)
        for k, v in (("originated_value", originated), ("executed_value", executed), ("withdrawn_value", withdrawn), ("outstanding_end_value", outstanding)):
            self.assertAlmostEqual(d[k], v, places=6, msg=k)
        self.assertAlmostEqual(d["lineage_identity_residual"], 0.0, places=6)
        self.assertAlmostEqual(o["new_demand_notional"], originated, places=6)
        self.assertAlmostEqual(o["executed_demand_value"], executed, places=6)
        if originated > 0:
            self.assertAlmostEqual(o["completion_aggregate"], executed / originated, places=12)
            self.assertTrue(0.0 <= o["completion_aggregate"] <= 1.0)

    def test_same_demand_fully_served_is_exactly_one_under_any_later_price(self):
        # 100 units wanted at $100; 50 fill on day 26; the price then moves; the other 50 fill on day 27.
        for later in (25.0, 100.0, 150.0, 200.0):          # above $300 the 100-unit holding would need leverage
            m = flat(T=29)
            m.close[26:] = later; m.open[27:] = later
            # day-27 target keeps the same 100-unit holding: weight = 100*later / (15000 + 50*later)
            w2 = 100 * later / (15_000 + 50 * later)
            f = feats(m, 5_000); f.adv[27] = 50 * later / 0.05
            res = engine.run(m, cfg(m, initial_cash=20_000.0, cost=NOCOST), targets={26: np.array([0.5]), 27: np.array([w2])}, feats=f)
            self.assertAlmostEqual(res.positions[-1, 0], 100.0, places=8)
            self.assertLineage(res, 10_000, 10_000, 0, 0)          # valued at the $100 basis whatever happened later
            self.assertEqual(lineage(res)[1]["completion_aggregate"], 1.0)

    def test_partial_fill_then_expansion(self):
        # day 26: want 100, fill 20. day 27: want 150 in total -> 80 continue, 50 new. Fills 20/day through day 39.
        res = run(flat(), {26: np.array([0.5]), 27: np.array([0.75])}, cap=2_000)
        filled = 20 * 14                                           # sessions 26..39, capped below the 150 wanted -> min(150, 280)
        self.assertAlmostEqual(res.positions[-1, 0], 150.0, places=8)
        self.assertLineage(res, 15_000, 15_000, 0, 0)

    def test_reduction_withdraws_newest_demand(self):
        # want 100, fill 20; then want only 50 in total -> remaining gap 30: 30 continue, 50 withdrawn.
        res = run(flat(), {26: np.array([0.5]), 27: np.array([0.25])}, cap=2_000)
        self.assertAlmostEqual(res.positions[-1, 0], 50.0, places=8)
        self.assertLineage(res, 10_000, 5_000, 5_000, 0)
        self.assertAlmostEqual(lineage(res)[1]["completion_aggregate"], 0.5, places=12)

    def test_reversal_withdraws_and_originates_opposite_demand(self):
        # want 100, fill 20; then target 0 -> the 80 outstanding buy demand is withdrawn and 20 units of sell demand arise.
        res = run(flat(), {26: np.array([0.5]), 27: np.array([0.0])}, cap=2_000)
        self.assertAlmostEqual(res.positions[-1, 0], 0.0, places=8)
        self.assertLineage(res, 10_000 + 2_000, 2_000 + 2_000, 8_000, 0)

    def test_zero_demand_episode(self):
        m = flat()
        res = run(m, {26: np.array([0.0])}, cap=2_000)
        d, o = lineage(res)
        self.assertEqual(d["originated_value"], 0.0)
        self.assertIsNone(o["completion_aggregate"])
        self.assertIsNone(metrics.summarize(res, m.close)["orders"]["completion_aggregate"])
        self.assertIn("completion_aggregate", metrics.summarize(res, m.close)["undefined"])

    def test_outstanding_at_end_of_sample(self):
        res = run(flat(T=30), {26: np.array([0.5])}, cap=2_000)     # 4 sessions x 20 units
        self.assertLineage(res, 10_000, 8_000, 0, 2_000)

    def test_identical_demand_and_fills_give_identical_completion_across_policies(self):
        # Daily targets (replace and cancel coincide in holdings), weekly-style gaps, and an interrupted schedule.
        schedules = {"daily": {d: np.array([0.5]) for d in range(26, 40)},
                     "repeated_same_target_with_gaps": {26: np.array([0.5]), 31: np.array([0.5]), 36: np.array([0.5])},
                     "interrupted": {26: np.array([0.5]), 27: np.array([0.5]), 33: np.array([0.3]), 34: np.array([0.6])}}
        for name, t in schedules.items():
            a, b = run(flat(), t, cap=1_000), run(flat(), t, cap=1_000, policy="cancel")
            if np.array_equal(a.positions, b.positions):
                ca, cb = lineage(a)[1], lineage(b)[1]
                for k in ("completion_aggregate", "episode_served_ratio", "new_demand_notional", "executed_demand_value"):
                    self.assertAlmostEqual(ca[k], cb[k], places=12, msg=f"{name}:{k}")
            elif name == "repeated_same_target_with_gaps":          # cancel trades less between targets: same single demand, less executed
                for r in (a, b):
                    self.assertAlmostEqual(lineage(r)[0]["originated_value"], 10_000, places=6)
                self.assertLess(lineage(b)[1]["completion_aggregate"], lineage(a)[1]["completion_aggregate"])
            else:                                                   # holdings differ, so later gaps (and demand) legitimately differ
                for r in (a, b):
                    self.assertTrue(0.0 <= lineage(r)[1]["completion_aggregate"] <= 1.0)
                    self.assertAlmostEqual(lineage(r)[0]["lineage_identity_residual"], 0.0, places=6)
        daily = run(flat(), schedules["daily"], cap=1_000)
        self.assertAlmostEqual(lineage(daily)[1]["new_demand_notional"], 10_000, places=6, msg="one economic demand, re-issued 14 times")

    def test_cancelled_then_reissued_demand_is_one_demand(self):
        t = {26: np.array([0.5]), 31: np.array([0.5])}
        res = run(flat(), t, cap=1_000, policy="cancel")           # trades only on the two target sessions: 10 + 10 units
        self.assertEqual(sorted(set(res.orders["session"])), [26, 31])
        self.assertLineage(res, 10_000, 2_000, 0, 8_000)

    def test_overshoot_is_recorded_separately_and_never_inflates_completion(self):
        t = {26: np.array([0.5]), 31: np.array([0.5]), 36: np.array([0.5])}
        res = run(flat(T=60), t, cap=1_200, policy="legacy_additive")
        d, o = lineage(res)
        self.assertGreater(res.positions[:, 0].max(), 100.0 + 1)   # the diagnostic overshoots the 100-unit target
        self.assertGreater(d["overshoot_value"], 0.0)
        self.assertLessEqual(o["completion_aggregate"], 1.0)
        self.assertAlmostEqual(d["lineage_identity_residual"], 0.0, places=6)
        clean = lineage(run(flat(T=60), t, cap=1_200))[0]
        self.assertEqual(clean["overshoot_value"], 0.0)

    def test_randomized_bounds_and_identity(self):
        rng = np.random.default_rng(20261119)
        cases = 0
        for seed in range(40):
            m = synth(T=60, seed=seed, dollar_volume=float(rng.choice([2e4, 2e5, 5e6])))
            t = {int(d): rng.dirichlet(np.ones(6))[:5] * rng.choice([0.0, 0.6, 1.0]) for d in rng.choice(np.arange(26, 60), size=12, replace=False)}
            for policy in POLICIES:
                for units in ("fractional", "integer"):
                    res = engine.run(m, cfg(m, initial_cash=float(rng.choice([1e5, 1e7, 1e9])), residual_policy=policy, units=units), targets=t)
                    d, o = lineage(res)
                    scale = max(1.0, d["originated_value"])
                    self.assertLess(abs(d["lineage_identity_residual"]) / scale, 1e-9)
                    if o["completion_aggregate"] is not None:
                        self.assertGreaterEqual(o["completion_aggregate"], 0.0)
                        self.assertLessEqual(o["completion_aggregate"], 1.0 + 1e-12)
                        self.assertLessEqual(o["episode_served_ratio"], 1.0 + 1e-12)
                    if policy != "legacy_additive":
                        self.assertEqual(d["overshoot_value"], 0.0)
                    self.assertAlmostEqual(res.parents["demand_new_value"].sum(), d["originated_value"], delta=1e-9 * scale)
                    self.assertAlmostEqual(res.parents["demand_executed_value"].sum(), d["executed_value"], delta=1e-9 * scale)
                    cases += 1
        self.assertEqual(cases, 240)


class MinimumTradeFloor(unittest.TestCase):
    COST1 = CostParams(a_bps=0.0, half_spread_bps=100.0)            # flat 1%

    def test_boundary_equality_executes(self):
        m = flat(T=28)
        res = engine.run(m, cfg(m, initial_cash=5_000.0, min_trade_notional=1_000.0, cost=NOCOST, cap_enabled=False), targets={26: np.array([0.2])})
        self.assertAlmostEqual(res.orders["notional"].iloc[0], 1_000.0, places=9)

    def test_cash_scaled_buy_below_floor_is_not_executed_and_cash_is_kept(self):
        m = flat(T=28)
        res = engine.run(m, cfg(m, initial_cash=1_000.0, min_trade_notional=1_000.0, cost=self.COST1, cap_enabled=False), targets={26: np.array([1.0])}, feats=feats(m, 1e9))
        self.assertEqual(res.orders["filled_units"].abs().sum(), 0.0)
        self.assertIn("min_trade", res.orders["limited_by"].iloc[0])
        self.assertAlmostEqual(res.daily["cash"].iloc[-1], 1_000.0, places=9)

    def test_concurrent_buys_no_reallocation(self):
        # $3,000 cash, 1% cost, wants $2,000 of A0 and $1,000 of A1: scale 1/1.01 -> $1,980.20 and $990.10.
        # A1 falls below the $1,000 floor and is dropped; its cash is NOT given to A0.
        m = flat(n=2, T=28)
        res = engine.run(m, cfg(m, initial_cash=3_000.0, min_trade_notional=1_000.0, cost=self.COST1, cap_enabled=False),
                         targets={26: np.array([2 / 3, 1 / 3])}, feats=feats(m, 1e9))
        o = res.orders[res.orders["session"] == 26].set_index("ticker")
        self.assertAlmostEqual(o.loc["A0", "notional"], 2_000 / 1.01, places=6)
        self.assertEqual(o.loc["A1", "filled_units"], 0.0)
        self.assertTrue((res.orders.loc[res.orders["filled_units"] != 0, "notional"].abs() >= 1_000.0 * (1 - 1e-12)).all())
        self.assertGreaterEqual(res.daily["cash"].min(), 0.0)

    def test_funding_sell_and_residual_retry(self):
        # Hold A0, rotate to A1; the buy is funded by the same-session sale and clears the floor.
        m = flat(n=2, T=32)
        res = engine.run(m, cfg(m, initial_cash=5_000.0, min_trade_notional=1_000.0, cost=NOCOST, cap_enabled=False),
                         targets={26: np.array([1.0, 0.0]), 28: np.array([0.0, 1.0])})
        self.assertAlmostEqual(res.positions[-1, 1], 50.0, places=8)
        # A buy dropped by the floor stays outstanding and is retried later under the replace policy.
        m = flat(T=32)
        f = feats(m, 600.0); f.adv[29:] = 5_000.0 / 0.05
        res = engine.run(m, cfg(m, initial_cash=20_000.0, min_trade_notional=1_000.0, cost=NOCOST), targets={26: np.array([0.5])}, feats=f)
        self.assertEqual(res.orders[res.orders["session"] < 29]["filled_units"].abs().sum(), 0.0)
        self.assertGreater(res.positions[-1, 0], 0.0)
        self.assertTrue((res.orders.loc[res.orders["filled_units"] != 0, "notional"].abs() >= 1_000.0).all())

    def test_integer_units_and_legacy_sequential_respect_the_floor(self):
        rng = np.random.default_rng(5)
        for rule in ("sells_first_proportional", "legacy_sequential"):
            for seed in range(10):
                m = synth(T=50, seed=seed)
                t = {int(d): rng.dirichlet(np.ones(5)) for d in range(26, 50, 3)}
                res = engine.run(m, cfg(m, initial_cash=30_000.0, units="integer", min_trade_notional=2_500.0, cash_rule=rule, cost=self.COST1), targets=t)
                traded = res.orders.loc[res.orders["filled_units"] != 0, "notional"].abs()
                self.assertTrue((traded >= 2_500.0 * (1 - 1e-12)).all(), rule)
                self.assertGreaterEqual(res.daily["cash"].min(), -1e-6)


if __name__ == "__main__":
    unittest.main()
