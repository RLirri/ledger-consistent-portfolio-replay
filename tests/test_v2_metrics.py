import unittest

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import engine, metrics
from ledger_consistent_etf_trading.v2.comparator import posthoc_cost_adjusted
from ledger_consistent_etf_trading.v2.config import CostParams
from ledger_consistent_etf_trading.v2.data import build_features
from tests.helpers import NOCOST, cfg, synth, weights


def parent(q, p_D, p_A, fills, p_H, cost):
    filled = sum(x for x, _ in fills)
    return dict(parent_id=0, ticker="X", open_session=0, q=q, p_D=p_D, p_A=p_A, filled=filled, fill_value=sum(x * p for x, p in fills),
                cost=cost, last_fill_session=0, reissued=0.0, close_session=3, p_H=p_H, unfilled=q - filled, close_reason="replaced")


class OrderShortfall(unittest.TestCase):
    """Expected values are computed here from the paper-versus-real definition, not from the implementation."""

    @staticmethod
    def paper_minus_real(q, p_D, fills, p_H, cost):
        paper = q * (p_H - p_D)
        real = sum(x * (p_H - p) for x, p in fills) - cost
        return paper - real

    def test_signed_buy_and_sell(self):
        cases = [(10, 100, 100, [(6, 102)], 110, 1.0, 53.0),            # partial buy, price rises
                 (-10, 100, 100, [(-6, 98)], 90, 1.0, 53.0),            # mirrored sell
                 (10, 100, 100, [], 90, 0.0, -100.0),                   # unfilled buy, price falls: not trading helped
                 (-250, 40, 41, [(-100, 40.5), (-50, 39)], 37, 7.5, None),
                 (800, 12.5, 12.4, [(300, 12.6), (500, 12.9)], 13.3, 4.0, None)]
        for q, p_D, p_A, fills, p_H, cost, expected in cases:
            df = metrics.order_shortfall(pd.DataFrame([parent(q, p_D, p_A, fills, p_H, cost)]))
            ind = self.paper_minus_real(q, p_D, fills, p_H, cost)
            if expected is not None:
                self.assertAlmostEqual(ind, expected, places=9)
            r = df.iloc[0]
            self.assertAlmostEqual(r["is_dollars"], ind, places=9)
            self.assertAlmostEqual(r["delay"] + r["execution"] + r["opportunity"] + r["cost"], ind, places=9)

    def test_engine_parents_match_independent_definition(self):
        m = synth(dollar_volume=3e5)
        res = engine.run(m, cfg(m, initial_cash=5e6), targets={26: weights(m, AAA=0.5, BBB=0.5), 34: weights(m, CCC=0.7), 45: weights(m, AAA=1.0)})
        p = metrics.order_shortfall(res.parents)
        self.assertGreater((p["unfilled"].abs() > 0).sum(), 0)
        for _, r in p.iterrows():
            o = res.orders[res.orders["parent_id"] == r["parent_id"]]
            fills = list(zip(o["filled_units"], o["fill_price"]))
            self.assertAlmostEqual(r["is_dollars"], self.paper_minus_real(r["q"], r["p_D"], fills, r["p_H"], o["cost"].sum()), delta=1e-6 * 5e6)

    def test_decision_price_follows_timing_convention(self):
        m = synth()
        t = {30: weights(m, AAA=0.5)}
        pc = engine.run(m, cfg(m, valuation="prior_close"), targets=t).parents.iloc[0]
        op = engine.run(m, cfg(m, valuation="open"), targets=t).parents.iloc[0]
        col = m.tickers.index("AAA")
        self.assertEqual(pc["p_D"], m.close[29, col]); self.assertEqual(pc["p_A"], m.open[30, col])
        self.assertEqual(op["p_D"], m.open[30, col])
        self.assertEqual(metrics.order_shortfall(pd.DataFrame([op]))["delay"].iloc[0], 0.0)


class Tracking(unittest.TestCase):
    def test_zero_when_on_target_and_no_initialization_spike(self):
        m = synth(flat=True, dollar_volume=1e9)
        res = engine.run(m, cfg(m, cost=NOCOST, cap_enabled=False), targets={27: weights(m, AAA=0.25, BBB=0.25, CCC=0.5)})
        w = metrics.weight_tracking(res, m.close)
        self.assertEqual(w.index[0], m.dates[27], "sessions before the first effective target are excluded")
        self.assertLess(w["wte_risky"].max(), 1e-12)
        self.assertLess(w["wte_with_cash"].max(), 1e-12)

    def test_cash_coordinate(self):
        m = synth(flat=True, dollar_volume=1e9)
        res = engine.run(m, cfg(m, cost=NOCOST, cap_enabled=True, max_participation=1e-9), targets={27: weights(m, AAA=1.0)})
        w = metrics.weight_tracking(res, m.close)
        self.assertAlmostEqual(w["wte_risky"].iloc[0], 1.0, delta=1e-4)              # the cap admits only a negligible fill
        self.assertAlmostEqual(w["wte_with_cash"].iloc[0], np.sqrt(2.0), delta=1e-4)

    def test_materiality_rule_near_zero_reference(self):
        self.assertTrue(metrics.wte_material(0.012, 0.010))        # +20% relative
        self.assertFalse(metrics.wte_material(0.0105, 0.010))      # +5%
        self.assertFalse(metrics.wte_material(0.004, 0.0))         # reference ~ 0: absolute rule, below 0.005
        self.assertTrue(metrics.wte_material(0.006, 0.00005))


class Performance(unittest.TestCase):
    def test_known_series(self):
        nav = pd.Series([100.0, 110.0, 99.0, 108.9])
        p = metrics.performance(nav)
        r = np.array([0.10, -0.10, 0.10])
        self.assertAlmostEqual(p["sharpe_rf0"], r.mean() / r.std() * np.sqrt(252), places=12)
        self.assertAlmostEqual(p["ann_vol"], r.std() * np.sqrt(252), places=12)
        self.assertAlmostEqual(p["cagr"], 1.089 ** (252 / 3) - 1, places=9)
        self.assertAlmostEqual(p["max_drawdown"], -0.10, places=12)
        self.assertTrue(np.isnan(metrics.performance(pd.Series([5.0, 5.0, 5.0]))["sharpe_rf0"]))

    def test_relative_sign(self):
        idx = pd.RangeIndex(3)
        g = metrics.relative(pd.Series([100, 100, 99.0], idx), pd.Series([100, 100, 100.0], idx))
        self.assertAlmostEqual(g["terminal_gap_bps"], 100.0, places=9)
        self.assertLess(g["active_return_ann"], 0)


class PostHocComparator(unittest.TestCase):
    def setUp(self):
        self.m = synth(dollar_volume=4e5)
        self.t = {d: weights(self.m, **{self.m.tickers[(d // 3) % 5]: 0.7}) for d in range(26, 60, 3)}
        self.ref = engine.run(self.m, cfg(self.m, initial_cash=2e6, cost=NOCOST, cap_enabled=False), targets=self.t)
        self.costcfg = cfg(self.m, initial_cash=2e6, cap_enabled=False)
        self.f = build_features(self.m, "lagged", "mean_dollar")

    def test_formula_recomputed_independently(self):
        adj = posthoc_cost_adjusted(self.ref, self.m, self.f, self.costcfg)
        c = self.costcfg.cost
        col = {t: i for i, t in enumerate(self.m.tickers)}
        nav_ref = self.ref.daily["nav"].to_numpy(); s = self.ref.info["start"]
        nav = [nav_ref[0]]
        for k in range(1, len(nav_ref)):
            cost = 0.0
            for _, o in self.ref.orders[self.ref.orders["session"] == s + k].iterrows():
                n = abs(o["notional"]); i = col[o["ticker"]]
                x = min(n / self.f.adv[s + k, i], 1.0)
                cost += n * (c.half_spread_bps + c.a_bps * x ** c.b * (self.f.vol[s + k, i] / c.vol_ref) ** c.c) / 1e4
            nav.append(nav[-1] * (nav_ref[k] / nav_ref[k - 1] - cost / nav_ref[k - 1]))
        self.assertTrue(np.allclose(adj["nav"].to_numpy(), nav, rtol=1e-12))
        self.assertLess(adj["nav"].iloc[-1], nav_ref[-1])
        self.assertEqual(list(adj.columns), ["nav", "posthoc_cost"], "return-only comparator: no holdings-based outputs")

    def test_differs_from_stateful_replay_when_costs_are_large(self):
        big = CostParams(half_spread_bps=50.0)
        stateful = engine.run(self.m, cfg(self.m, initial_cash=2e6, cap_enabled=False, cost=big), targets=self.t)
        adj = posthoc_cost_adjusted(self.ref, self.m, self.f, cfg(self.m, initial_cash=2e6, cap_enabled=False, cost=big))   # 3C: capital must match the reference (F13)
        self.assertNotAlmostEqual(stateful.daily["nav"].iloc[-1], adj["nav"].iloc[-1], delta=1e-6)

    def test_rejects_constrained_reference_and_mismatched_features(self):
        constrained = engine.run(self.m, cfg(self.m, initial_cash=2e6), targets=self.t)
        with self.assertRaises(ValueError):
            posthoc_cost_adjusted(constrained, self.m, self.f, self.costcfg)
        with self.assertRaises(ValueError):
            posthoc_cost_adjusted(self.ref, self.m, build_features(self.m, "same_day", "mean_dollar"), self.costcfg)


if __name__ == "__main__":
    unittest.main()
