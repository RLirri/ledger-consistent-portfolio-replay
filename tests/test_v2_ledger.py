"""Ledger, cash and residual invariants. Tolerances scale with initial capital."""
import unittest

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import engine
from ledger_consistent_etf_trading.v2.config import CostParams
from ledger_consistent_etf_trading.v2.data import market_from_frames
from tests.helpers import NOCOST, cfg, synth, weights


def rotating_targets(m, start=25, end=70, step=3):
    names = m.tickers
    out = {}
    for j, d in enumerate(range(start + 1, end, step)):
        w = np.zeros(len(names)); w[j % len(names)] = 0.6; w[(j + 2) % len(names)] = 0.4
        out[d] = w
    return out


def check_invariants(tc, m, res):
    c = res.cfg
    tol = 1e-9 * max(1.0, c.initial_cash)
    o, d = res.orders, res.daily
    s, e = res.info["start"], res.info["end"]
    # cash conservation and single cost debit
    tc.assertAlmostEqual(d["cash"].iloc[-1], c.initial_cash - o["notional"].sum() - o["cost"].sum(), delta=tol * 10)
    by = o.groupby("session").agg(n=("notional", "sum"), c=("cost", "sum")).reindex(range(s, e), fill_value=0.0)
    cash_path = c.initial_cash - (by["n"] + by["c"]).cumsum().to_numpy()
    tc.assertTrue(np.allclose(cash_path, d["cash"].to_numpy(), rtol=0, atol=tol * 10))
    tc.assertAlmostEqual(o["cost"].sum(), d["cost"].sum(), delta=tol)
    tc.assertAlmostEqual(o["cost"].sum(), res.parents["cost"].sum(), delta=tol)
    # position reconciliation
    fills = o.pivot_table(index="session", columns="ticker", values="filled_units", aggfunc="sum").reindex(index=range(s, e), columns=m.tickers).fillna(0.0)
    tc.assertTrue(np.allclose(fills.cumsum().to_numpy(), res.positions, rtol=1e-12, atol=1e-9 * c.initial_cash / 100))
    # mark-to-market identity
    hv = (res.positions * np.where(res.positions != 0, m.close[s:e], 0.0)).sum(axis=1)
    tc.assertTrue(np.allclose(d["cash"].to_numpy() + hv, d["nav"].to_numpy(), rtol=1e-12, atol=tol))
    # feasibility
    if not c.allow_borrowing:
        tc.assertGreaterEqual(d["cash"].min(), -tol)
    tc.assertGreaterEqual(res.positions.min(), -1e-9)
    tc.assertTrue((o["filled_units"].abs() <= o["cash_eligible_units"].abs() + 1e-9).all())
    tc.assertTrue((o["cash_eligible_units"].abs() <= o["liq_eligible_units"].abs() + 1e-9).all())
    tc.assertTrue((o["liq_eligible_units"].abs() <= o["intent_units"].abs() + 1e-9).all())
    tc.assertTrue((np.sign(o["filled_units"]) * np.sign(o["intent_units"]) >= 0).all())
    # parents reconcile with fills
    pf = o.groupby("parent_id")["filled_units"].sum()
    p = res.parents.set_index("parent_id")
    tc.assertTrue(np.allclose(p.loc[pf.index, "filled"], pf, atol=1e-9 * c.initial_cash / 100))
    tc.assertTrue(np.allclose(p["q"], p["filled"] + p["unfilled"], atol=1e-6 * c.initial_cash / 100))
    tc.assertTrue((p["close_session"] >= p["open_session"]).all())


class Invariants(unittest.TestCase):
    def test_invariants_across_configurations_and_scales(self):
        m = synth()
        t = rotating_targets(m)
        for scale in (1e5, 1e7, 1e10):
            for kw in (dict(), dict(units="integer", min_trade_notional=500.0), dict(cap_enabled=False), dict(cost=NOCOST),
                       dict(residual_policy="cancel"), dict(residual_policy="legacy_additive"), dict(cash_rule="legacy_sequential"),
                       dict(valuation="open"), dict(allow_borrowing=True, cap_enabled=False),
                       dict(cost=CostParams(a_bps=30.0, b=1.0, c=2.0, half_spread_bps=2.0))):
                with self.subTest(scale=scale, **{k: str(v) for k, v in kw.items()}):
                    res = engine.run(m, cfg(m, initial_cash=scale, **kw), targets=t)
                    check_invariants(self, m, res)

    def test_participation_limit_enforced(self):
        m = synth(dollar_volume=2e5)
        c = cfg(m, initial_cash=1e7, max_participation=0.05)
        res = engine.run(m, c, targets=rotating_targets(m))
        from ledger_consistent_etf_trading.v2.data import build_features
        f = build_features(m, "lagged", "mean_dollar")
        col = {t: i for i, t in enumerate(m.tickers)}
        cap = np.array([0.05 * f.adv[s, col[t]] for s, t in zip(res.orders["session"], res.orders["ticker"])])
        self.assertTrue((res.orders["notional"].abs().to_numpy() <= cap * (1 + 1e-12)).all())
        self.assertGreater(res.orders["limited_by"].str.contains("liquidity").mean(), 0.5, "fixture must actually bind")
        self.assertLessEqual(res.orders["participation"].max(), 0.05 * (1 + 1e-12))

    def test_integer_units_and_min_trade(self):
        m = synth()
        res = engine.run(m, cfg(m, units="integer", min_trade_notional=2000.0), targets=rotating_targets(m))
        f = res.orders["filled_units"]
        self.assertTrue(np.all(f == np.round(f)))
        self.assertTrue((res.orders.loc[f != 0, "notional"].abs() >= 2000.0).all())


class Residuals(unittest.TestCase):
    def setUp(self):
        self.m = synth(T=60, flat=True, dollar_volume=1.2e5)     # constant price 100; 5% ADV about $6k-$7.8k a day
        self.w = weights(self.m, AAA=0.5)
        self.t = {26: self.w, 31: self.w, 36: self.w, 41: self.w}

    def test_replace_never_overshoots(self):
        res = engine.run(self.m, cfg(self.m, initial_cash=1e5, cost=NOCOST), targets=self.t)
        col = self.m.tickers.index("AAA")
        self.assertLessEqual(res.positions[:, col].max(), 500 + 1e-9)
        self.assertAlmostEqual(res.positions[-1, col], 500, places=6)
        self.assertGreater((res.parents["close_reason"] == "replaced").sum(), 0, "fixture must carry a residual across a target event")

    def test_legacy_additive_diagnostic_overshoots(self):
        res = engine.run(self.m, cfg(self.m, initial_cash=1e5, cost=NOCOST, residual_policy="legacy_additive"), targets=self.t)
        self.assertGreater(res.positions[:, self.m.tickers.index("AAA")].max(), 500 + 1)

    def test_cancel_does_not_carry_between_targets(self):
        res = engine.run(self.m, cfg(self.m, initial_cash=1e5, cost=NOCOST, residual_policy="cancel"), targets=self.t)
        self.assertEqual(set(res.orders["session"]), set(self.t), "orders may exist only on target sessions")
        self.assertTrue((res.parents["close_reason"].isin(["cancelled_eod"])).all())

    def test_order_reconciliation_after_target_change(self):
        t = {26: weights(self.m, AAA=0.8), 28: weights(self.m, BBB=0.8)}     # reversed before AAA completes
        res = engine.run(self.m, cfg(self.m, initial_cash=1e5, cost=NOCOST), targets=t)
        p = res.parents
        first = p[(p["ticker"] == "AAA") & (p["open_session"] == 26)].iloc[0]
        self.assertEqual(first["close_reason"], "replaced")
        self.assertGreater(first["unfilled"], 0)
        second = p[(p["ticker"] == "AAA") & (p["open_session"] == 28)].iloc[0]
        self.assertAlmostEqual(second["q"], -first["filled"], places=6, msg="new parent is target minus current; stale demand is not added")
        self.assertEqual(second["reissued"], 0.0)
        self.assertAlmostEqual(res.positions[-1, self.m.tickers.index("AAA")], 0.0, places=6)
        check_invariants(self, self.m, res)

    def test_reissued_demand_is_not_counted_twice(self):
        from ledger_consistent_etf_trading.v2.metrics import order_summary
        res = engine.run(self.m, cfg(self.m, initial_cash=1e5, cost=NOCOST), targets=self.t)
        s = order_summary(res.parents)
        self.assertAlmostEqual(s["new_demand_notional"], 50_000, delta=1.0)       # one economic demand of $50k
        self.assertGreater(s["intended_notional_gross"], 50_000 * 1.2)             # gross parent notional repeats it
        self.assertAlmostEqual(s["completion_aggregate"], 1.0, places=6)


class CashRule(unittest.TestCase):
    def rotation(self, tickers, held, target, rule):
        m = synth(T=40, tickers=tickers, flat=True, dollar_volume=1e9)
        t = {26: weights(m, **{held: 1.0}), 30: weights(m, **{target: 1.0})}
        res = engine.run(m, cfg(m, initial_cash=1e5, cost=NOCOST, cap_enabled=False, cash_rule=rule), targets=t)
        k = 30 - res.info["start"]
        return m, res, res.positions[k, m.tickers.index(target)], res.daily["cash"].iloc[k]

    def test_funded_rotation_completes_in_one_session_either_direction(self):
        for held, target in (("AAA", "BBB"), ("BBB", "AAA")):
            m, res, pos, cash = self.rotation(("AAA", "BBB"), held, target, "sells_first_proportional")
            self.assertAlmostEqual(pos, 1000.0, places=6)
            self.assertAlmostEqual(cash, 0.0, places=6)

    def test_legacy_sequential_diagnostic_is_order_dependent(self):
        _, _, pos_ab, _ = self.rotation(("AAA", "BBB"), "AAA", "BBB", "legacy_sequential")
        _, _, pos_ba, cash_ba = self.rotation(("AAA", "BBB"), "BBB", "AAA", "legacy_sequential")
        self.assertAlmostEqual(pos_ab, 1000.0, places=6)
        self.assertAlmostEqual(pos_ba, 0.0, places=6)
        self.assertAlmostEqual(cash_ba, 1e5, places=6)

    def test_invariance_to_ticker_processing_order(self):
        """Relabel assets so the processing order reverses; economic results per asset must be identical."""
        a = synth(seed=11)
        names = list(a.tickers)
        relabel = dict(zip(names, [f"Z{len(names) - i}" for i in range(len(names))]))       # reverses sort order
        f = lambda arr: pd.DataFrame(arr, index=a.dates, columns=[relabel[t] for t in names])
        b = market_from_frames(f(a.open), f(a.close), f(a.volume))
        ta = rotating_targets(a)
        tb = {d: np.array([w[names.index(n)] for n in sorted(names, key=lambda n: relabel[n])]) for d, w in ta.items()}
        for rule, expect_equal in (("sells_first_proportional", True), ("legacy_sequential", False)):
            kw = dict(initial_cash=1e6, cash_rule=rule, cap_enabled=False, valuation="prior_close")
            ra, rb = engine.run(a, cfg(a, **kw), targets=ta), engine.run(b, cfg(b, **kw), targets=tb)
            same = np.allclose(ra.daily["nav"].to_numpy(), rb.daily["nav"].to_numpy(), rtol=1e-12, atol=1e-6)
            pos_b = rb.positions[:, [rb.tickers.index(relabel[n]) for n in names]]
            same = same and np.allclose(ra.positions, pos_b, rtol=1e-10, atol=1e-8)
            self.assertEqual(same, expect_equal, rule)
            if rule == "sells_first_proportional":
                self.assertGreater(ra.orders["limited_by"].str.contains("cash").sum(), 0, "fixture must exercise the cash gate")


if __name__ == "__main__":
    unittest.main()
