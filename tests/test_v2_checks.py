"""Run validator: a clean run passes; corrupting any one field of any tape is rejected."""
import copy
import unittest

import numpy as np

from ledger_consistent_etf_trading.v2 import checks, engine
from ledger_consistent_etf_trading.v2.config import ScenarioConfig
from ledger_consistent_etf_trading.v2.data import build_features
from ledger_consistent_etf_trading.v2.errors import RunValidationError
from tests.helpers import synth

M = synth(T=90, seed=21, dollar_volume=2e6)
START = str(M.dates[30].date())
F = build_features(M)


def base(capital=5e7, **kw):
    kw.setdefault("experiment", "E2")
    cfg = ScenarioConfig(name="v", eval_start=START, strategy="momentum_top4", initial_cash=capital, **kw)
    feats = build_features(M, cfg.features, cfg.adv_estimator)
    return engine.run(M, cfg, feats=feats), feats


def validate(res, feats, cfg=None, info=None, production=True, **frames):
    f = dict(daily=res.daily, orders=res.orders, parents=res.parents, demand=res.demand, positions=res.positions, target_eff=res.target_eff)
    f.update(frames)
    return checks.validate(cfg or res.cfg, info or res.info, f["daily"], f["orders"], f["parents"], f["demand"], f["positions"], f["target_eff"], M, feats, production)


class ValidRuns(unittest.TestCase):
    def test_clean_runs_pass_at_every_capital_level(self):
        for capital in (1e7, 1e8, 1e9):
            for kw in (dict(), dict(cap_enabled=False), dict(strategy="equal_weight_weekly"), dict(residual_policy="cancel")):
                kw2 = dict(kw); strat = kw2.pop("strategy", "momentum_top4")
                cfg = ScenarioConfig(name="v", eval_start=START, strategy=strat, initial_cash=capital, experiment="E2", **kw2)
                res = engine.run(M, cfg)
                rep = checks.validate_result(res, M, build_features(M))
                self.assertTrue(rep.ok, rep.failures)
                self.assertGreater(len(rep.checks), 35)
                rep.require()

    def test_fixture_exercises_the_gates(self):
        res, _ = base()
        self.assertGreater(res.orders["limited_by"].str.contains("liquidity").sum(), 10)
        self.assertGreater((res.parents["close_reason"] == "replaced").sum(), 10)
        self.assertGreater(res.orders["cost"].sum(), 0)

    def test_authorized_diagnostic_exceptions(self):
        res, f = base(allow_borrowing=True, cap_enabled=False, experiment="E2b")
        rep = checks.validate_result(res, M, f)
        self.assertTrue(rep.ok, rep.failures)
        self.assertEqual(len(rep.diagnostics_authorized), 1)
        self.assertFalse(validate(res, f, cfg=res.cfg.with_(experiment="E2")).ok, "borrowing outside E2b is unauthorized")
        for kw in (dict(valuation="same_close"), dict(features="same_day"), dict(residual_policy="legacy_additive"),
                   dict(cash_rule="legacy_sequential"), dict(max_trade_notional=1e5), dict(units="integer"), dict(min_trade_notional=500.0)):
            res, f = base(**kw)                                              # experiment E2: none of these is allowed there
            rep = checks.validate_result(res, M, f)
            self.assertIn("diagnostic settings authorized for this experiment", [c["check"] for c in rep.failures], kw)
            self.assertTrue(checks.validate_result(res, M, f, production=False).ok or kw.get("residual_policy") == "legacy_additive")
        for val, feat in (("same_close", "same_day"), ("prior_close", "same_day")):
            res, f = base(valuation=val, features=feat, experiment="E1")
            self.assertTrue(checks.validate_result(res, M, f).ok, "labelled negative controls are valid runs of an invalid convention")

    def test_borrowing_detected_when_not_authorized(self):
        res, f = base(allow_borrowing=True, cap_enabled=False, experiment="E2b", valuation="prior_close")
        if res.daily["cash"].min() < -1.0:
            rep = validate(res, f, cfg=res.cfg.with_(allow_borrowing=False, experiment="E2"))
            self.assertIn("no unauthorized borrowing", [c["check"] for c in rep.failures])


class Corruptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.f = base()
        assert checks.validate_result(cls.res, M, cls.f).ok

    def reject(self, expect, **frames):
        rep = validate(self.res, self.f, **frames)
        self.assertFalse(rep.ok, f"corruption not detected (expected: {expect})")
        self.assertIn(expect, [c["check"] for c in rep.failures])
        with self.assertRaises(RunValidationError):
            rep.require()

    def mut(self, frame, col, row, fn):
        df = getattr(self.res, frame).copy()
        i = df.index[row] if frame == "daily" else row
        df.loc[i, col] = fn(df.loc[i, col])
        return {frame: df}

    def traded_row(self):
        o = self.res.orders
        return int(o.index[(o["filled_units"] != 0) & (o["cost"] > 0)][5])

    def test_daily_tape(self):
        amt = 1e-6 * self.res.cfg.initial_cash
        self.reject("cash conservation (every session)", **self.mut("daily", "cash", 20, lambda v: v + amt))
        self.reject("daily tape finite", **self.mut("daily", "nav", 20, lambda v: np.nan))
        self.reject("NAV = cash + holdings", **self.mut("daily", "nav", 20, lambda v: v + amt))
        self.reject("NAV positive", **self.mut("daily", "nav", 20, lambda v: -v))
        self.reject("cost charged exactly once (orders = daily = parents)", **self.mut("daily", "cost", 20, lambda v: v + amt))
        self.reject("holdings value at current close marks", **self.mut("daily", "holdings_value", 20, lambda v: v + amt))
        self.reject("opening pre-trade NAV", **self.mut("daily", "nav_open", 20, lambda v: v + amt))
        self.reject("gross traded equals summed fills", **self.mut("daily", "gross_traded", 20, lambda v: v + amt))
        self.reject("target-event flags equal the target sessions", **self.mut("daily", "target_event", 20, lambda v: not v))
        self.reject("tape dimensions match the evaluation calendar", daily=self.res.daily.iloc[:-1])

    def test_order_tape(self):
        r = self.traded_row()
        self.reject("cost equals the independent model recalculation", **self.mut("orders", "cost", r, lambda v: v * 1.01))
        self.reject("notional equals units x price", **self.mut("orders", "notional", r, lambda v: v * 1.001))
        self.reject("fill price equals the session open", **self.mut("orders", "fill_price", r, lambda v: v * 1.001))
        self.reject("position reconciliation", **self.mut("orders", "filled_units", r, lambda v: v * 1.01))
        self.reject("every order belongs to one parent", **self.mut("orders", "parent_id", r, lambda v: 10 ** 9))
        self.reject("order sessions and tickers lie in the run", **self.mut("orders", "session", r, lambda v: 10 ** 6))
        self.reject("order and parent agree on the asset", **self.mut("orders", "ticker", r, lambda v: "AAA" if v != "AAA" else "BBB"))
        self.reject("order tape finite", **self.mut("orders", "cost", r, lambda v: np.inf))
        self.reject("fills have the sign of the intent", **self.mut("orders", "intent_units", r, lambda v: -v))
        self.reject("gate hierarchy: fill <= cash-eligible <= liquidity-eligible <= intent", **self.mut("orders", "cash_eligible_units", r, lambda v: v * 0.5))
        self.reject("costs non-negative", **self.mut("orders", "cost", r, lambda v: -v))
        o = self.res.orders.drop(index=r)
        self.reject("cash conservation (every session)", orders=o)

    def test_parent_tape(self):
        p = self.res.parents
        r = int(p.index[p["filled"] != 0][3])
        self.reject("parent quantity = filled + unfilled", **self.mut("parents", "unfilled", r, lambda v: v + 1.0))
        self.reject("parent tape finite, including every unfilled-opportunity mark", **self.mut("parents", "p_H", r, lambda v: np.nan))
        self.reject("parent fills, value and cost equal their orders", **self.mut("parents", "fill_value", r, lambda v: v * 1.01))
        self.reject("parent life and close reason valid", **self.mut("parents", "close_reason", r, lambda v: "vanished"))
        self.reject("parent prices and marks positive", **self.mut("parents", "p_H", r, lambda v: -v))
        self.reject("parents and demand ledger agree on originated and executed demand", **self.mut("parents", "demand_executed_value", r, lambda v: v * 2 + 1e6))
        self.reject("parents open only at target events", **self.mut("parents", "open_session", r, lambda v: 3))
        dup = p.copy(); dup.loc[r, "parent_id"] = dup.loc[r + 1, "parent_id"]
        self.reject("parent ids unique", parents=dup)

    def test_demand_ledger(self):
        d = self.res.demand
        r = int(d.index[d["event"] == "executed"][2])
        self.reject("demand lineage: originated = executed + withdrawn + outstanding", demand=d.drop(index=r))
        self.reject("demand ledger events and units valid", **self.mut("demand", "event", r, lambda v: "invented"))
        self.reject("demand ledger finite", **self.mut("demand", "value", r, lambda v: np.nan))
        ex = d["event"] == "executed"
        factor = 1.5 * d.loc[d["event"] == "originated", "value"].sum() / d.loc[ex, "value"].sum()     # push executed to 150% of originated
        inflated = d.copy(); inflated.loc[ex, "value"] *= factor
        self.reject("completion is a proportion in [0, 1] or undefined", demand=inflated)

    def test_positions_and_targets(self):
        pos = self.res.positions.copy(); pos[40, 0] -= 1e6
        self.reject("no short positions", positions=pos)
        pos = self.res.positions.copy(); pos[40, 1] += 5.0
        self.reject("position reconciliation", positions=pos)
        t = self.res.target_eff.copy(); t[40, 0] = -0.1
        self.reject("target weights valid", target_eff=t)
        t = self.res.target_eff.copy(); t[40:, :] = 0.3
        self.reject("target weights valid", target_eff=t)

    def test_configuration_and_metadata(self):
        rep = validate(self.res, self.f, cfg=self.res.cfg.with_(max_participation=0.01))
        self.assertIn("liquidity-budget compliance", [c["check"] for c in rep.failures])
        info = dict(self.res.info, features_binding="unbound")
        self.assertIn("features bound to the run's market data", [c["check"] for c in validate(self.res, self.f, info=info).failures])
        info = dict(self.res.info, market_id="0" * 64)
        self.assertIn("market identity matches", [c["check"] for c in validate(self.res, self.f, info=info).failures])
        wrong = build_features(M, "same_day")
        self.assertIn("feature mode and estimator match configuration", [c["check"] for c in validate(self.res, wrong).failures])
        rep = validate(self.res, self.f, cfg=self.res.cfg.with_(cost=self.res.cfg.cost.__class__(enabled=False)))
        self.assertIn("no cost charged when costs are disabled", [c["check"] for c in rep.failures])
        rep = validate(self.res, self.f, cfg=self.res.cfg.with_(residual_policy="cancel"))
        self.assertFalse(rep.ok)

    def test_empty_report_is_not_valid(self):
        self.assertFalse(checks.Report("x").ok)


if __name__ == "__main__":
    unittest.main()
