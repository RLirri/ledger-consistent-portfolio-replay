"""Sprint 3C regression tests: invalid inputs, configuration and numerical state must stop a run."""
import unittest

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import engine, metrics
from ledger_consistent_etf_trading.v2.config import CostParams, ScenarioConfig
from ledger_consistent_etf_trading.v2.data import Features, Market, build_features
from ledger_consistent_etf_trading.v2.errors import ScenarioInputError
from tests.helpers import NOCOST, cfg, synth, weights

NAN, INF = float("nan"), float("inf")


class ConfigurationDomain(unittest.TestCase):
    def test_cost_parameters(self):
        for k in ("fees_bps", "half_spread_bps", "a_bps", "b", "c", "vol_ref"):
            for bad in (NAN, INF, -INF, -1.0, "1", None, True):
                with self.subTest(k=k, bad=bad), self.assertRaises(ScenarioInputError):
                    CostParams(**{k: bad})
        with self.assertRaises(ScenarioInputError):
            CostParams(vol_ref=0.0)
        with self.assertRaises(ScenarioInputError):
            CostParams(enabled=1)
        CostParams(b=0.0, c=0.0, a_bps=0.0, half_spread_bps=0.0)      # legitimate zero-cost boundary

    def test_scenario_parameters(self):
        bad = dict(initial_cash=(NAN, INF, 0.0, -1.0, "1e6"), max_participation=(NAN, 0.0, -0.1, 1.5), min_trade_notional=(NAN, -1.0, INF),
                   max_trade_notional=(NAN, 0.0, -5.0, INF), momentum_lookback=(0, -1, 2.5, True), momentum_top=(0, 1.5),
                   max_sessions=(0, -3, 2.5), valuation=("close", ""), features=("lead",), residual_policy=("carry",), cash_rule=("fifo",),
                   units=("shares",), strategy=("alpha",), adv_estimator=("median",), cap_enabled=(1, "yes"), allow_borrowing=(0,),
                   name=("", 5), eval_start=("",), cost=({"enabled": True},))
        for k, values in bad.items():
            for v in values:
                with self.subTest(k=k, v=v), self.assertRaises(ScenarioInputError):
                    ScenarioConfig(**{"name": "x", k: v})
        self.assertTrue(issubclass(ScenarioInputError, ValueError))


class MarketAndTargets(unittest.TestCase):
    def setUp(self):
        self.m = synth()
        self.t = {26: weights(self.m, AAA=0.6, BBB=0.4)}

    def test_inconsistent_calendar_or_shapes(self):
        m = self.m
        dup = Market(m.dates[:30].append(m.dates[29:69]), m.tickers, m.open, m.close, m.volume, {})
        rev = Market(m.dates[::-1], m.tickers, m.open, m.close, m.volume, {})
        naive = Market(m.dates.tz_localize(None), m.tickers, m.open, m.close, m.volume, {})
        short = Market(m.dates, m.tickers, m.open[:-1], m.close, m.volume, {})
        unsorted = Market(m.dates, m.tickers[::-1], m.open, m.close, m.volume, {})
        for name, bad in (("duplicate", dup), ("descending", rev), ("naive", naive), ("shape", short), ("tickers", unsorted)):
            with self.subTest(name=name), self.assertRaises(ScenarioInputError):
                engine.run(bad, cfg(m), targets=self.t)

    def test_target_weights_and_sessions(self):
        m = self.m
        for w in (weights(m, AAA=0.8, BBB=0.4), weights(m, AAA=-0.1), weights(m, AAA=NAN), weights(m, AAA=INF), np.array([0.5, 0.5])):
            with self.assertRaises(ScenarioInputError):
                engine.run(m, cfg(m), targets={26: w})
        for key in (3, 24, 70, 1000, -1, 26.0, "26", True):
            with self.subTest(key=key), self.assertRaises(ScenarioInputError):
                engine.run(m, cfg(m), targets={key: self.t[26]})

    def test_every_required_price(self):
        cases = [("open", 26, 0, dict()),                       # fill price
                 ("close", 25, 0, dict()),                      # prior-close sizing price
                 ("close", 30, 0, dict()),                      # mark of a held asset
                 ("open", 30, 0, dict()),                       # opening pre-trade NAV of a held asset
                 ("close", 26, 0, dict(valuation="same_close")),
                 ("close", 69, 0, dict(cap_enabled=True, max_participation=1e-6)),   # end-of-sample mark of an unfilled order
                 ("close", 26, 0, dict(residual_policy="cancel", max_participation=1e-6))]
        for field, d, i, kw in cases:
            for bad in (NAN, 0.0, -3.0, INF):
                m = self.m.copy(); getattr(m, field)[d, i] = bad
                with self.subTest(field=field, d=d, bad=bad, **kw), self.assertRaises(ScenarioInputError):
                    engine.run(m, cfg(m, **kw), targets=self.t)

    def test_unneeded_missing_price_is_not_an_error(self):
        m = self.m.copy(); m.close[40, 4] = NAN; m.open[40, 4] = NAN      # EEE is never targeted or held
        engine.run(m, cfg(m), targets=self.t)

    def test_adv_and_volatility(self):
        for arr, kw in (("adv", dict(cap_enabled=True, cost=NOCOST)), ("adv", dict(cap_enabled=False)), ("vol", dict(cap_enabled=False))):
            for bad in (NAN, 0.0, -1.0, INF):
                f = build_features(self.m); f.market_id = None     # final remediation: edited features must use the unbound synthetic route
                getattr(f, arr)[26, 0] = bad
                with self.subTest(arr=arr, bad=bad), self.assertRaises(ScenarioInputError):
                    engine.run(self.m, cfg(self.m, **kw), targets=self.t, feats=f)
        f = build_features(self.m); f.market_id = None; f.adv[26, 0] = NAN; f.vol[26, 0] = NAN
        engine.run(self.m, cfg(self.m, cap_enabled=False, cost=NOCOST), targets=self.t, feats=f)   # nothing configured consumes them


class FeatureContract(unittest.TestCase):
    def setUp(self):
        self.m = synth()
        self.t = {26: weights(self.m, AAA=0.5)}

    def test_wrong_mode_estimator_shape_or_source(self):
        m = self.m
        with self.assertRaises(ScenarioInputError):
            engine.run(m, cfg(m), targets=self.t, feats=build_features(m, "same_day"))
        with self.assertRaises(ScenarioInputError):
            engine.run(m, cfg(m), targets=self.t, feats=build_features(m, "lagged", "shares_x_close"))
        other = synth(seed=99)
        with self.assertRaises(ScenarioInputError):
            engine.run(m, cfg(m), targets=self.t, feats=build_features(other))
        f = build_features(m)
        with self.assertRaises(ScenarioInputError):
            engine.run(m, cfg(m), targets=self.t, feats=Features(f.vol[:-1], f.adv[:-1], "lagged", "mean_dollar"))

    def test_same_day_data_relabelled_as_lagged_is_caught_by_identity_only_when_bound(self):
        m = self.m
        same = build_features(m, "same_day")
        forged = Features(same.vol, same.adv, "lagged", "mean_dollar")          # unbound: no market identity
        res = engine.run(m, cfg(m), targets=self.t, feats=forged)
        self.assertEqual(res.info["features_binding"], "unbound", "an unbound feature set must be visible to the run validator")
        self.assertEqual(engine.run(m, cfg(m), targets=self.t).info["features_binding"], "bound")

    def test_labelled_negative_controls_still_run(self):
        for val, feat in (("prior_close", "same_day"), ("open", "same_day"), ("same_close", "lagged"), ("same_close", "same_day")):
            res = engine.run(self.m, cfg(self.m, valuation=val, features=feat, experiment="E1"), targets=self.t)
            self.assertFalse(res.info["information"]["valid"])
            self.assertEqual(res.info["information"]["role"], "negative_control")


class MetricsFailClosed(unittest.TestCase):
    def test_missing_mark_is_not_summed_away(self):
        p = pd.DataFrame([dict(q=10.0, filled=4.0, unfilled=6.0, p_D=100.0, p_A=100.0, p_H=NAN, fill_value=400.0, cost=0.0, open_session=1, close_session=2)])
        with self.assertRaises(ScenarioInputError):
            metrics.order_shortfall(p)

    def test_relative_refuses_mismatched_calendars_and_bad_nav(self):
        a = pd.Series([100.0, 101.0, 102.0], index=pd.RangeIndex(3))
        with self.assertRaises(ScenarioInputError):
            metrics.relative(a, pd.Series([100.0, 101.0, 102.0], index=pd.RangeIndex(1, 4)))
        for bad in (NAN, 0.0, -1.0):
            with self.assertRaises(ScenarioInputError):
                metrics.relative(a, pd.Series([100.0, bad, 102.0], index=pd.RangeIndex(3)))
            with self.assertRaises(ScenarioInputError):
                metrics.performance(pd.Series([100.0, bad, 102.0]))

    def test_annualized_terminal_gap(self):
        # reference doubles, arm ends 1% lower, over exactly 252 returns: annualized geometric gap = 1/0.99 - 1
        idx = pd.RangeIndex(253)
        ref = pd.Series(np.linspace(100, 200, 253), idx); arm = ref.copy(); arm.iloc[-1] *= 0.99
        g = metrics.relative(arm, ref)
        self.assertAlmostEqual(g["terminal_gap_bps"], 100.0, places=9)
        self.assertAlmostEqual(g["terminal_gap_ann_bps"], 1e4 * (1 / 0.99 - 1), places=9)

    def test_summary_has_no_nan_and_names_undefined_metrics(self):
        import json
        m = synth(flat=True)
        res = engine.run(m, cfg(m, cost=NOCOST, cap_enabled=False), targets={})       # no trade, zero variance
        s = metrics.summarize(res, m.close)
        text = json.dumps(s, allow_nan=False)                                          # raises if any NaN/inf remains
        self.assertIsNone(s["sharpe_rf0"]); self.assertEqual(s["undefined"]["sharpe_rf0"], "zero return variance")
        self.assertIn("completion_aggregate", s["undefined"]); self.assertNotIn("NaN", text)
        self.assertEqual(s["cost_dollars"], 0.0)


if __name__ == "__main__":
    unittest.main()
