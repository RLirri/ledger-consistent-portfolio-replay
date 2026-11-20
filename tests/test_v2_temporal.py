"""Information contract. Only contract-valid configurations may pass the future-data perturbation
tests; the four intentionally invalid configurations are negative controls and must be detected."""
import unittest

import numpy as np

from ledger_consistent_etf_trading.v2 import engine
from ledger_consistent_etf_trading.v2.config import information_class
from ledger_consistent_etf_trading.v2.strategies import momentum_top4
from tests.helpers import cfg, synth, weights

GRID = [(v, f) for v in ("prior_close", "open", "same_close") for f in ("lagged", "same_day")]
D = 40          # perturbed session (a target session below)


def run(mkt, val, feat, **kw):
    c = cfg(mkt, valuation=val, features=feat, initial_cash=1e6, max_participation=0.05, **kw)
    s = c and 25
    w1 = weights(mkt, AAA=0.5, BBB=0.3, CCC=0.2)
    w2 = weights(mkt, AAA=0.1, BBB=0.2, DDD=0.4, EEE=0.3)
    targets = {26: w1, 33: w2, D: w1, 47: w2}
    return engine.run(mkt, c, targets=targets)


def state_at(res, d):
    o = res.orders[res.orders["session"] == d][["ticker", "intent_units", "filled_units", "cost"]].round(10)
    return o.to_dict("records"), round(float(res.daily["cash"].iloc[d - res.info["start"]]), 6)


class InformationContract(unittest.TestCase):
    def test_classification(self):
        cls = {g: information_class(*g) for g in GRID}
        self.assertEqual([g for g in GRID if cls[g]["valid"]], [("prior_close", "lagged"), ("open", "lagged")])
        self.assertEqual(cls[("prior_close", "lagged")]["role"], "primary")
        self.assertEqual(cls[("open", "lagged")]["role"], "sensitivity")
        self.assertEqual(sum(c["role"] == "negative_control" for c in cls.values()), 4)

    def test_future_close_and_volume_perturbation(self):
        base = synth()
        pert = base.copy()
        pert.close[D] *= 1.31
        pert.volume[D] *= 0.2
        for val, feat in GRID:
            with self.subTest(valuation=val, features=feat):
                a, b = state_at(run(base, val, feat), D), state_at(run(pert, val, feat), D)
                if information_class(val, feat)["valid"]:
                    self.assertEqual(a, b, "contract-valid configuration reacted to same-session close/volume")
                else:
                    self.assertNotEqual(a, b, "negative control was not detected by the perturbation")

    def test_later_sessions_cannot_change_earlier_state(self):
        base = synth()
        pert = base.copy()
        pert.close[D + 1:] *= 1.5; pert.open[D + 1:] *= 1.5; pert.volume[D + 1:] *= 3
        for val, feat in GRID:
            with self.subTest(valuation=val, features=feat):
                a, b = run(base, val, feat), run(pert, val, feat)
                k = D - a.info["start"] + 1
                self.assertTrue(np.allclose(a.daily["nav"].to_numpy()[:k], b.daily["nav"].to_numpy()[:k], rtol=0, atol=0))
                self.assertTrue(a.orders[a.orders["session"] <= D].equals(b.orders[b.orders["session"] <= D]))

    def test_truncated_input_equals_full_input(self):
        full = synth()
        import pandas as pd
        from ledger_consistent_etf_trading.v2.data import market_from_frames
        f = lambda arr: pd.DataFrame(arr[:D + 1], index=full.dates[:D + 1], columns=full.tickers)
        cut = market_from_frames(f(full.open), f(full.close), f(full.volume))      # exact prefix of the full input
        for val, feat in [g for g in GRID if information_class(*g)["valid"]]:
            a = run(full, val, feat)
            c = cfg(cut, valuation=val, features=feat, initial_cash=1e6)
            w1 = weights(cut, AAA=0.5, BBB=0.3, CCC=0.2); w2 = weights(cut, AAA=0.1, BBB=0.2, DDD=0.4, EEE=0.3)
            b = engine.run(cut, c, targets={26: w1, 33: w2, D: w1})
            self.assertEqual(state_at(a, D), state_at(b, D))

    def test_open_price_role(self):
        base = synth()
        pert = base.copy(); pert.open[D, 0] *= 1.08          # one asset: a uniform shift would cancel in weight-based sizing
        pc = [state_at(run(m, "prior_close", "lagged"), D)[0] for m in (base, pert)]
        self.assertEqual([r["intent_units"] for r in pc[0]], [r["intent_units"] for r in pc[1]],
                         "prior-close quantities must be fixed before the open")
        op = [state_at(run(m, "open", "lagged"), D)[0] for m in (base, pert)]
        self.assertNotEqual([r["intent_units"] for r in op[0]], [r["intent_units"] for r in op[1]],
                            "observed-open sizing is expected to depend on the open (disclosed approximation)")

    def test_prior_session_data_is_used(self):
        base = synth(); pert = base.copy(); pert.close[D - 1] *= 1.2
        self.assertNotEqual(state_at(run(base, "prior_close", "lagged"), D), state_at(run(pert, "prior_close", "lagged"), D))

    def test_momentum_signal_has_no_same_session_dependence(self):
        base = synth(); pert = base.copy()
        pert.close[D] *= np.array([3.0, 0.2, 1.0, 5.0, 0.1]); pert.open[D] *= 2; pert.volume[D] *= 9
        a, b = momentum_top4(base, 25, 60), momentum_top4(pert, 25, 60)
        for d in range(26, D + 1):
            self.assertTrue(np.array_equal(a[d], b[d]))
        self.assertFalse(np.array_equal(a[D + 1], b[D + 1]))


class FailClosed(unittest.TestCase):
    def setUp(self):
        self.m = synth()
        self.t = {26: weights(self.m, AAA=0.6, BBB=0.4)}

    def test_bad_adv_with_cap_raises(self):
        for bad in (np.nan, 0.0, -5.0, np.inf):
            m = self.m.copy(); m.volume[:26, 0] = bad if np.isfinite(bad) else bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                engine.run(m, cfg(m, cap_enabled=True), targets=self.t)

    def test_bad_inputs_with_costs_raise(self):
        m = self.m.copy(); m.volume[:26, 1] = np.nan
        with self.assertRaises(ValueError):
            engine.run(m, cfg(m, cap_enabled=False), targets=self.t)
        m = self.m.copy(); m.close[5:26, 0] = m.close[5, 0]        # zero volatility cannot scale the cost model
        with self.assertRaises(ValueError):
            engine.run(m, cfg(m, cap_enabled=False), targets=self.t)

    def test_bad_prices_raise(self):
        for field, d in (("open", 26), ("close", 25), ("close", 30)):
            m = self.m.copy(); getattr(m, field)[d, 0] = np.nan
            with self.subTest(field=field, d=d), self.assertRaises(ValueError):
                engine.run(m, cfg(m), targets=self.t)

    def test_invalid_weights_raise(self):
        for w in (weights(self.m, AAA=0.8, BBB=0.4), weights(self.m, AAA=-0.1), weights(self.m, AAA=np.nan)):
            with self.assertRaises(ValueError):
                engine.run(self.m, cfg(self.m), targets={26: w})


if __name__ == "__main__":
    unittest.main()
