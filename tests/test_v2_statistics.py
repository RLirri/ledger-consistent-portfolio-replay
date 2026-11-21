"""Bootstrap, contrast recipes and comparator matching (synthetic known-answer tests)."""
import unittest

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import bootstrap, contrasts, engine, registry
from ledger_consistent_etf_trading.v2.comparator import posthoc_cost_adjusted
from ledger_consistent_etf_trading.v2.config import CostParams
from ledger_consistent_etf_trading.v2.data import build_features
from ledger_consistent_etf_trading.v2.errors import ScenarioInputError
from tests.helpers import NOCOST, cfg, synth, weights


def naive_indices(n, replicates, mean_block, seed):
    """Independent re-implementation of the documented draw order, written without reference to the module's loop."""
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for _ in range(replicates):
        starts = rng.integers(0, n, size=n); u = rng.random(n)
        seq = [int(starts[0])]
        for t in range(1, n):
            seq.append(int(starts[t]) if u[t] < 1.0 / mean_block else (seq[-1] + 1) % n)
        rows.append(seq)
    return np.array(rows)


def nav_from(r, start=100.0):
    return pd.Series(start * np.concatenate([[1.0], np.cumprod(1 + np.asarray(r))]))


class StationaryBootstrap(unittest.TestCase):
    def test_indices_match_an_independent_implementation_and_are_reproducible(self):
        a = bootstrap.stationary_indices(37, 25, 5, 20261119)
        self.assertTrue(np.array_equal(a, naive_indices(37, 25, 5, 20261119)))
        self.assertTrue(np.array_equal(a, bootstrap.stationary_indices(37, 25, 5, 20261119)))
        self.assertFalse(np.array_equal(a, bootstrap.stationary_indices(37, 25, 5, 1)))
        self.assertTrue(a.min() >= 0 and a.max() <= 36)

    def test_blocks_are_consecutive_and_wrap_around(self):
        idx = bootstrap.stationary_indices(50, 200, 10, 7)
        step = (idx[:, 1:] - idx[:, :-1]) % 50
        self.assertGreater((step == 1).mean(), 0.85)                       # about 1 - 1/10 plus chance continuations
        wraps = (idx[:, :-1] == 49) & (idx[:, 1:] == 0)
        self.assertGreater(wraps.sum(), 0, "circular wrap must occur")
        lengths = []
        for row in idx:
            run = 1
            for s in (row[1:] - row[:-1]) % 50:
                if s == 1:
                    run += 1
                else:
                    lengths.append(run); run = 1
        self.assertAlmostEqual(np.mean(lengths), 10, delta=2.0)
        one = bootstrap.stationary_indices(30, 5, 1, 3)                     # mean block 1 = iid resampling of single sessions
        self.assertEqual(one.shape, (5, 30))

    def test_identical_arms_give_exactly_zero(self):
        rng = np.random.default_rng(1)
        nav = nav_from(rng.normal(0.0004, 0.01, 300))
        out = bootstrap.paired_bootstrap(nav, nav, replicates=200)
        for name, s in out["statistics"].items():
            self.assertEqual((s["point"], s["lower"], s["upper"]), (0.0, 0.0, 0.0), name)

    def test_constant_return_difference_is_recovered_exactly_by_pairing(self):
        rng = np.random.default_rng(2)
        rb = rng.normal(0.0003, 0.012, 400)
        ra = rb + 0.0002                                                    # arm a earns 2 bps a day more, every day
        out = bootstrap.paired_bootstrap(nav_from(ra), nav_from(rb), replicates=300)
        s = out["statistics"]["active_return_ann"]
        for k in ("point", "lower", "upper"):
            self.assertAlmostEqual(s[k], 252 * 0.0002, places=10)
        self.assertAlmostEqual(s["bootstrap_sd"], 0.0, places=10)
        # resampling the arms independently would not have this property: pairing is what removes the common variation
        idx = bootstrap.stationary_indices(400, 300, 20, 20261119); jdx = bootstrap.stationary_indices(400, 300, 20, 99)
        unpaired = np.array([252 * (ra[i].mean() - rb[j].mean()) for i, j in zip(idx, jdx)])
        self.assertGreater(unpaired.std(), 0.01)

    def test_interval_matches_independent_computation(self):
        rng = np.random.default_rng(3)
        common = rng.normal(0.0003, 0.01, 250)
        ra, rb = common + rng.normal(0, 0.002, 250), common + rng.normal(0, 0.002, 250)
        out = bootstrap.paired_bootstrap(nav_from(ra), nav_from(rb), replicates=400, mean_block=20, seed=11, level=0.9)
        idx = naive_indices(250, 400, 20, 11)
        draws = np.array([252 * (ra[i] - rb[i]).mean() for i in idx])
        s = out["statistics"]["active_return_ann"]
        self.assertAlmostEqual(s["lower"], np.quantile(draws, 0.05), places=10)
        self.assertAlmostEqual(s["upper"], np.quantile(draws, 0.95), places=10)
        sh = np.array([np.sqrt(252) * (ra[i].mean() / ra[i].std() - rb[i].mean() / rb[i].std()) for i in idx])
        self.assertAlmostEqual(out["statistics"]["sharpe_difference"]["upper"], np.quantile(sh, 0.95), places=10)
        tg = np.array([np.exp((np.log1p(rb[i]).sum() - np.log1p(ra[i]).sum()) * 252 / 250) - 1 for i in idx])
        self.assertAlmostEqual(out["statistics"]["terminal_gap_ann"]["lower"], np.quantile(tg, 0.05), places=10)
        self.assertLessEqual(s["lower"], s["point"] + 0.2); self.assertEqual(out["level"], 0.9)

    def test_undefined_sharpe_is_reported_not_dropped(self):
        flat = nav_from(np.zeros(100)); other = nav_from(np.random.default_rng(4).normal(0, 0.01, 100))
        s = bootstrap.paired_bootstrap(flat, other, replicates=50)["statistics"]
        self.assertIsNone(s["sharpe_difference"]["point"]); self.assertIsNone(s["sharpe_difference"]["lower"])
        self.assertEqual(s["sharpe_difference"]["undefined_replicates"], 50)
        self.assertIsNotNone(s["active_return_ann"]["lower"])

    def test_invalid_inputs(self):
        good = nav_from(np.full(20, 0.01))
        for bad in (pd.Series([100.0, np.nan, 101.0, 102.0]), pd.Series([100.0, 0.0, 101.0, 102.0]), pd.Series([100.0, 101.0])):
            with self.assertRaises(ScenarioInputError):
                bootstrap.paired_bootstrap(bad, bad)
        with self.assertRaises(ScenarioInputError):
            bootstrap.paired_bootstrap(good, good.iloc[:-1])
        shifted = good.copy(); shifted.index = shifted.index + 1
        with self.assertRaises(ScenarioInputError):
            bootstrap.paired_bootstrap(good, shifted)
        with self.assertRaises(ScenarioInputError):
            bootstrap.stationary_indices(10, 5, 0.5, 1)


class ContrastRecipes(unittest.TestCase):
    def test_every_contrast_is_matched_and_tiered(self):
        reg = {c.name: c for c in registry.build()}
        cs = contrasts.build()
        self.assertEqual(len(cs), 66)
        self.assertEqual(sum(c["tier"] == "primary" for c in cs), 12)
        for c in cs:
            self.assertIn(c["arm"], reg); self.assertIn(c["reference"], reg)
            self.assertLessEqual(contrasts.differing_fields(reg[c["arm"]], reg[c["reference"]]) - {"name"}, set(c["allowed"]))
        e1 = [c for c in cs if c["experiment"] == "E1"]
        self.assertEqual({c["reference"] for c in e1}, {"E1_prior_close_lagged"})
        self.assertEqual(sum(c["tier"] == "diagnostic" for c in e1), 4, "negative controls are diagnostic only")
        self.assertEqual({c["arm"] for c in cs} | {c["reference"] for c in cs}, set(reg), "every registered scenario appears in a contrast")

    def test_unintended_difference_is_rejected(self):
        reg = {c.name: c for c in registry.build()}
        a, b = reg["E2_C_1b_cost1_cap1"], reg["E2_C_1b_cost0_cap0"]
        contrasts.check_matched(a, b, {"cost.enabled", "cap_enabled"})
        for kw in (dict(initial_cash=2e9), dict(strategy="equal_weight_weekly"), dict(valuation="open"), dict(units="integer"),
                   dict(cash_rule="legacy_sequential"), dict(min_trade_notional=5.0), dict(eval_start="2011-01-03"), dict(cost=CostParams(a_bps=20.0))):
            with self.subTest(**{k: str(v) for k, v in kw.items()}), self.assertRaises(ScenarioInputError):
                contrasts.check_matched(a.with_(**kw), b, {"cost.enabled", "cap_enabled"})


class ComparatorGuards(unittest.TestCase):
    def setUp(self):
        self.m = synth(dollar_volume=4e5)
        self.t = {d: weights(self.m, **{self.m.tickers[(d // 3) % 5]: 0.7}) for d in range(26, 60, 3)}
        self.base = cfg(self.m, initial_cash=2e6, cost=NOCOST, cap_enabled=False)
        self.ref = engine.run(self.m, self.base, targets=self.t)
        self.f = build_features(self.m)
        self.arm = self.base.with_(cost=CostParams())

    def test_matched_reference_accepted_with_and_without_limit(self):
        out = posthoc_cost_adjusted(self.ref, self.m, self.f, self.arm)
        self.assertEqual(out.attrs["reference_scenario_id"], self.ref.cfg.scenario_id)
        posthoc_cost_adjusted(self.ref, self.m, self.f, self.arm.with_(cap_enabled=True))

    def test_mismatched_reference_rejected(self):
        for kw in (dict(initial_cash=1e6), dict(strategy="momentum_top4"), dict(valuation="open"), dict(cash_rule="legacy_sequential"),
                   dict(units="integer"), dict(min_trade_notional=10.0), dict(residual_policy="cancel"), dict(allow_borrowing=True),
                   dict(eval_start=str(self.m.dates[30].date())), dict(max_sessions=20)):
            with self.subTest(**kw), self.assertRaises(ScenarioInputError):
                posthoc_cost_adjusted(self.ref, self.m, self.f, self.arm.with_(**kw))
        other = synth(seed=3, dollar_volume=4e5)
        with self.assertRaises(ScenarioInputError):
            posthoc_cost_adjusted(self.ref, other, build_features(other), self.arm)

    def test_non_positive_growth_is_rejected(self):
        ruinous = self.arm.with_(cost=CostParams(half_spread_bps=20_000.0))     # 200% of traded notional
        with self.assertRaises(ScenarioInputError):
            posthoc_cost_adjusted(self.ref, self.m, self.f, ruinous)


if __name__ == "__main__":
    unittest.main()


class RuntimeProfile(unittest.TestCase):
    def test_phase_accounting_on_synthetic_data(self):
        from ledger_consistent_etf_trading.v2 import profile
        m = synth(T=120, seed=5)
        c = cfg(m, strategy="momentum_top4", initial_cash=5e7)
        p = profile.profile(m, c, repetitions=3)
        self.assertEqual(p["repetitions"], 3)
        self.assertEqual(p["events"]["sessions"], 95); self.assertEqual(p["events"]["assets"], 5)
        self.assertTrue(p["events"]["valid"]); self.assertGreater(p["events"]["order_rows"], 0)
        ph = p["phases"]
        parts = sum(ph[k]["cpu_seconds_median"] for k in ("features", "replay", "validation", "metrics"))
        self.assertLessEqual(parts, ph["total"]["cpu_seconds_max"] * 1.5 + 0.05)
        for k in ph:
            self.assertLessEqual(ph[k]["wall_seconds_min"], ph[k]["wall_seconds_median"]); self.assertLessEqual(ph[k]["wall_seconds_median"], ph[k]["wall_seconds_max"])
        self.assertTrue(10 < p["peak_rss_mib_process_high_water"] < 4096, "MiB, not bytes or kB")
