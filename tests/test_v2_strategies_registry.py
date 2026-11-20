import json
import os
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import engine, metrics, provenance, registry
from ledger_consistent_etf_trading.v2.config import ScenarioConfig, information_class
from ledger_consistent_etf_trading.v2.data import load_snapshot, market_from_frames
from ledger_consistent_etf_trading.v2.strategies import equal_weight_weekly, momentum_top4
from tests.helpers import cfg, synth, weights

REPO = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = pathlib.Path(os.environ.get("PV2_SNAPSHOT", REPO.parents[1] / "baseline/sprint1/public_snapshot"))
UNIVERSE = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLY", "XLI", "XLV", "XLP", "XLU", "TLT", "IEF", "GLD"]


def panel(closes):
    c = np.array(closes, float)
    idx = pd.date_range("2030-01-01", periods=len(c), freq="B", tz="UTC")
    f = lambda a: pd.DataFrame(a, index=idx, columns=[f"T{i}" for i in range(c.shape[1])])
    return market_from_frames(f(c), f(c), f(np.ones_like(c)))


class Momentum(unittest.TestCase):
    def rows(self, last, n=6, lookback=2):
        base = np.full((lookback + 3, n), 100.0)
        base[-2] = last                                    # session s = d-1 for d = last index
        return panel(base), len(base) - 1

    def test_top4_positive_equal_allocation_and_tie_break(self):
        m, d = self.rows([110, 110, 110, 110, 110, 90])
        w = momentum_top4(m, d - 1, d + 1, lookback=2)[d]
        self.assertTrue(np.allclose(w, [0.25, 0.25, 0.25, 0.25, 0, 0]), "ties resolve in ticker order")

    def test_fewer_than_four_positive(self):
        m, d = self.rows([105, 99, 120, 100, 80, 100])     # score exactly 0 is not eligible
        w = momentum_top4(m, d - 1, d + 1, lookback=2)[d]
        self.assertTrue(np.allclose(w, [0.5, 0, 0.5, 0, 0, 0]))

    def test_all_cash_when_none_positive(self):
        m, d = self.rows([100, 99, 98, 97, 96, 95])
        self.assertEqual(momentum_top4(m, d - 1, d + 1, lookback=2)[d].sum(), 0.0)

    def test_missing_price_makes_asset_ineligible(self):
        m, d = self.rows([np.nan, 130, 120, 110, 105, 104])
        w = momentum_top4(m, d - 1, d + 1, lookback=2)[d]
        self.assertEqual(w[0], 0.0); self.assertAlmostEqual(w.sum(), 1.0)

    def test_first_target_on_common_first_execution_session(self):
        m = synth()
        self.assertEqual(min(momentum_top4(m, 25, 60)), 26)
        self.assertEqual(min(equal_weight_weekly(m, 25, 60)), 26)


class WeeklyCalendar(unittest.TestCase):
    def test_first_session_of_week_signal(self):
        m = synth(T=40)
        t = equal_weight_weekly(m, 22, 40)
        for e in t:
            s = e - 1
            self.assertTrue(s == 22 or m.dates[s].isocalendar()[1] != m.dates[s - 1].isocalendar()[1])
            self.assertAlmostEqual(t[e].sum(), 1.0)

    @unittest.skipUnless((SNAPSHOT / "artifacts/weights_equal_weight_weekly.csv").exists(), "pinned snapshot not available")
    def test_matches_saved_legacy_signal_tape(self):
        mkt = load_snapshot(SNAPSHOT, UNIVERSE)
        start = int(mkt.dates.searchsorted(pd.Timestamp("2010-03-31", tz="UTC")))
        t = equal_weight_weekly(mkt, start, len(mkt.dates))
        saved = pd.read_csv(SNAPSHOT / "artifacts/weights_equal_weight_weekly.csv", parse_dates=["date"])
        sig = pd.DatetimeIndex(sorted(saved["date"].unique()))
        mine = mkt.dates[[e - 1 for e in sorted(t)]]
        self.assertTrue(mine.equals(sig[:len(mine)]))
        self.assertLessEqual(len(sig) - len(mine), 1, "only a final signal with no following session may be absent")
        self.assertEqual(len(mkt.dates) - start, 4077)


class Registry(unittest.TestCase):
    def test_ids_deterministic_and_sensitive(self):
        a, b = ScenarioConfig(name="x"), ScenarioConfig(name="x")
        self.assertEqual(a.scenario_id, b.scenario_id)
        self.assertNotEqual(a.scenario_id, a.with_(max_participation=0.06).scenario_id)
        self.assertEqual(ScenarioConfig.from_dict(json.loads(json.dumps(a.to_dict()))).scenario_id, a.scenario_id)

    def test_counts_and_primary_conventions(self):
        reg = registry.build()
        by = {}
        for c in reg:
            by.setdefault(c.experiment, []).append(c)
        self.assertEqual({k: len(v) for k, v in by.items()}, {"E1": 6, "E2": 24, "E2b": 16, "E3": 4, "E4": 2, "E5": 10})
        for c in reg:
            if c.experiment != "E1":
                self.assertEqual((c.valuation, c.features, c.adv_estimator), ("prior_close", "lagged", "mean_dollar"))
                self.assertIsNone(c.max_trade_notional, "absolute cap is excluded from primary experiments")
        valid = [c for c in by["E1"] if information_class(c.valuation, c.features)["valid"]]
        self.assertEqual(len(valid), 2)

    def test_e2_cells_are_matched(self):
        cells = [c for c in registry.build() if c.experiment == "E2"]
        groups = {}
        for c in cells:
            d = c.to_dict(); d.pop("name"); d["cost_enabled"] = d["cost"].pop("enabled"); cap = d.pop("cap_enabled"); cost = d.pop("cost_enabled")
            groups.setdefault(json.dumps(d, sort_keys=True), set()).add((cost, cap))
        self.assertEqual(len(groups), 6)
        for g in groups.values():
            self.assertEqual(g, {(False, False), (False, True), (True, False), (True, True)})

    def test_e2b_is_full_factorial(self):
        cells = [c for c in registry.build() if c.experiment == "E2b"]
        combos = {(c.cost.enabled, c.min_trade_notional, c.allow_borrowing, c.units) for c in cells}
        self.assertEqual(len(combos), 16)


class Provenance(unittest.TestCase):
    def test_immutable_run_directory(self):
        m = synth()
        res = engine.run(m, cfg(m), targets={26: weights(m, AAA=0.5)})
        with tempfile.TemporaryDirectory() as tmp:
            out = provenance.write_run(pathlib.Path(tmp), res, metrics.summarize(res, m.close), m, REPO, {"wall": 0.0}, {"record": "test"})
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(man["scenario_id"], res.cfg.scenario_id)
            self.assertEqual(len(man["code"]["commit"]), 40)
            for name, h in man["output_sha256"].items():
                self.assertEqual(provenance._sha(out / name), h)
            self.assertFalse(os.access(out / "daily.csv", os.W_OK))
            with self.assertRaises(FileExistsError):
                provenance.write_run(pathlib.Path(tmp), res, {}, m, REPO, {}, {})
            for p in out.iterdir():
                os.chmod(p, 0o644)


if __name__ == "__main__":
    unittest.main()
