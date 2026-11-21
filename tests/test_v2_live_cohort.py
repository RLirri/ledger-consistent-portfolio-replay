"""A small live cohort must survive the origination of a newer cohort."""
import unittest
import numpy as np
import pandas as pd
from ledger_consistent_etf_trading.v2 import engine, checks
from ledger_consistent_etf_trading.v2.config import ScenarioConfig, CostParams
from ledger_consistent_etf_trading.v2.data import Market, build_features


class LiveCohort(unittest.TestCase):
    def test_small_live_remainder_is_served_before_new_demand(self):
        dates = pd.date_range('2035-01-01', periods=30, freq='B', tz='UTC')
        p = np.full((30, 2), 100.)
        m = Market(dates, ['A', 'B'], p.copy(), p.copy(), np.full_like(p, 10000.), {})
        m.open[26] = 100.0000001
        cfg = ScenarioConfig(name='live', eval_start=str(dates[25].date()), initial_cash=1e8,
                             cap_enabled=False, cost=CostParams(enabled=False))
        targets = {26: np.array([.5, .5]), 27: np.array([.6, .4])}
        f = build_features(m)
        r = engine.run(m, cfg, targets=targets, feats=f)
        served = r.demand[(r.demand.event == 'executed') & (r.demand.session == 27)
                          & (r.demand.origin_session == 26) & (r.demand.ticker == 'A')]
        self.assertEqual(len(served), 1)
        self.assertGreater(served.units.iloc[0], 1e-12)
        self.assertLess(served.units.iloc[0], .001)
        report = checks.validate_result(r, m, f, production=False)
        self.assertTrue(report.ok, report.failures)
