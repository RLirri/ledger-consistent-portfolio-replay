"""Final targeted remediation (Sprint 3E): D01 crash-safe accounting, D02 validator lineage and marks,
D03 production preflight, D04 design receipt and legacy gate, C4 shared deployment session.
Synthetic only. Fault injection uses patches at the named crash boundaries."""
import json
import os
import pathlib
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2 import batch, checks, engine, metrics, registry
from ledger_consistent_etf_trading.v2.config import CostParams, ScenarioConfig
from ledger_consistent_etf_trading.v2.data import Features, Market, build_features
from ledger_consistent_etf_trading.v2.errors import ScenarioInputError
from ledger_consistent_etf_trading.v2.provenance import code_identity
from tests import batch_fixtures
from tests.helpers import synth
from tests.test_v2_batch import DESIGN, LOADER, MKT, REPO, START, auth, scen, unlock

WORKSPACE = REPO.parents[1]
PY = sys.executable


class Crash(BaseException):
    """Simulated process death: not an Exception, so nothing in the runner catches it."""


class Root(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="v2final-"))
        self.addCleanup(lambda: (unlock(self.root), shutil.rmtree(self.root, ignore_errors=True)))
        self.cfg = scen("g_a")

    def go(self, attempts=1, cfgs=None, **budget):
        cfgs = cfgs or [self.cfg]
        return batch.run_batch(cfgs, LOADER, self.root, REPO, auth(cfgs, attempts=attempts, **budget), DESIGN)

    def attempts(self):
        return [r for r in batch._ledger(self.root) if r["event"] == "attempt"]

    def published(self):
        return sorted(p.name for p in (self.root / "runs").iterdir())


class D01CrashSafeAccounting(Root):
    def test_structured_running_status_consumes_the_allowance(self):
        spec = batch.run_spec(self.cfg, code_identity(REPO), MKT.identity(), MKT.input_hashes, DESIGN, batch.environment())
        work = self.root / "work" / f"g_a-{spec['run_spec_id']}-a1"; work.mkdir(parents=True)
        (work / "status.json").write_text(json.dumps({"status": "running", "run_spec": spec, "attempt": 1}))
        out = self.go()
        self.assertEqual(out["scenarios"][0]["status"], "refused")
        rec = self.attempts()
        self.assertEqual([(r["status"], r["scenario_id"], r["attempt"]) for r in rec], [("interrupted", self.cfg.scenario_id, 1)])
        self.assertEqual(rec[0]["cpu_seconds_charged"], 30)
        self.assertEqual(self.published(), [])

    def crash_then_resume(self, where, expect_status, expect_published):
        """Kill the parent at a boundary, then resume twice with a one-attempt allowance."""
        real_replace, real_append, real_validate = os.replace, batch._append, checks.validate_saved

        def replace(src, dst):
            if where == "before_rename" and pathlib.Path(dst).parent.name == "runs":
                raise Crash()
            return real_replace(src, dst)

        def append(root, rec):
            if where == "before_settlement" and rec.get("event") == "attempt":
                raise Crash()
            return real_append(root, rec)

        def validate(*a, **k):
            if where == "after_child":
                raise Crash()
            return real_validate(*a, **k)

        with patch.object(batch.os, "replace", replace), patch.object(batch, "_append", append), patch.object(batch.checks, "validate_saved", validate):
            with self.assertRaises(Crash):
                self.go()
        reserved = [r for r in batch._ledger(self.root) if r["event"] == "reserved"]
        self.assertEqual(len(reserved), 1, "the reservation is written before the child starts")
        self.assertEqual(self.attempts(), [], "the crash left no settlement")
        first = self.go()
        rec = self.attempts()
        self.assertEqual(len(rec), 1)
        self.assertEqual((rec[0]["status"], rec[0]["scenario_id"], rec[0]["attempt"], rec[0]["recovered"]), (expect_status, self.cfg.scenario_id, 1, True))
        self.assertGreater(rec[0]["cpu_seconds_charged"], 0.0)
        self.assertEqual(len(self.published()), 1 if expect_published else 0)
        self.assertEqual(first["scenarios"][0]["status"], "skipped_verified" if expect_published else "refused")
        self.assertEqual(first["attempts_total"], 1)
        charged = first["cpu_seconds_charged_total"]
        second = self.go()
        self.assertEqual(len(self.attempts()), 1, "repeated recovery must not settle or charge the attempt again")
        self.assertEqual(second["attempts_total"], 1)
        self.assertLess(second["cpu_seconds_charged_total"] - charged, 5.0, "only the resume's own start-up overhead may be added")
        self.assertEqual(list((self.root / "work").iterdir()), [])
        return rec[0]

    def test_crash_after_child_before_validation(self):
        r = self.crash_then_resume("after_child", "interrupted", False)
        self.assertEqual(r["cpu_seconds_charged"], 30)
        self.assertTrue((self.root / r["dir"] / "daily.csv").exists(), "partial outputs are preserved as evidence")

    def test_crash_between_completion_marker_and_rename(self):
        r = self.crash_then_resume("before_rename", "interrupted", False)
        self.assertTrue((self.root / r["dir"] / "COMPLETE.json").exists())
        self.assertEqual(self.published(), [], "an unpublished marker is evidence, never a result")

    def test_crash_after_publication_before_settlement(self):
        r = self.crash_then_resume("before_settlement", "succeeded", True)
        done = json.loads((self.root / r["dir"] / "COMPLETE.json").read_text())
        self.assertEqual(r["cpu_seconds_charged"], done["cpu_seconds_charged"], "settled at the measured charge recorded in the marker")

    def test_crash_after_failed_attempt_is_archived_before_settlement(self):
        real = batch._append

        def append(root, rec):
            if rec.get("event") == "attempt":
                raise Crash()
            return real(root, rec)

        with patch.object(batch, "_append", append):
            with self.assertRaises(Crash):
                batch.run_batch([self.cfg], LOADER, self.root, REPO, auth([self.cfg]), DESIGN, faults={"g_a": "raise"})
        out = self.go()
        rec = self.attempts()
        self.assertEqual([(r["status"], r["recovered"]) for r in rec], [("failed", True)])
        self.assertEqual(out["scenarios"][0]["status"], "refused")

    def test_crash_before_the_status_file_exists(self):
        real = batch._atomic_json

        def atomic(path, obj):
            if path.name == "status.json":
                raise Crash()
            return real(path, obj)

        with patch.object(batch, "_atomic_json", atomic):
            with self.assertRaises(Crash):
                self.go()
        out = self.go()
        rec = self.attempts()
        self.assertEqual(len(rec), 1, "an unreadable work directory is matched to its reservation, not charged twice")
        self.assertEqual((rec[0]["status"], rec[0]["scenario_id"]), ("interrupted", self.cfg.scenario_id))
        self.assertEqual(out["scenarios"][0]["status"], "refused")

    def test_over_budget_parent_work_cannot_publish(self):
        with patch.object(batch.time, "process_time", side_effect=[0.0, 20.0]):
            out = self.go(cpu_seconds_total=15, cpu_seconds_per_scenario=10)
        self.assertEqual(out["scenarios"][0]["status"], "budget_exceeded")
        self.assertEqual(self.published(), [])
        rec = self.attempts()[0]
        self.assertGreater(rec["cpu_seconds_charged"], 20.0)
        self.assertTrue((self.root / rec["dir"] / "status.json").exists())
        self.assertFalse((self.root / rec["dir"] / "COMPLETE.json").exists())
        self.assertEqual(self.go(cpu_seconds_total=15, cpu_seconds_per_scenario=10)["scenarios"][0]["status"], "refused")

    def test_parent_bound_is_reserved_and_enforced(self):
        with patch.object(batch.time, "process_time", side_effect=[0.0, 4.0]):
            out = self.go(parent_cpu_seconds_per_scenario=2)
        self.assertEqual(out["scenarios"][0]["status"], "budget_exceeded")
        self.assertIn("parent work", out["scenarios"][0]["detail"])
        reserved = [r for r in batch._ledger(self.root) if r["event"] == "reserved"][0]
        self.assertEqual(reserved["reserved_cpu_seconds"], 32)

    def test_child_cpu_includes_interpreter_startup(self):
        self.go()
        rec = self.attempts()[0]
        self.assertGreater(rec["cpu_seconds"], rec["child_reported_cpu_seconds"], "kernel child accounting covers start-up the child cannot time itself")

    def test_budget_and_attempt_domains(self):
        code, mid = code_identity(REPO), MKT.identity()
        for key, values in (("cpu_seconds_total", (float("inf"), float("nan"), 0, -1, True, "600", None)),
                            ("cpu_seconds_per_scenario", (float("inf"), 2.5, 0, -3, True, "30")),
                            ("wall_seconds_per_scenario", (float("inf"), float("nan"), 0, -1)),
                            ("rss_mb", (float("inf"), 0, -5, None)),
                            ("parent_cpu_seconds_per_scenario", (float("inf"), 0, -1, True))):
            for v in values:
                a = auth([self.cfg]); a["budget"][key] = v
                with self.subTest(key=key, v=v), self.assertRaises(batch.BatchStop):
                    batch.check_authorization(a, [self.cfg], code, mid, DESIGN, self.root)
        for n in (0, -1, 1.0, 1.5, True, "1", None, float("inf")):
            a = auth([self.cfg]); a["authorized"][self.cfg.scenario_id] = n
            with self.subTest(attempts=n), self.assertRaises(batch.BatchStop):
                batch.check_authorization(a, [self.cfg], code, mid, DESIGN, self.root)
        batch.check_authorization(auth([self.cfg]), [self.cfg], code, mid, DESIGN, self.root)

    def test_environment_and_input_hashes_are_bound_when_authorized(self):
        code, mid = code_identity(REPO), MKT.identity()
        good = auth([self.cfg]) | {"environment": batch.environment(), "data_input_sha256": dict(MKT.input_hashes)}
        batch.check_authorization(good, [self.cfg], code, mid, DESIGN, self.root, MKT.input_hashes)
        for change in ({"environment": batch.environment() | {"numpy": "0.0"}}, {"environment": {}}, {"data_input_sha256": {"x.parquet": "0" * 64}}):
            with self.subTest(change=list(change)), self.assertRaises(batch.BatchStop):
                batch.check_authorization(good | change, [self.cfg], code, mid, DESIGN, self.root, MKT.input_hashes)


def good_run(**kw):
    kw.setdefault("experiment", "E2")
    cfg = ScenarioConfig(name="v", eval_start=START, strategy="momentum_top4", initial_cash=5e7, **kw)
    f = build_features(MKT, cfg.features, cfg.adv_estimator)
    return engine.run(MKT, cfg, feats=f, production=True), f


class D02ValidatorLineageAndMarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.f = good_run()
        cls.base = checks.validate_result(cls.res, MKT, cls.f)
        assert cls.base.ok, cls.base.failures

    def on_disk(self, res=None):
        res = res or self.res
        d = pathlib.Path(tempfile.mkdtemp(prefix="v2disk-")); self.addCleanup(shutil.rmtree, d, True)
        batch.write_outputs(d, res, metrics.summarize(res, MKT.close), checks.validate_result(res, MKT, self.f))
        (d / "manifest.json").write_text(json.dumps({"scenario_id": res.cfg.scenario_id, "config": res.cfg.to_dict(),
                                                     "run_info": json.loads(json.dumps(batch._run_info(d, res.cfg, MKT, self.f)))}))
        return d

    def rejected(self, expect, **frames):
        f = dict(daily=self.res.daily, orders=self.res.orders, parents=self.res.parents, demand=self.res.demand,
                 positions=self.res.positions, target_eff=self.res.target_eff)
        f.update(frames)
        rep = checks.validate(self.res.cfg, self.res.info, f["daily"], f["orders"], f["parents"], f["demand"], f["positions"], f["target_eff"], MKT, self.f)
        self.assertFalse(rep.ok, expect)
        self.assertIn(expect, [c["check"] for c in rep.failures])
        # the same corruption written to disk must be rejected on read-back
        res2 = type(self.res)(self.res.cfg, self.res.dates, self.res.tickers, f["daily"], f["positions"], f["target_eff"], f["orders"], f["parents"], self.res.info, f["demand"])
        d = pathlib.Path(tempfile.mkdtemp(prefix="v2disk-")); self.addCleanup(shutil.rmtree, d, True)
        try:
            summary = metrics.summarize(res2, MKT.close)
        except Exception:
            summary = {"scenario_id": self.res.cfg.scenario_id}
        batch.write_outputs(d, res2, json.loads(json.dumps(summary, default=str)), rep)
        (d / "manifest.json").write_text(json.dumps({"scenario_id": self.res.cfg.scenario_id, "config": self.res.cfg.to_dict(),
                                                     "run_info": json.loads(json.dumps(batch._run_info(d, self.res.cfg, MKT, self.f)))}))
        self.assertIn(expect, [c["check"] for c in checks.validate_saved(d, MKT, self.f).failures], "read-back")

    def edit(self, frame, row, col, fn):
        df = getattr(self.res, frame).copy(); df.loc[row, col] = fn(df.loc[row, col]); return {frame: df}

    def test_clean_run_passes_in_memory_and_on_disk_with_more_checks(self):
        self.assertGreaterEqual(len(self.base.checks), 65)
        rb = checks.validate_saved(self.on_disk(), MKT, self.f)
        self.assertTrue(rb.ok, rb.failures)
        for val, feat in (("prior_close", "lagged"), ("open", "lagged"), ("same_close", "same_day"), ("prior_close", "same_day")):
            res, f = good_run(valuation=val, features=feat, experiment="E1")
            self.assertTrue(checks.validate_result(res, MKT, f).ok, (val, feat))
        for kw in (dict(residual_policy="cancel", experiment="E3"), dict(residual_policy="legacy_additive", experiment="E3"), dict(units="integer", experiment="E2b"),
                   dict(cash_rule="legacy_sequential", experiment="E4"), dict(cap_enabled=False), dict(strategy="equal_weight_weekly")):
            kw2 = dict(kw); strat = kw2.pop("strategy", "momentum_top4")
            cfg = ScenarioConfig(name="v", eval_start=START, strategy=strat, initial_cash=5e7, **({"experiment": "E2"} | kw2))
            rep = checks.validate_result(engine.run(MKT, cfg, production=True), MKT, build_features(MKT))
            self.assertTrue(rep.ok, (kw, rep.failures))

    def test_positive_but_wrong_marks_and_reference_prices(self):
        p = self.res.parents
        inc = int(p.index[p["unfilled"].abs() > 0][0])
        self.rejected("parent horizon mark equals the price implied by its close reason and session", **self.edit("parents", inc, "p_H", lambda v: v * 2))
        self.rejected("parent horizon mark equals the price implied by its close reason and session", **self.edit("parents", inc, "p_H", lambda v: v * 1.0001))
        self.rejected("parent decision price equals the sizing price of its target event", **self.edit("parents", inc, "p_D", lambda v: v * 1.01))
        self.rejected("parent arrival price equals the open of its target session", **self.edit("parents", inc, "p_A", lambda v: v * 1.01))
        self.rejected("parent sessions and assets lie in the run", **self.edit("parents", inc, "close_session", lambda v: 10 ** 6))
        rep_row = int(p.index[p["close_reason"] == "replaced"][0])
        self.rejected("parent horizon mark equals the price implied by its close reason and session", **self.edit("parents", rep_row, "close_reason", lambda v: "cancelled_eod"))

    def test_cohort_units_values_origins_and_order(self):
        d = self.res.demand
        ex = int(d.index[d["event"] == "executed"][3]); og = int(d.index[d["event"] == "originated"][3])
        self.rejected("demand value equals units x basis price", **self.edit("demand", 0, "units", lambda v: v * 2))
        self.rejected("demand value equals units x basis price", **self.edit("demand", ex, "value", lambda v: v * 1.5))
        both = d.copy(); both.loc[ex, ["units", "value"]] *= 2            # units and value doubled consistently
        self.rejected("executed plus overshoot demand equals the fills of each asset-session", demand=both)
        basis = d.copy(); basis.loc[og, "basis_price"] *= 1.1; basis.loc[og, "value"] *= 1.1
        self.rejected("demand basis and origin equal the sizing price of the originating event", demand=basis)
        self.rejected("demand basis and origin equal the sizing price of the originating event", **self.edit("demand", ex, "origin_session", lambda v: v - 1))
        # an executed row relabelled as a withdrawal no longer matches the fills; a row pointing at another cohort breaks the service order
        self.rejected("executed plus overshoot demand equals the fills of each asset-session", **self.edit("demand", ex, "event", lambda v: "withdrawn"))
        self.rejected("demand served oldest-first, withdrawn newest-first, within cohort size", **self.edit("demand", ex, "origin_session", lambda v: v - 1))
        self.rejected("every cohort conserved: originated = executed + withdrawn + outstanding (units)", demand=d.drop(index=int(d.index[d["event"] == "outstanding_end"][0])))
        self.rejected("demand gap of every target event equals target units minus prior holdings", **self.edit("parents", 5, "demand_gap", lambda v: v * 1.2 + 1))
        self.rejected("demand ledger sessions, origins and assets lie in the run", **self.edit("demand", ex, "session", lambda v: 10 ** 6))

    def test_targets_and_strategy(self):
        t = self.res.target_eff.copy(); t[30:, [0, 1]] = t[30:, [1, 0]]                 # still valid weights, wrong assets
        self.rejected("target tape equals the strategy's targets recomputed from market data", target_eff=t)
        info = dict(self.res.info, targets_source="supplied")
        rep = checks.validate(self.res.cfg, info, self.res.daily, self.res.orders, self.res.parents, self.res.demand, self.res.positions, self.res.target_eff, MKT, self.f)
        self.assertIn("targets were generated by the registered strategy", [c["check"] for c in rep.failures])

    def test_feature_recomputation(self):
        bad = build_features(MKT); bad.adv = bad.adv * 1.0001
        rep = checks.validate_result(self.res, MKT, bad)
        self.assertIn("features equal an independent recomputation with the protocol window", [c["check"] for c in rep.failures])

    def test_saved_labels_and_every_saved_metric(self):
        cases = []
        d = self.on_disk(); pos = pd.read_csv(d / "positions.csv", index_col=0)
        pos.columns = list(pos.columns[::-1]); pos.to_csv(d / "positions.csv"); cases.append((d, "saved position and target tapes carry the run's asset and session labels"))
        d = self.on_disk(); tw = pd.read_csv(d / "target_weights.csv", index_col=0)
        tw.index = pd.to_datetime(tw.index) + pd.Timedelta(days=1); tw.to_csv(d / "target_weights.csv"); cases.append((d, "saved position and target tapes carry the run's asset and session labels"))
        for path in (("sharpe_rf0",), ("turnover_gross_mean_daily_excl_deployment_session",), ("orders", "completion_aggregate"), ("orders", "is_dollars"),
                     ("wte_risky_mean",), ("full_period", "nav_max"), ("demand", "withdrawn_value"), ("cagr",), ("deployment_session",)):
            d = self.on_disk(); m = json.loads((d / "metrics.json").read_text())
            node = m
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = (node[path[-1]] or 0) + 1e-3 * max(1.0, abs(node[path[-1]] or 0))
            (d / "metrics.json").write_text(json.dumps(m)); cases.append((d, "every saved metric equals its recomputation from the saved tapes"))
        d = self.on_disk(); m = json.loads((d / "metrics.json").read_text()); del m["max_drawdown"]; (d / "metrics.json").write_text(json.dumps(m))
        cases.append((d, "every saved metric equals its recomputation from the saved tapes"))
        for d, expect in cases:
            rep = checks.validate_saved(d, MKT, self.f)
            self.assertIn(expect, [c["check"] for c in rep.failures])


class D03ProductionPreflight(unittest.TestCase):
    def cfg(self, **kw):
        return ScenarioConfig(name="p", eval_start=START, strategy="momentum_top4", initial_cash=5e7, **({"experiment": "E2"} | kw))

    def refused_before_replay(self, mkt, cfg, **kw):
        with patch.object(engine, "build_targets", side_effect=AssertionError("replay entered")) as spy:
            with self.assertRaises(ScenarioInputError):
                engine.run(mkt, cfg, **kw)
        self.assertEqual(spy.call_count, 0)

    def test_relabelled_or_wrong_window_features(self):
        f = build_features(MKT, "same_day"); f.mode = "lagged"
        self.refused_before_replay(MKT, self.cfg(), feats=f)
        self.refused_before_replay(MKT, self.cfg(), feats=build_features(MKT, window=5))
        f = build_features(MKT, window=5); f.window = 20                      # relabelled window
        self.refused_before_replay(MKT, self.cfg(), feats=f)
        f = build_features(MKT); f.vol = f.vol.copy(); f.vol[40, 0] *= 1.0000001   # a single edited value
        self.refused_before_replay(MKT, self.cfg(), feats=f)
        f = build_features(MKT, "lagged", "shares_x_close"); f.estimator = "mean_dollar"
        self.refused_before_replay(MKT, self.cfg(), feats=f)

    def test_unbound_features_and_supplied_targets_are_a_synthetic_route_only(self):
        f = build_features(MKT); f.market_id = None
        self.refused_before_replay(MKT, self.cfg(), feats=f, production=True)
        self.assertEqual(engine.run(MKT, self.cfg(), feats=f).info["features_binding"], "unbound")
        w = np.zeros(len(MKT.tickers)); w[0] = 0.5
        with self.assertRaises(ScenarioInputError):
            engine.run(MKT, self.cfg(), targets={31: w}, production=True)
        self.assertEqual(engine.run(MKT, self.cfg(), targets={31: w}).info["targets_source"], "supplied")

    def test_raw_domain(self):
        for field, row, bad in (("volume", 10, -1.0), ("volume", 3, float("inf")), ("close", 5, 0.0), ("close", 12, -4.0), ("open", 60, float("inf")), ("open", 8, -1.0)):
            m = MKT.copy(); getattr(m, field)[row, 0] = bad
            with self.subTest(field=field, row=row, bad=bad):
                self.refused_before_replay(m, self.cfg())
        m = MKT.copy(); m.volume[10, 0] = 0.0                                   # zero volume is a valid observation
        engine.run(m, self.cfg(), production=True)
        for field in ("open", "close", "volume"):                              # a missing value is tolerated only outside production
            m = MKT.copy(); getattr(m, field)[2, 4] = np.nan
            self.refused_before_replay(m, self.cfg(), production=True)

    def test_valid_and_declared_control_paths_keep_their_classification(self):
        for val, feat, role in (("prior_close", "lagged", "primary"), ("open", "lagged", "sensitivity"), ("prior_close", "same_day", "negative_control"),
                                ("open", "same_day", "negative_control"), ("same_close", "lagged", "negative_control"), ("same_close", "same_day", "negative_control")):
            res = engine.run(MKT, self.cfg(valuation=val, features=feat, experiment="E1"), production=True)
            self.assertEqual(res.info["information"]["role"], role)
            self.assertEqual((res.info["features_binding"], res.info["targets_source"], res.info["production"]), ("bound", "strategy", True))


class D04DesignReceiptAndLegacyGate(unittest.TestCase):
    def workspace(self):
        w = pathlib.Path(tempfile.mkdtemp(prefix="v2ws-")); self.addCleanup(shutil.rmtree, w, True)
        (w / "PROTOCOL.md").write_text("protocol text\n")
        (w / "registry.json").write_text(json.dumps({"scenarios": registry.as_records()}))
        comps = {p: batch._sha(w / p) for p in ("PROTOCOL.md", "registry.json")}
        commit = code_identity(REPO)["commit"]
        import hashlib
        fp = hashlib.sha256(json.dumps({**comps, "code": commit}, sort_keys=True).encode()).hexdigest()
        (w / "receipt.json").write_text(json.dumps({"design_fingerprint": fp, "components_sha256": comps, "v2_checkout_commit": commit}))
        return w, fp

    def rewrite(self, w, **change):
        r = json.loads((w / "receipt.json").read_text()); r.update(change); (w / "receipt.json").write_text(json.dumps(r))

    def test_receipt_recomputed_from_files(self):
        w, fp = self.workspace()
        self.assertEqual(batch.verify_design_receipt(w / "receipt.json", w, REPO), fp)
        for name, breaker in (("component edited", lambda: (w / "PROTOCOL.md").write_text("changed\n")),
                              ("component missing", lambda: (w / "PROTOCOL.md").unlink()),
                              ("wrong commit", lambda: self.rewrite(w, v2_checkout_commit="0" * 40)),
                              ("wrong fingerprint", lambda: self.rewrite(w, design_fingerprint="f" * 64)),
                              ("no components", lambda: self.rewrite(w, components_sha256={})),
                              ("unreadable", lambda: (w / "receipt.json").write_text("{not json")),
                              ("registry altered", lambda: (w / "registry.json").write_text(json.dumps({"scenarios": registry.as_records()[:-1]})))):
            w, fp = self.workspace(); breaker()
            with self.subTest(name=name), self.assertRaises(batch.BatchStop):
                batch.verify_design_receipt(w / "receipt.json", w, REPO)
        w, fp = self.workspace()                                             # a receipt with no registry binds no scenario set
        import hashlib
        comps = {"PROTOCOL.md": batch._sha(w / "PROTOCOL.md")}; commit = code_identity(REPO)["commit"]
        self.rewrite(w, components_sha256=comps, design_fingerprint=hashlib.sha256(json.dumps({**comps, "code": commit}, sort_keys=True).encode()).hexdigest())
        with self.assertRaises(batch.BatchStop):
            batch.verify_design_receipt(w / "receipt.json", w, REPO)

    def dispatch(self, w, fp, auth_change=None, receipt="receipt.json"):
        cfg = registry.pilot(); called = []
        a = {"design_fingerprint": fp, "code_commit": code_identity(REPO)["commit"], "market_id": "m", "data_input_sha256": {}, "environment": batch.environment(),
             "output_root": str(w / "out"), "authorized": {cfg.scenario_id: 1},
             "budget": {"cpu_seconds_total": 60, "cpu_seconds_per_scenario": 10, "wall_seconds_per_scenario": 30, "rss_mb": 500, "parent_cpu_seconds_per_scenario": 5}}
        a.update(auth_change or {})
        for k in [k for k, v in a.items() if v is None]:
            del a[k]
        (w / "author.json").write_text(json.dumps({"execution_authorization": a}))
        argv = ["v2_batch.py", "--scenario", "pilot_252", "--snapshot", "/never-open", "--out", str(w / "out"), "--authorization", str(w / "author.json"),
                "--design-fingerprint", str(w / receipt), "--workspace", str(w)]
        with patch.object(sys, "argv", argv), patch.object(batch, "run_batch", lambda *x, **k: called.append(x) or {"stopped": None}):
            try:
                runpy.run_path(str(REPO / "scripts/v2_batch.py"), run_name="__main__")
            except SystemExit:
                pass
        return bool(called)

    def test_cli_dispatches_only_a_verified_receipt_and_complete_authorization(self):
        w, fp = self.workspace()
        self.assertTrue(self.dispatch(w, fp), "an approved synthetic receipt must pass")
        w, fp = self.workspace(); (w / "PROTOCOL.md").write_text("tampered\n")
        self.assertFalse(self.dispatch(w, fp))
        w, fp = self.workspace(); self.rewrite(w, v2_checkout_commit="incorrect")
        self.assertFalse(self.dispatch(w, fp))
        w, fp = self.workspace()
        self.assertFalse(self.dispatch(w, fp, {"design_fingerprint": "another"}))
        for missing in ("environment", "data_input_sha256", "output_root", "market_id"):
            w, fp = self.workspace()
            self.assertFalse(self.dispatch(w, fp, {missing: None}), missing)
        w, fp = self.workspace()
        self.assertFalse(self.dispatch(w, fp, {"budget": {"cpu_seconds_total": 60, "cpu_seconds_per_scenario": 10, "wall_seconds_per_scenario": 30, "rss_mb": 500}}),
                         "the parent-side bound is mandatory on the historical entry point")

    def test_current_author_record_cannot_enter_either_historical_path(self):
        record = WORKSPACE / "revisions/author_decisions.json"
        if not record.exists():
            self.skipTest("workspace author record not available")
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
        out = pathlib.Path(tempfile.mkdtemp(prefix="v2legacy-")); self.addCleanup(shutil.rmtree, out, True)
        target = out / "never"
        a = subprocess.run([PY, "-B", str(REPO / "scripts/v2_legacy_reproduction.py"), "/nonexistent-snapshot", str(target), str(record)], env=env, capture_output=True, text=True)
        self.assertNotEqual(a.returncode, 0); self.assertIn("refusing", a.stdout + a.stderr); self.assertFalse(target.exists())
        b = subprocess.run([PY, "-B", str(REPO / "scripts/v2_batch.py"), "--scenario", "pilot_252", "--snapshot", "/nonexistent-snapshot", "--out", str(target),
                            "--authorization", str(record), "--design-fingerprint", str(WORKSPACE / "validation/sprint3/design_fingerprint.json")], env=env, capture_output=True, text=True)
        self.assertNotEqual(b.returncode, 0); self.assertIn("refus", b.stdout + b.stderr); self.assertFalse(target.exists())

    def test_legacy_gate_conditions(self):
        out = pathlib.Path(tempfile.mkdtemp(prefix="v2legacy-")); self.addCleanup(shutil.rmtree, out, True)
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
        head = code_identity(REPO)["commit"]

        def run(block, target):
            (out / "auth.json").write_text(json.dumps({"authorized_runs": ["legacy_reproduction"], "execution_authorization": {"legacy_reproduction": block} if block is not None else {}}))
            return subprocess.run([PY, "-B", str(REPO / "scripts/v2_legacy_reproduction.py"), "/nonexistent-snapshot", str(target), str(out / "auth.json")], env=env, capture_output=True, text=True)

        t = out / "run1"
        for block in (None, {"authorized": False, "code_commit": head, "output_dir": str(t)}, {"authorized": True, "code_commit": "0" * 40, "output_dir": str(t)},
                      {"authorized": True, "code_commit": head, "output_dir": str(out / "elsewhere")}):
            r = run(block, t)
            self.assertNotEqual(r.returncode, 0); self.assertIn("refusing", r.stdout + r.stderr); self.assertFalse(t.exists())
        used = out / "used"; used.mkdir()
        r = run({"authorized": True, "code_commit": head, "output_dir": str(used)}, used)
        self.assertIn("already exists", r.stdout + r.stderr)
        r = run({"authorized": True, "code_commit": head, "output_dir": str(t)}, t)      # passes the gate, then fails on the absent synthetic snapshot
        self.assertNotIn("refusing", r.stdout + r.stderr); self.assertNotEqual(r.returncode, 0)


class C4SharedDeploymentSession(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2035-01-01", periods=30, freq="B", tz="UTC"); p = np.full((30, 1), 100.0)
        self.m = Market(idx, ["SYNTH"], p.copy(), p.copy(), p.copy() * 100, {})
        self.c = ScenarioConfig(name="synthetic", eval_start=str(idx[25].date()), initial_cash=20000.0, cost=CostParams(enabled=False), min_trade_notional=1000.0)
        self.t = {26: np.array([0.5])}
        self.f = Features(np.full_like(p, 0.01), np.full_like(p, 100000.0), "lagged", "mean_dollar"); self.f.adv[26] = 10000.0

    def arms(self):
        a = engine.run(self.m, self.c.with_(cap_enabled=False), self.t, feats=self.f)       # fills on the scheduled session
        b = engine.run(self.m, self.c, self.t, feats=self.f)                               # first fill one session later
        return a, b

    def test_delayed_first_fill_uses_the_scheduled_session(self):
        a, b = self.arms()
        first = lambda r: int(np.flatnonzero(r.daily["gross_traded"].to_numpy() > 0)[0]) + r.info["start"]
        self.assertEqual((first(a), first(b)), (26, 27), "the arms first execute on different dates")
        ta, tb = metrics.summarize(a, self.m.close), metrics.summarize(b, self.m.close)
        self.assertEqual(metrics.require_common_deployment(a, b), 26)
        for t in (ta, tb):
            self.assertEqual((t["deployment_session"], t["deployment_session_date"]), (26, str(self.m.dates[26].date())))
            self.assertEqual((t["turnover_observations"], t["turnover_observations_excl_deployment_session"]), (5, 4))
            self.assertAlmostEqual(t["turnover_gross_annualized"], 252 * t["turnover_gross_mean_daily"], places=12)
        self.assertAlmostEqual(tb["turnover_gross_mean_daily_excl_deployment_session"], 0.125, places=12)
        self.assertAlmostEqual(tb["turnover_gross_annualized_excl_deployment_session"], 31.5, places=10)
        self.assertEqual(ta["turnover_gross_mean_daily_excl_deployment_session"], 0.0)
        self.assertAlmostEqual(ta["turnover_gross_mean_daily"], 0.5 / 5, places=12)
        self.assertNotIn("turnover_gross_mean_daily_excl_first_execution", tb)

    def test_arm_with_no_fills_uses_the_same_mask(self):
        f = Features(self.f.vol, np.full_like(self.f.adv, 1.0), "lagged", "mean_dollar")       # limit far below the minimum trade: never fills
        z = engine.run(self.m, self.c, self.t, feats=f)
        a, _ = self.arms()
        self.assertEqual(z.daily["gross_traded"].sum(), 0.0)
        t = metrics.summarize(z, self.m.close)
        self.assertEqual((t["deployment_session"], t["turnover_observations_excl_deployment_session"], t["turnover_gross_mean_daily_excl_deployment_session"]), (26, 4, 0.0))
        self.assertEqual(metrics.require_common_deployment(a, z), 26)

    def test_subperiod_boundaries(self):
        _, b = self.arms()
        d = [str(x.date()) for x in self.m.dates]
        inside = metrics.period_summary(b, self.m.close, d[25], d[27])
        self.assertEqual((inside["turnover_observations"], inside["turnover_observations_excl_deployment_session"], inside["deployment_session_in_period"]), (3, 2, True))
        self.assertAlmostEqual(inside["turnover_gross_mean_daily_excl_deployment_session"], 0.25 / 2, places=12)
        after = metrics.period_summary(b, self.m.close, d[27], d[29])
        self.assertEqual((after["turnover_observations"], after["turnover_observations_excl_deployment_session"], after["deployment_session_in_period"]), (3, 3, False))
        self.assertEqual(after["turnover_gross_mean_daily"], after["turnover_gross_mean_daily_excl_deployment_session"])
        only = metrics.period_summary(b, self.m.close, d[26], d[26])
        self.assertEqual((only["turnover_observations_excl_deployment_session"], only["turnover_gross_mean_daily_excl_deployment_session"]), (0, None))

    def test_mismatched_calendars_or_schedules_are_refused(self):
        a, _ = self.arms()
        later = engine.run(self.m, self.c.with_(cap_enabled=False), {27: np.array([0.5])}, feats=self.f)
        with self.assertRaises(ScenarioInputError):
            metrics.require_common_deployment(a, later)
        shorter = engine.run(self.m, self.c.with_(cap_enabled=False, max_sessions=4), self.t, feats=self.f)
        with self.assertRaises(ScenarioInputError):
            metrics.require_common_deployment(a, shorter)

    def test_registered_strategies_share_the_scheduled_deployment_session(self):
        for strat in ("equal_weight_weekly", "momentum_top4"):
            runs = [engine.run(MKT, ScenarioConfig(name="x", eval_start=START, strategy=strat, initial_cash=cap, cap_enabled=lim), production=True)
                    for cap in (1e7, 1e9) for lim in (False, True)]
            self.assertEqual({metrics.deployment_session(r) for r in runs}, {runs[0].info["start"] + 1})


if __name__ == "__main__":
    unittest.main()
