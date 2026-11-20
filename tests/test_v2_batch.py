"""Batch execution safeguards on synthetic data with simulated failures. No historical data is touched."""
import json
import os
import pathlib
import shutil
import stat
import tempfile
import unittest

from ledger_consistent_etf_trading.v2 import batch
from ledger_consistent_etf_trading.v2.config import CostParams, ScenarioConfig
from ledger_consistent_etf_trading.v2.provenance import code_identity
from tests import batch_fixtures

REPO = pathlib.Path(__file__).resolve().parents[1]
LOADER = "tests.batch_fixtures:load"
DESIGN = "synthetic-design-fingerprint"
MKT = batch_fixtures.load()
START = str(MKT.dates[30].date())


def scen(name, **kw):
    kw.setdefault("strategy", "momentum_top4")
    kw.setdefault("initial_cash", 5e6)
    return ScenarioConfig(name=name, eval_start=START, experiment="E2", **kw)


def auth(cfgs, attempts=1, **budget):
    b = dict(cpu_seconds_total=600, cpu_seconds_per_scenario=30, wall_seconds_per_scenario=60, rss_mb=1500)
    b.update(budget)
    return dict(design_fingerprint=DESIGN, code_commit=code_identity(REPO)["commit"], allow_dirty=True, market_id=MKT.identity(),
                authorized={c.scenario_id: attempts for c in cfgs}, budget=b, receipt="synthetic test authorization")


def unlock(root):
    for dp, dn, fn in os.walk(root):
        os.chmod(dp, 0o755)
        for f in fn:
            os.chmod(os.path.join(dp, f), 0o644)


class BatchCase(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="v2batch-"))
        self.addCleanup(lambda: (unlock(self.root), shutil.rmtree(self.root, ignore_errors=True)))
        self.cfgs = [scen("s_a"), scen("s_b", strategy="equal_weight_weekly"), scen("s_c", cap_enabled=False)]

    def go(self, cfgs=None, authorization=None, root=None, **kw):
        cfgs = self.cfgs if cfgs is None else cfgs
        return batch.run_batch(cfgs, kw.pop("loader", LOADER), root or self.root, REPO, authorization or auth(cfgs), kw.pop("design", DESIGN), **kw)

    def ledger(self, root=None):
        return [r for r in batch._ledger(root or self.root) if r["event"] == "attempt"]


class CompletionAndResume(BatchCase):
    def test_clean_batch_publishes_validated_immutable_results_and_resumes(self):
        s = self.go()
        self.assertIsNone(s["stopped"])
        self.assertEqual([x["status"] for x in s["scenarios"]], ["succeeded"] * 3)
        self.assertEqual([x["name"] for x in s["scenarios"]], ["s_a", "s_b", "s_c"], "deterministic order")
        for x in s["scenarios"]:
            d = self.root / x["dir"]
            done = json.loads((d / "COMPLETE.json").read_text())
            self.assertEqual(done["status"], "succeeded")
            for name, h in done["output_sha256"].items():
                self.assertEqual(batch._sha(d / name), h)
            v = json.loads((d / "validation.json").read_text()); rb = json.loads((d / "readback_validation.json").read_text())
            self.assertTrue(v["valid"] and rb["valid"]); self.assertGreater(v["n_checks"], 35)
            man = json.loads((d / "manifest.json").read_text())
            for k in ("code_commit", "market_id", "design_fingerprint", "environment", "run_spec_id", "data_input_sha256"):
                self.assertIn(k, man["run_spec"])
            self.assertIn("phases_cpu_seconds", man["resources"]); self.assertGreater(man["resources"]["peak_rss_mib"], 10)
            self.assertFalse(os.access(d / "daily.csv", os.W_OK)); self.assertFalse(os.access(d, os.W_OK))
            self.assertNotIn("NaN", (d / "metrics.json").read_text())
        self.assertEqual(list((self.root / "work").iterdir()), [])
        self.assertEqual(len(self.ledger()), 3)
        again = self.go()
        self.assertEqual([x["status"] for x in again["scenarios"]], ["skipped_verified"] * 3)
        self.assertEqual(len(self.ledger()), 3, "resume must not create attempts or spend budget")

    def test_outputs_are_deterministic_across_roots(self):
        other = pathlib.Path(tempfile.mkdtemp(prefix="v2batch-"))
        self.addCleanup(lambda: (unlock(other), shutil.rmtree(other, ignore_errors=True)))
        a = self.go(cfgs=self.cfgs[:1]); b = self.go(cfgs=self.cfgs[:1], root=other)
        da, db = self.root / a["scenarios"][0]["dir"], other / b["scenarios"][0]["dir"]
        for name in ("daily.csv", "orders.csv", "parents.csv", "demand.csv", "positions.csv", "metrics.json"):
            self.assertEqual(batch._sha(da / name), batch._sha(db / name), name)
        self.assertEqual(a["scenarios"][0]["run_spec_id"], b["scenarios"][0]["run_spec_id"])

    def test_tampered_result_stops_the_batch_and_is_not_overwritten(self):
        s = self.go()
        d = self.root / s["scenarios"][1]["dir"]
        os.chmod(d, 0o755); os.chmod(d / "daily.csv", 0o644)
        (d / "daily.csv").write_text((d / "daily.csv").read_text().replace("1", "2", 1))
        tampered = batch._sha(d / "daily.csv")
        again = self.go()
        self.assertEqual([x["status"] for x in again["scenarios"]], ["skipped_verified", "stale_or_tampered"])
        self.assertIn("failed verification", again["stopped"])
        self.assertEqual(batch._sha(d / "daily.csv"), tampered)
        self.assertEqual(len(self.ledger()), 3)

    def test_changed_design_or_data_is_never_reused(self):
        self.go(cfgs=self.cfgs[:1])
        a2 = auth(self.cfgs[:1], attempts=2); a2["design_fingerprint"] = "revised-design"   # a second attempt must be authorized explicitly
        s = self.go(cfgs=self.cfgs[:1], authorization=a2, design="revised-design")
        e = s["scenarios"][0]
        self.assertEqual(e["status"], "succeeded")
        self.assertEqual(len(e["incompatible_prior_results_not_reused"]), 1)
        self.assertEqual(len(list((self.root / "runs").iterdir())), 2, "old result preserved beside the new one")


class Refusals(BatchCase):
    def refused(self, **kw):
        with self.assertRaises(batch.BatchStop):
            self.go(**kw)
        self.assertEqual(self.ledger(), [])
        self.assertEqual(list((self.root / "runs").iterdir()), [])

    def test_missing_or_incomplete_authorization(self):
        self.refused(authorization={"authorized_runs": ["s_a"]})         # a bare name allowlist is not enough
        a = auth(self.cfgs); del a["budget"]["rss_mb"]
        self.refused(authorization=a)

    def test_unauthorized_scenario_or_changed_configuration_under_an_authorized_name(self):
        self.refused(authorization=auth(self.cfgs[:2]))
        changed = [self.cfgs[0].with_(max_participation=0.5)] + self.cfgs[1:]
        self.assertEqual(changed[0].name, self.cfgs[0].name)
        self.refused(cfgs=changed, authorization=auth(self.cfgs))

    def test_changed_design_code_or_data(self):
        self.refused(design="another-design")
        a = auth(self.cfgs); a["code_commit"] = "0" * 40
        self.refused(authorization=a)
        a = auth(self.cfgs); a["allow_dirty"] = False
        if code_identity(REPO)["dirty"]:
            self.refused(authorization=a)
        self.refused(loader="tests.batch_fixtures:load_other")

    def test_output_root_binding(self):
        a = auth(self.cfgs); a["output_root"] = str(self.root / "elsewhere")
        self.refused(authorization=a)
        a["output_root"] = str(self.root)
        self.assertIsNone(self.go(cfgs=self.cfgs[:1], authorization=auth(self.cfgs[:1]) | {"output_root": str(self.root)})["stopped"])

    def test_parent_side_cpu_is_charged(self):
        self.go(cfgs=self.cfgs[:1])
        rec = self.ledger()[0]
        self.assertGreater(rec["parent_cpu_seconds"], 0.0)
        self.assertAlmostEqual(rec["cpu_seconds_charged"], rec["cpu_seconds"] + rec["parent_cpu_seconds"], places=9)

    def test_attempts_exhausted(self):
        s = self.go(faults={"s_a": "raise"})
        self.assertEqual(s["scenarios"][0]["status"], "failed")
        again = self.go()
        self.assertEqual(again["scenarios"][0]["status"], "refused"); self.assertIn("attempts", again["stopped"])
        self.assertEqual(len(self.ledger()), 1)

    def test_cumulative_cpu_budget(self):
        # the per-scenario limit is reserved before each attempt: 0 + 30 <= 30.01, then (actual CPU of the first) + 30 > 30.01
        a = auth(self.cfgs, cpu_seconds_total=30.01, cpu_seconds_per_scenario=30)
        s = self.go(authorization=a)
        self.assertEqual([x["status"] for x in s["scenarios"]], ["succeeded", "refused"])
        self.assertIn("budget", s["stopped"])
        self.assertLessEqual(s["cpu_seconds_charged_total"], 30.01)


class FailuresAndLimits(BatchCase):
    def one(self, fault, expect, **budget):
        a = auth(self.cfgs, attempts=2, **budget)
        s = self.go(authorization=a, faults={"s_b": fault})
        self.assertEqual([x["status"] for x in s["scenarios"]], ["succeeded", expect], s["scenarios"][-1])
        self.assertIn("s_b", s["stopped"])
        self.assertFalse(any(p.name.startswith("s_b") for p in (self.root / "runs").iterdir()), "no success published")
        self.assertFalse(any(p.name.startswith("s_c") for p in (self.root / "runs").iterdir()), "batch stopped")
        kept = self.root / s["scenarios"][1]["dir"]
        self.assertEqual(json.loads((kept / "status.json").read_text())["status"], expect)
        self.assertFalse((kept / "COMPLETE.json").exists())
        self.assertEqual(list((self.root / "work").iterdir()), [])
        return s, kept, a

    def test_exception_is_recorded_then_a_clean_retry_resumes(self):
        s, kept, a = self.one("raise", "failed")
        self.assertIn("injected failure", json.loads((kept / "status.json").read_text())["error"])
        again = self.go(authorization=a)
        self.assertEqual([x["status"] for x in again["scenarios"]], ["skipped_verified", "succeeded", "succeeded"])
        self.assertEqual(again["scenarios"][1]["attempt"], 2)
        self.assertTrue(kept.exists(), "failed attempt preserved")
        self.assertEqual([r["status"] for r in self.ledger()], ["succeeded", "failed", "succeeded", "succeeded"])

    def test_failed_invariant_blocks_publication(self):
        s, kept, _ = self.one("corrupt_cash", "invalid")
        v = json.loads((kept / "validation.json").read_text())
        self.assertFalse(v["valid"])
        self.assertIn("cash conservation (every session)", [c["check"] for c in v["checks"] if not c["passed"]])
        self.assertIn("withheld", json.loads((kept / "metrics.json").read_text())["status"])

    def test_nonfinite_output_blocks_publication(self):
        s, kept, _ = self.one("nan_nav", "invalid")
        self.assertIn("daily tape finite", [c["check"] for c in json.loads((kept / "validation.json").read_text())["checks"] if not c["passed"]])

    def test_wall_timeout(self):
        self.one("sleep", "timed_out", wall_seconds_per_scenario=6)

    def test_cpu_limit(self):
        self.one("burn_cpu", "cpu_limit", cpu_seconds_per_scenario=3)

    def test_memory_limit(self):
        self.one("allocate", "memory_limit", rss_mb=700)

    def test_death_mid_write_leaves_no_success(self):
        s, kept, _ = self.one("die_mid_write", "interrupted")
        self.assertTrue((kept / "daily.csv").exists(), "partial output preserved as evidence")

    def test_leftover_work_directory_is_preserved_and_charged(self):
        (self.root / "work").mkdir(parents=True)
        crashed = self.root / "work" / "s_a-deadbeef-a1"; crashed.mkdir(); (crashed / "daily.csv").write_text("partial")
        s = self.go(cfgs=self.cfgs[:1])
        self.assertEqual(s["scenarios"][0]["status"], "succeeded")
        self.assertTrue((self.root / "attempts" / "s_a-deadbeef-a1-interrupted" / "daily.csv").exists())
        self.assertEqual(self.ledger()[0]["status"], "interrupted"); self.assertEqual(self.ledger()[0]["cpu_seconds_charged"], 30)

    def test_unauthorized_diagnostic_setting_is_invalid_in_production(self):
        bad = [scen("s_diag", residual_policy="legacy_additive")]              # experiment E2 does not allow it
        s = self.go(cfgs=bad)
        self.assertEqual(s["scenarios"][0]["status"], "invalid")
        ok = [ScenarioConfig(name="s_diag_ok", eval_start=START, experiment="E3", strategy="momentum_top4", initial_cash=5e6, residual_policy="legacy_additive")]
        self.assertEqual(self.go(cfgs=ok)["scenarios"][0]["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
