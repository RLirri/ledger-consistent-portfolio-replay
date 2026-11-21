"""Synthetic public-interface contracts; no historical execution or network."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from ledger_consistent_etf_trading.v2 import bootstrap, registry
from ledger_consistent_etf_trading.v2.loaders import UNIVERSE
from scripts import reproduction, statistical_recipes
from tests.helpers import synth

ROOT = Path(__file__).resolve().parents[1]


class SourceAndRecipes(unittest.TestCase):
    def test_exact_source(self):
        j = json.loads((ROOT / "docs/experimental_identity.json").read_text())
        for f, h in j["exact_source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / f).read_bytes()).hexdigest(), h)

    def test_registry(self):
        self.assertEqual(
            json.loads((ROOT / "configs/v2/registry.json").read_text())["scenarios"],
            registry.as_records(),
        )
        self.assertEqual(len(registry.build()), 62)

    def test_point_known_answer(self):
        a = np.array([100.0, 102.0, 101.0, 104.0])
        b = np.array([100.0, 101.0, 102.0, 103.0])
        r = statistical_recipes.point(a, b)
        self.assertAlmostEqual(
            r["active_return_ann"], 252 * np.mean(a[1:] / a[:-1] - b[1:] / b[:-1])
        )
        self.assertAlmostEqual(
            r["terminal_gap_ann_bps"], 10000 * ((103 / 104) ** 84 - 1)
        )

    def test_identical_arms_joint_resampling(self):
        r = np.array([0.01, -0.02, 0.005, 0.003, -0.001])
        ix = bootstrap.stationary_indices(len(r), 2000, 20, 20261119)
        for s in statistical_recipes.boot(r, r, ix)["statistics"].values():
            for k in ["point", "lower", "upper"]:
                self.assertAlmostEqual(s[k], 0.0)

    def test_analysis_import_no_expression_side_effect(self):
        t = ast.parse((ROOT / "scripts/analyze_saved_runs.py").read_text())
        self.assertFalse(
            [
                n
                for n in t.body
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            ]
        )


class OperatorPlans(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="replay-public-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo"
        (self.repo / "configs/v2").mkdir(parents=True)
        (self.repo / "docs").mkdir()
        for f in ["configs/v2/registry.json", "docs/MODEL.md"]:
            (self.repo / f).write_bytes((ROOT / f).read_bytes())
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Synthetic fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "Synthetic test fixture",
            ],
            check=True,
        )
        self.snapshot = self.base / "snapshot"
        out = self.snapshot / "data/processed"
        out.mkdir(parents=True)
        m = synth(tickers=tuple(sorted(UNIVERSE)), T=40)
        for i, t in enumerate(m.tickers):
            pd.DataFrame(
                {
                    "open": m.open[:, i],
                    "close": m.close[:, i],
                    "volume": m.volume[:, i],
                },
                index=m.dates,
            ).to_parquet(out / (t + "_1d_clean.parquet"))
        self.budget = dict(
            cpu_seconds_total=100,
            cpu_seconds_per_scenario=20,
            parent_cpu_seconds_per_scenario=10,
            wall_seconds_per_scenario=60,
            rss_mb=500,
        )

    def plan(self, **kw):
        a = dict(
            repo=self.repo,
            snapshot=self.snapshot,
            out=self.base / "results",
            plan=self.base / "plan",
            names=["E2_A_10m_cost0_cap0"],
            budget=self.budget,
            owner="synthetic fixture operator",
            rights=True,
        )
        a.update(kw)
        return reproduction.create_plan(**a)

    def test_plan_does_not_execute_or_inherit(self):
        with patch.object(reproduction.batch, "run_batch") as run:
            r = self.plan()
            run.assert_not_called()
        self.assertEqual(r["approval_origin"], "LOCAL_OPERATOR")
        self.assertNotEqual(
            r["execution_authorization"]["code_commit"],
            "8e8096fcac68adc7b2f3f7784035369c5895c4f3",
        )

    def test_missing_rights_and_nan_budget(self):
        with self.assertRaises(ValueError):
            self.plan(rights=False)
        with self.assertRaises(Exception):
            self.plan(budget=self.budget | {"rss_mb": float("nan")})
        self.assertFalse((self.base / "plan").exists())

    def test_tamper_never_dispatches(self):
        self.plan()
        (self.base / "plan/MODEL.md").write_text("tampered")
        with patch.object(reproduction.batch, "run_batch") as run:
            with self.assertRaises(Exception):
                reproduction.execute_plan(self.repo, self.base / "plan", True)
            run.assert_not_called()

    def test_separate_action_required(self):
        with self.assertRaises(ValueError):
            reproduction.execute_plan(self.repo, self.base / "absent", False)

    def test_pilot_duplicate_unknown_refused(self):
        for n in [["pilot_252"], ["unknown"], ["E2_A_10m_cost0_cap0"] * 2]:
            with self.assertRaises(ValueError):
                self.plan(names=n)
