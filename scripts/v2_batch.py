"""Run registered V2 scenarios through the validated, budgeted, resumable batch runner.

  v2_batch.py --experiment E1 [--scenario NAME ...] --snapshot <dir> --out <dir>
              --authorization <decision record json> --design-fingerprint <design receipt json> [--workspace <dir>]

Before anything is dispatched or any data is opened:
  1. the decision record must contain an `execution_authorization` block that names design fingerprint, code
     commit, market identity, input file hashes, environment, output root, scenario ids with attempt counts and
     a budget including a bound on parent-side CPU per scenario;
  2. the design receipt is recomputed from the files it lists (in --workspace, default two levels above the
     receipt), its commit must be this checkout's HEAD, its fingerprint must equal the hash of its components
     and commit, and its registry component must equal this code's registry;
  3. the receipt's fingerprint must be the authorized one.
A list of scenario names (`authorized_runs`) authorizes nothing here.
"""
import argparse, json, os, pathlib, sys
from ledger_consistent_etf_trading.v2 import batch, registry

REQUIRED = ("design_fingerprint", "code_commit", "market_id", "data_input_sha256", "environment", "output_root", "authorized", "budget")

ap = argparse.ArgumentParser()
ap.add_argument("--experiment"); ap.add_argument("--scenario", action="append", default=[]); ap.add_argument("--workspace")
for a in ("--snapshot", "--out", "--authorization", "--design-fingerprint"):
    ap.add_argument(a, required=True)
args = ap.parse_args()
record = json.loads(pathlib.Path(args.authorization).read_text())
auth = record.get("execution_authorization")
if not auth:
    raise SystemExit("the decision record has no execution_authorization block; refusing to run any historical scenario")
missing = [k for k in REQUIRED if k not in auth] + ([] if "parent_cpu_seconds_per_scenario" in auth.get("budget", {}) else ["budget.parent_cpu_seconds_per_scenario"])
if missing:
    raise SystemExit(f"execution_authorization lacks {missing}; refusing")
repo = pathlib.Path(__file__).resolve().parents[1]
receipt = pathlib.Path(args.design_fingerprint).resolve()
workspace = pathlib.Path(args.workspace).resolve() if args.workspace else (receipt.parents[2] if len(receipt.parents) > 2 else receipt.parent)
try:
    design = batch.verify_design_receipt(receipt, workspace, repo)
except batch.BatchStop as stop:
    raise SystemExit(f"refused: {stop}")
if design != auth["design_fingerprint"]:
    raise SystemExit("refused: the verified design receipt is not the authorized design")
allcfg = registry.build() + [registry.pilot()]
chosen = [c for c in allcfg if c.name in args.scenario or (args.experiment and c.experiment == args.experiment)]
if not chosen:
    raise SystemExit("no scenario selected")
os.environ["PV2_SNAPSHOT"] = str(pathlib.Path(args.snapshot).resolve())
try:
    summary = batch.run_batch(chosen, "ledger_consistent_etf_trading.v2.loaders:snapshot", pathlib.Path(args.out), repo, auth, design, production=True)
except batch.BatchStop as stop:
    raise SystemExit(f"refused: {stop}")
print(json.dumps(summary, indent=1))
sys.exit(0 if summary["stopped"] is None else 2)
