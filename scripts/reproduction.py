"""Explicit local operator plans for the unchanged validated batch runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from ledger_consistent_etf_trading.v2 import batch, registry
from ledger_consistent_etf_trading.v2.loaders import UNIVERSE
from ledger_consistent_etf_trading.v2.data import load_snapshot
from ledger_consistent_etf_trading.v2.provenance import code_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_plan(repo, snapshot, out, plan, names, budget, owner, rights):
    """Bind an operator's new resource/data permission; never import author approval."""
    if not owner.strip() or not rights:
        raise ValueError(
            "a named local operator and explicit data-rights acknowledgement are required"
        )
    code = code_identity(repo)
    if code["dirty"]:
        raise ValueError(
            "commit or discard local changes before creating a new reproduction identity"
        )
    configs = {c.name: c for c in registry.build()}
    if not names or len(set(names)) != len(names) or set(names) - set(configs):
        raise ValueError(
            "select unique registered economic scenarios explicitly; pilot is excluded"
        )
    if plan.exists() or out.exists():
        raise FileExistsError(
            "plan and output must be new; do not reset an existing budget ledger"
        )
    # Snapshot read and strict domain validation only. No scenario is executed.
    market = load_snapshot(snapshot, UNIVERSE)
    market.validate()
    market.validate_domain(len(market.dates), strict=True)
    components = {
        "registry.json": sha(repo / "configs/v2/registry.json"),
        "MODEL.md": sha(repo / "docs/MODEL.md"),
    }
    fp = hashlib.sha256(
        json.dumps({**components, "code": code["commit"]}, sort_keys=True).encode()
    ).hexdigest()
    auth = {
        "design_fingerprint": fp,
        "code_commit": code["commit"],
        "market_id": market.identity(),
        "data_input_sha256": market.input_hashes,
        "environment": batch.environment(),
        "output_root": str(out.resolve()),
        "authorized": {configs[n].scenario_id: 1 for n in names},
        "budget": budget,
    }
    batch.check_authorization(
        auth,
        [configs[n] for n in names],
        code,
        market.identity(),
        fp,
        out,
        market.input_hashes,
    )
    if "parent_cpu_seconds_per_scenario" not in budget:
        raise ValueError("a finite parent-side CPU bound is mandatory")
    receipt = {
        "components_sha256": components,
        "v2_checkout_commit": code["commit"],
        "design_fingerprint": fp,
    }
    plan.mkdir(parents=True)
    shutil.copyfile(repo / "configs/v2/registry.json", plan / "registry.json")
    shutil.copyfile(repo / "docs/MODEL.md", plan / "MODEL.md")
    (plan / "design.json").write_text(json.dumps(receipt, indent=2))
    record = {
        "approval_origin": "LOCAL_OPERATOR",
        "operator": owner,
        "data_rights_acknowledged": True,
        "note": "New reproduction permission; not the paper author's historical authorization.",
        "snapshot": str(snapshot.resolve()),
        "scenarios": names,
        "execution_authorization": auth,
    }
    (plan / "authorization.json").write_text(json.dumps(record, indent=2))
    batch.verify_design_receipt(plan / "design.json", plan, repo)
    return record


def execute_plan(repo, plan, acknowledge):
    """Require a second explicit local action; keep original validation and accounting."""
    if not acknowledge:
        raise ValueError("--execute-local-plan is required")
    record = json.loads((plan / "authorization.json").read_text())
    if record.get("approval_origin") != "LOCAL_OPERATOR" or not record.get(
        "data_rights_acknowledged"
    ):
        raise ValueError("not a local operator reproduction plan")
    auth = record["execution_authorization"]
    fp = batch.verify_design_receipt(plan / "design.json", plan, repo)
    configs = {c.name: c for c in registry.build()}
    chosen = [configs[n] for n in record["scenarios"]]
    os.environ["PV2_SNAPSHOT"] = record["snapshot"]
    return batch.run_batch(
        chosen,
        "ledger_consistent_etf_trading.v2.loaders:snapshot",
        Path(auth["output_root"]),
        repo,
        auth,
        fp,
        production=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("registry")
    plan = sub.add_parser("plan")
    for key in ["snapshot", "out", "plan", "budget"]:
        plan.add_argument("--" + key, type=Path, required=True)
    plan.add_argument("--scenario", action="append", required=True)
    plan.add_argument("--owner", required=True)
    plan.add_argument("--acknowledge-data-rights", action="store_true")
    run = sub.add_parser("execute")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--execute-local-plan", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.action == "registry":
        print(json.dumps(registry.as_records(), indent=2))
    elif args.action == "plan":
        print(
            json.dumps(
                create_plan(
                    repo,
                    args.snapshot,
                    args.out,
                    args.plan,
                    args.scenario,
                    json.loads(args.budget.read_text()),
                    args.owner,
                    args.acknowledge_data_rights,
                ),
                indent=2,
            )
        )
    else:
        result = execute_plan(repo, args.plan, args.execute_local_plan)
        print(json.dumps(result, indent=2))
        if result["stopped"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
