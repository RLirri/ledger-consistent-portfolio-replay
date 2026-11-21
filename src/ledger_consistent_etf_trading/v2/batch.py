"""Deterministic, resumable, budgeted batch execution with validated atomic completion.

Identities
  scenario_id   design-level: configuration + protocol spec tag (unchanged by code fixes)
  run_spec_id   scenario_id + code commit + dirty hash + market identity + input hashes + design
                fingerprint + environment; a result is reusable only under the same run_spec_id
  attempt       one execution of a run spec; every attempt is recorded, including failures

Layout under the output root
  ledger.jsonl                       append-only attempt and budget record
  runs/<name>-<run_spec_id>/         published only after validation, read-back validation and hashing;
                                     COMPLETE.json is written last, atomically; directory made read-only
  attempts/<name>-<run_spec_id>-aN-<status>/   preserved failed, invalid, timed-out or interrupted attempts
  work/                              attempts in progress

Each attempt runs in a child process with a CPU limit (RLIMIT_CPU), a resident-memory watchdog and a
wall-clock timeout. The batch stops at the first attempt that does not succeed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import shutil
import stat
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import SPEC_VERSION, checks, engine, metrics
from .config import ScenarioConfig
from .data import build_features
from .provenance import code_identity

OUTPUTS = ("daily.csv", "orders.csv", "parents.csv", "demand.csv", "positions.csv", "target_weights.csv", "metrics.json", "validation.json")
EXIT_MEMORY = 86


class BatchStop(RuntimeError):
    """The batch was stopped; nothing was published for the triggering scenario."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def environment() -> dict:
    env = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()}
    try:
        import pyarrow
        env["pyarrow"] = pyarrow.__version__
    except Exception:                                       # recorded, not required by the engine itself
        env["pyarrow"] = None
    return env


def run_spec(cfg: ScenarioConfig, code: dict, market_id: str, input_hashes: dict, design_fingerprint: str, env: dict) -> dict:
    spec = {"spec_version": SPEC_VERSION, "scenario_id": cfg.scenario_id, "config": cfg.to_dict(), "code_commit": code["commit"],
            "code_dirty": code["dirty"], "code_dirty_sha256": code["dirty_sha256"], "market_id": market_id,
            "data_input_sha256": dict(sorted(input_hashes.items())), "design_fingerprint": design_fingerprint, "environment": env}
    spec["run_spec_id"] = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    return spec


def _load(loader: str):
    mod, fn = loader.split(":")
    return getattr(importlib.import_module(mod), fn)()


def write_outputs(out: Path, res, summary: dict, report: checks.Report) -> None:
    res.daily.to_csv(out / "daily.csv")
    res.orders.to_csv(out / "orders.csv", index=False)
    res.parents.to_csv(out / "parents.csv", index=False)
    res.demand.to_csv(out / "demand.csv", index=False)
    pd.DataFrame(res.positions, index=res.dates, columns=res.tickers).to_csv(out / "positions.csv")
    pd.DataFrame(res.target_eff, index=res.dates, columns=res.tickers).to_csv(out / "target_weights.csv")
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n")
    (out / "validation.json").write_text(json.dumps(report.to_dict(), indent=1, allow_nan=False) + "\n")


def _child(payload: dict) -> None:
    """Runs in a separate process. Writes child_result.json; never publishes."""
    work = Path(payload["work"])
    lim = payload["limits"]
    resource.setrlimit(resource.RLIMIT_CPU, (int(lim["cpu_seconds"]), int(lim["cpu_seconds"]) + 1))
    rss_limit = lim["rss_mb"] * 1024 * 1024
    to_bytes = (lambda r: r) if sys.platform == "darwin" else (lambda r: r * 1024)

    def watchdog():
        while True:
            if to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) > rss_limit:
                os._exit(EXIT_MEMORY)
            time.sleep(0.05)

    threading.Thread(target=watchdog, daemon=True).start()
    result = {"status": "failed", "error": None}
    t0 = time.process_time(); w0 = time.time()
    phases = {}
    try:
        fault = payload.get("fault")                        # test hook only; the command-line tools never set it
        cfg = ScenarioConfig.from_dict(payload["config"])
        mkt = _load(payload["loader"])
        if mkt.identity() != payload["market_id"]:
            raise RuntimeError("market data changed between authorization and execution")
        feats = build_features(mkt, cfg.features, cfg.adv_estimator)
        phases["load_and_features_cpu"] = time.process_time() - t0
        if fault == "sleep":
            time.sleep(3600)
        if fault == "burn_cpu":
            while True:
                pass
        if fault == "allocate":
            hog = [np.ones(8_000_000) for _ in range(400)]  # about 25 GB if allowed to finish
            time.sleep(5)
        if fault == "raise":
            raise RuntimeError("injected failure")
        t1 = time.process_time()
        res = engine.run(mkt, cfg, feats=feats, production=payload["production"])
        phases["replay_cpu"] = time.process_time() - t1
        if fault == "corrupt_cash":
            res.daily.loc[res.daily.index[-1], "cash"] += 1_000.0
        if fault == "nan_nav":
            res.daily.loc[res.daily.index[-1], "nav"] = float("nan")
        t2 = time.process_time()
        report = checks.validate_result(res, mkt, feats, production=payload["production"])
        phases["validation_cpu"] = time.process_time() - t2
        if report.ok:
            t3 = time.process_time()
            summary = metrics.summarize(res, mkt.close)
            phases["metrics_cpu"] = time.process_time() - t3
        else:
            summary = {"scenario_id": cfg.scenario_id, "name": cfg.name, "status": "invalid: metrics withheld"}
        t4 = time.process_time()
        if fault == "die_mid_write":
            res.daily.to_csv(work / "daily.csv")
            os._exit(1)
        write_outputs(work, res, summary, report)
        phases["write_cpu"] = time.process_time() - t4
        result = {"status": "ok" if report.ok else "invalid", "error": None if report.ok else [c["check"] for c in report.failures],
                  "events": {"sessions": int(len(res.daily)), "order_rows": int(len(res.orders)), "parents": int(len(res.parents))}}
    except BaseException as exc:                            # typed input/state errors and anything else: an explicit failure record
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    result.update(phases_cpu_seconds=phases, cpu_seconds=time.process_time() - t0, wall_seconds=time.time() - w0,
                  peak_rss_mib=to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1048576)
    _atomic_json(work / "child_result.json", result)


def _ledger(root: Path) -> list:
    p = root / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _append(root: Path, rec: dict) -> None:
    with (root / "ledger.jsonl").open("a") as f:
        f.write(json.dumps(rec, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def verify_completed(run_dir: Path, spec: dict) -> tuple:
    """(ok, reason) for an existing published run directory against the current run spec."""
    marker = run_dir / "COMPLETE.json"
    if not marker.exists():
        return False, "no completion marker"
    done = json.loads(marker.read_text())
    if done.get("run_spec_id") != spec["run_spec_id"]:
        return False, "run spec differs"
    man = json.loads((run_dir / "manifest.json").read_text())
    if man.get("run_spec") != spec:
        return False, "manifest run spec differs"
    for name, h in done["output_sha256"].items():
        f = run_dir / name
        if not f.exists() or _sha(f) != h:
            return False, f"output changed or missing: {name}"
    if _sha(run_dir / "manifest.json") != done["manifest_sha256"]:
        return False, "manifest changed"
    return True, "verified"


def _num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _posint(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, int) and v >= 1


def check_authorization(auth: dict, cfgs: list, code: dict, market_id: str, design_fingerprint: str, out_root: Path | None = None,
                        input_hashes: dict | None = None) -> None:
    need = ("design_fingerprint", "code_commit", "market_id", "authorized", "budget")
    if not isinstance(auth, dict) or any(k not in auth for k in need):
        raise BatchStop("no execution authorization bound to design, code, data, scenarios and budget; refusing to run")
    if auth["design_fingerprint"] != design_fingerprint:
        raise BatchStop("design fingerprint differs from the authorized design")
    if auth["code_commit"] != code["commit"] or (code["dirty"] and not auth.get("allow_dirty", False)):
        raise BatchStop("code differs from the authorized commit, or the working tree is dirty")
    if auth["market_id"] != market_id:
        raise BatchStop("market data differ from the authorized inputs")
    if "output_root" in auth and (out_root is None or Path(auth["output_root"]).resolve() != Path(out_root).resolve()):
        raise BatchStop("output root differs from the authorized one; attempt and budget accounting cannot be bypassed by moving the output")
    if "data_input_sha256" in auth and dict(auth["data_input_sha256"]) != dict(input_hashes or {}):
        raise BatchStop("input file hashes differ from the authorized inputs")
    if "environment" in auth:
        env = environment()
        diff = sorted(k for k in ("python", "pandas", "numpy", "pyarrow") if auth["environment"].get(k) != env.get(k))
        if diff:
            raise BatchStop(f"environment differs from the authorized one in {diff}")
    if not isinstance(auth["authorized"], dict):
        raise BatchStop("authorized must map scenario ids to attempt counts")
    for c in cfgs:
        if c.scenario_id not in auth["authorized"]:
            raise BatchStop(f"scenario {c.name} ({c.scenario_id}) is not authorized")
        if not _posint(auth["authorized"][c.scenario_id]):
            raise BatchStop(f"attempt allowance for {c.name} must be a positive integer")
    b = auth["budget"]
    if not isinstance(b, dict):
        raise BatchStop("budget must be a mapping")
    for k in ("cpu_seconds_total", "wall_seconds_per_scenario", "rss_mb"):
        if not _num(b.get(k)):
            raise BatchStop(f"authorization budget lacks a finite positive {k}")
    if not _posint(b.get("cpu_seconds_per_scenario")):
        raise BatchStop("cpu_seconds_per_scenario must be a positive whole number of seconds (the kernel CPU limit has one-second resolution)")
    if "parent_cpu_seconds_per_scenario" in b and not _num(b["parent_cpu_seconds_per_scenario"]):
        raise BatchStop("parent_cpu_seconds_per_scenario must be finite and positive when given")


def verify_design_receipt(receipt_path: Path, workspace: Path, repo: Path) -> str:
    """Recompute a design receipt from the files it names. Returns the fingerprint; raises BatchStop on any mismatch.

    Checked: every component file exists and has the recorded SHA-256; the receipt's commit is the checkout's HEAD;
    the fingerprint equals SHA-256 of the sorted JSON of {components..., "code": commit}; a registry component is
    present and its scenarios equal the code's own registry."""
    try:
        rec = json.loads(Path(receipt_path).read_text())
        comps, commit, fp = rec["components_sha256"], rec["v2_checkout_commit"], rec["design_fingerprint"]
    except Exception as exc:
        raise BatchStop(f"design receipt unreadable or incomplete: {type(exc).__name__}: {exc}")
    if not isinstance(comps, dict) or not comps:
        raise BatchStop("design receipt lists no components")
    for rel, h in comps.items():
        f = Path(workspace) / rel
        if not f.is_file():
            raise BatchStop(f"design component missing: {rel}")
        if _sha(f) != h:
            raise BatchStop(f"design component changed since the receipt: {rel}")
    head_commit = code_identity(Path(repo))["commit"]
    if commit != head_commit:
        raise BatchStop("design receipt names a different implementation commit than the checkout")
    if hashlib.sha256(json.dumps({**comps, "code": commit}, sort_keys=True).encode()).hexdigest() != fp:
        raise BatchStop("design fingerprint does not equal the hash of its components and commit")
    reg = [k for k in comps if "registry" in k]
    if not reg:
        raise BatchStop("design receipt binds no scenario registry")
    from . import registry
    for k in reg:
        if json.loads((Path(workspace) / k).read_text()).get("scenarios") != registry.as_records():
            raise BatchStop(f"scenario registry component {k} differs from the implementation's registry")
    return fp


def _self_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _children_cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def accounting(led: list) -> dict:
    """Budget and attempt state from the ledger. A reservation with no settlement is charged at its reserved amount."""
    settled, reserved, overhead = {}, {}, 0.0
    for r in led:
        if r.get("event") == "reserved":
            reserved[r["key"]] = r
        elif r.get("event") == "attempt":
            settled[r.get("key") or f"legacy:{len(settled)}"] = r
        elif r.get("event") == "overhead":
            overhead += r.get("cpu_seconds_charged", 0.0)
    cpu = overhead + sum(r.get("cpu_seconds_charged", 0.0) for r in settled.values())
    cpu += sum(r["reserved_cpu_seconds"] for k, r in reserved.items() if k not in settled)
    attempts = {}
    for k in set(settled) | set(reserved):
        sid = (settled.get(k) or reserved[k]).get("scenario_id")
        if sid:
            attempts[sid] = attempts.get(sid, 0) + 1
    return {"cpu": cpu, "attempts": attempts, "settled": settled, "reserved": reserved, "n_attempts": len(set(settled) | set(reserved))}


def _archive(src: Path, dest: Path) -> Path:
    n = 0
    target = dest
    while target.exists():
        n += 1
        target = dest.with_name(f"{dest.name}.{n}")
    os.replace(src, target)
    return target


def recover(out_root: Path, budget: dict) -> list:
    """Idempotent reconciliation of whatever an interrupted invocation left behind. Every started attempt ends with
    exactly one settlement; nothing recovered here can be run again without a new attempt allowance."""
    notes = []
    reserve = budget["cpu_seconds_per_scenario"] + budget.get("parent_cpu_seconds_per_scenario", 0)
    acct = accounting(_ledger(out_root))
    for stale in sorted((out_root / "work").iterdir()):
        ident = {}
        try:
            st = json.loads((stale / "status.json").read_text())
            sp = st["run_spec"]
            ident = {"scenario_id": sp["scenario_id"], "run_spec_id": sp["run_spec_id"], "attempt": int(st["attempt"]), "name": sp["config"]["name"]}
            key = f"{ident['run_spec_id']}:a{ident['attempt']}"
        except Exception:
            key = f"unidentified:{stale.name}"
            for r in acct["reserved"].values():              # no readable status: identify the attempt from its reservation
                if stale.name == f"{r['name']}-{r['run_spec_id']}-a{r['attempt']}":
                    ident = {"scenario_id": r["scenario_id"], "run_spec_id": r["run_spec_id"], "attempt": r["attempt"], "name": r["name"]}
                    key = r["key"]
        dest = _archive(stale, out_root / "attempts" / (stale.name + "-interrupted"))
        if key not in acct["settled"]:
            charged = acct["reserved"].get(key, {}).get("reserved_cpu_seconds", reserve)
            _append(out_root, {"event": "attempt", "key": key, **ident, "status": "interrupted", "dir": str(dest.relative_to(out_root)),
                               "cpu_seconds_charged": charged, "recovered": True, "note": "found in work/ at batch start; charged at the reserved amount"})
            notes.append(f"{stale.name}: interrupted attempt settled")
            acct = accounting(_ledger(out_root))
    for key, r in acct["reserved"].items():
        if key in acct["settled"]:
            continue
        final = out_root / "runs" / f"{r['name']}-{r['run_spec_id']}"
        marker = final / "COMPLETE.json"
        rec = {"event": "attempt", "key": key, "name": r["name"], "scenario_id": r["scenario_id"], "run_spec_id": r["run_spec_id"],
               "attempt": r["attempt"], "recovered": True}
        done = json.loads(marker.read_text()) if marker.exists() else {}
        if done.get("run_spec_id") == r["run_spec_id"] and done.get("attempt") == r["attempt"]:
            rec.update(status="succeeded", dir=str(final.relative_to(out_root)), cpu_seconds_charged=float(done.get("cpu_seconds_charged", r["reserved_cpu_seconds"])),
                       note="published before the ledger was written; settled from the completion marker")
        else:
            arch = sorted((out_root / "attempts").glob(f"{r['name']}-{r['run_spec_id']}-a{r['attempt']}-*"))
            status = "interrupted"
            if arch:
                try:
                    status = json.loads((arch[0] / "status.json").read_text()).get("status", "interrupted")
                except Exception:
                    pass
            rec.update(status=status if status != "running" else "interrupted", dir=str(arch[0].relative_to(out_root)) if arch else None,
                       cpu_seconds_charged=r["reserved_cpu_seconds"], note="reservation without settlement; charged at the reserved amount")
        _append(out_root, rec)
        notes.append(f"{key}: settled as {rec['status']}")
    return notes


def run_batch(cfgs: list, loader: str, out_root: Path, repo: Path, authorization: dict, design_fingerprint: str,
              production: bool = True, faults: dict | None = None) -> dict:
    """Run scenarios in deterministic (name) order. Returns a summary; raises BatchStop on refusal.

    Accounting. Before a child starts, a reservation (scenario, run spec, attempt, reserved CPU) is appended to the
    ledger. Every reservation ends in exactly one settlement, written by this invocation or recovered by a later one.
    An attempt allowance is consumed at reservation. Charged CPU is the child's whole process CPU (interpreter start
    included, from the kernel's child accounting) plus the parent's measured validation work; the sum must fit the
    total before a result is published.
    """
    overhead0 = _self_cpu()
    out_root = Path(out_root)
    for sub in ("runs", "attempts", "work"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    faults = faults or {}
    cfgs = sorted(cfgs, key=lambda c: c.name)
    if len({c.name for c in cfgs}) != len(cfgs):
        raise BatchStop("duplicate scenario names")
    code = code_identity(Path(repo))
    mkt = _load(loader)
    market_id = mkt.identity()
    check_authorization(authorization, cfgs, code, market_id, design_fingerprint, out_root, mkt.input_hashes)
    budget = authorization["budget"]
    parent_bound = budget.get("parent_cpu_seconds_per_scenario")
    reserve = budget["cpu_seconds_per_scenario"] + (parent_bound or 0)
    env = environment()
    summary = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": code, "market_id": market_id,
               "design_fingerprint": design_fingerprint, "budget": budget, "scenarios": [], "stopped": None}
    summary["recovered"] = recover(out_root, budget)
    _append(out_root, {"event": "overhead", "cpu_seconds_charged": _self_cpu() - overhead0, "note": "parent start-up: load, authorization, recovery"})

    for cfg in cfgs:
        spec = run_spec(cfg, code, market_id, mkt.input_hashes, design_fingerprint, env)
        tag = f"{cfg.name}-{spec['run_spec_id']}"
        final = out_root / "runs" / tag
        entry = {"name": cfg.name, "scenario_id": cfg.scenario_id, "run_spec_id": spec["run_spec_id"]}
        others = sorted(p.name for p in (out_root / "runs").glob(cfg.name + "-*") if p.name != tag)
        if others:
            entry["incompatible_prior_results_not_reused"] = others
        if final.exists():
            ok, why = verify_completed(final, spec)
            entry.update(status="skipped_verified" if ok else "stale_or_tampered", detail=why)
            summary["scenarios"].append(entry)
            if not ok:
                summary["stopped"] = f"{cfg.name}: existing result failed verification ({why}); nothing overwritten"
                break
            continue
        acct = accounting(_ledger(out_root))
        attempts = acct["attempts"].get(cfg.scenario_id, 0)
        if attempts >= authorization["authorized"][cfg.scenario_id]:
            entry.update(status="refused", detail=f"authorized attempts exhausted ({attempts})")
            summary["scenarios"].append(entry); summary["stopped"] = f"{cfg.name}: attempts exhausted"
            break
        if acct["cpu"] + reserve > budget["cpu_seconds_total"]:
            entry.update(status="refused", detail=f"cumulative CPU budget would be exceeded ({acct['cpu']:.1f}s used of {budget['cpu_seconds_total']}s)")
            summary["scenarios"].append(entry); summary["stopped"] = f"{cfg.name}: budget exhausted"
            break
        n = attempts + 1
        key = f"{spec['run_spec_id']}:a{n}"
        _append(out_root, {"event": "reserved", "key": key, "name": cfg.name, "scenario_id": cfg.scenario_id, "run_spec_id": spec["run_spec_id"],
                           "attempt": n, "reserved_cpu_seconds": reserve})
        work = out_root / "work" / f"{tag}-a{n}"
        work.mkdir()
        _atomic_json(work / "status.json", {"status": "running", "run_spec": spec, "attempt": n, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        payload = {"config": cfg.to_dict(), "loader": loader, "work": str(work), "market_id": market_id, "production": production,
                   "limits": {"cpu_seconds": budget["cpu_seconds_per_scenario"], "rss_mb": budget["rss_mb"]}, "fault": faults.get(cfg.name)}
        w0 = time.time(); parent_cpu0 = time.process_time(); child_cpu0 = _children_cpu()
        proc = mp.get_context("spawn").Process(target=_child, args=(payload,))
        proc.start()
        proc.join(budget["wall_seconds_per_scenario"])
        if proc.is_alive():
            proc.kill(); proc.join()
            status, error = "timed_out", f"wall limit {budget['wall_seconds_per_scenario']}s"
        elif (work / "child_result.json").exists():
            child = json.loads((work / "child_result.json").read_text())
            status, error = child["status"], child.get("error")
        elif proc.exitcode == EXIT_MEMORY:
            status, error = "memory_limit", f"resident memory exceeded {budget['rss_mb']} MB"
        elif proc.exitcode is not None and proc.exitcode < 0:
            status, error = ("cpu_limit", f"CPU limit {budget['cpu_seconds_per_scenario']}s") if -proc.exitcode in (24, 9) else ("killed", f"signal {-proc.exitcode}")
        else:
            status, error = "interrupted", f"child exited with code {proc.exitcode} before reporting"
        child = json.loads((work / "child_result.json").read_text()) if (work / "child_result.json").exists() else {}
        cpu = _children_cpu() - child_cpu0                   # whole child process, interpreter start-up included
        record = {"event": "attempt", "key": key, "name": cfg.name, "scenario_id": cfg.scenario_id, "run_spec_id": spec["run_spec_id"], "attempt": n,
                  "status": status, "error": error, "cpu_seconds": cpu, "child_reported_cpu_seconds": child.get("cpu_seconds"),
                  "peak_rss_mib": child.get("peak_rss_mib"), "phases_cpu_seconds": child.get("phases_cpu_seconds"), "events": child.get("events")}
        outs = None
        if status == "ok":
            # read-back validation in the parent, on the files, before anything is published
            feats = build_features(mkt, cfg.features, cfg.adv_estimator)
            manifest = {"spec_version": SPEC_VERSION, "scenario_id": cfg.scenario_id, "config": cfg.to_dict(), "run_spec": spec, "attempt": n,
                        "run_info": json.loads(json.dumps(_run_info(work, cfg, mkt, feats))),
                        "authorization": {k: authorization.get(k) for k in ("design_fingerprint", "code_commit", "market_id", "budget", "receipt", "environment", "data_input_sha256", "output_root")},
                        "resources": {k: record[k] for k in ("cpu_seconds", "child_reported_cpu_seconds", "peak_rss_mib", "phases_cpu_seconds", "events")}}
            _atomic_json(work / "manifest.json", manifest)
            try:
                rb = checks.validate_saved(work, mkt, feats, production=production)
            except Exception as exc:
                rb = None
                status, error = "invalid", f"read-back failed: {type(exc).__name__}: {exc}"
            if rb is not None and not rb.ok:
                status, error = "invalid", ["read-back: " + c["check"] for c in rb.failures]
            if status == "ok":
                _atomic_json(work / "readback_validation.json", rb.to_dict())
                outs = {name: _sha(work / name) for name in OUTPUTS + ("readback_validation.json",)}
        parent_cpu = time.process_time() - parent_cpu0      # second and last reading of this clock for the attempt
        record["parent_cpu_seconds"] = parent_cpu
        record["cpu_seconds_charged"] = cpu + parent_cpu
        if status == "ok":
            total_after = acct["cpu"] + record["cpu_seconds_charged"]
            if total_after > budget["cpu_seconds_total"] or (parent_bound is not None and parent_cpu > parent_bound):
                status = "budget_exceeded"
                error = (f"charged CPU would reach {total_after:.2f}s of {budget['cpu_seconds_total']}s" if total_after > budget["cpu_seconds_total"]
                         else f"parent work {parent_cpu:.2f}s exceeded its bound {parent_bound}s") + "; result not published"
        if status == "ok":
            (work / "status.json").unlink()
            (work / "child_result.json").unlink()
            _atomic_json(work / "COMPLETE.json", {"status": "succeeded", "run_spec_id": spec["run_spec_id"], "scenario_id": cfg.scenario_id,
                                                  "attempt": n, "output_sha256": outs, "manifest_sha256": _sha(work / "manifest.json"),
                                                  "cpu_seconds_charged": record["cpu_seconds_charged"],
                                                  "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            for f in work.iterdir():
                os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.replace(work, final)                         # atomic publication
            os.chmod(final, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
            record["status"] = "succeeded"; record["dir"] = str(final.relative_to(out_root))
        else:
            record["status"], record["error"] = status, error
            _atomic_json(work / "status.json", {"status": status, "error": error, "run_spec": spec, "attempt": n})
            dest = _archive(work, out_root / "attempts" / f"{tag}-a{n}-{status}")
            record["dir"] = str(dest.relative_to(out_root))
        record["wall_seconds"] = time.time() - w0
        _append(out_root, record)                           # settlement; if this is lost, recover() settles from the marker or archive
        entry.update(status=record["status"], detail=record["error"], attempt=n, cpu_seconds=cpu, dir=record["dir"])
        summary["scenarios"].append(entry)
        if record["status"] != "succeeded":
            summary["stopped"] = f"{cfg.name}: {record['status']}"
            break
    acct = accounting(_ledger(out_root))
    summary["cpu_seconds_charged_total"] = acct["cpu"]
    summary["attempts_total"] = acct["n_attempts"]
    summary["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _append(out_root, {"event": "batch", "stopped": summary["stopped"], "statuses": [s["status"] for s in summary["scenarios"]],
                       "cpu_seconds_charged_total": summary["cpu_seconds_charged_total"]})
    return summary


def _run_info(work: Path, cfg: ScenarioConfig, mkt, feats) -> dict:
    """Run metadata needed to validate from disk; recomputed here, not trusted from the child."""
    from .config import information_class
    from .data import check_features, session_range
    from .strategies import build_targets
    start, end = session_range(mkt, cfg)
    targets = build_targets(mkt, cfg, start, end)
    return {"information": information_class(cfg.valuation, cfg.features), "dollar_tolerance": 1e-9 * max(1.0, cfg.initial_cash),
            "start": start, "end": end, "features_binding": check_features(mkt, cfg, feats), "features_market_id": feats.market_id,
            "market_id": mkt.identity(), "feature_window": feats.window, "target_sessions": sorted(int(k) for k in targets),
            "targets_source": "strategy"}
