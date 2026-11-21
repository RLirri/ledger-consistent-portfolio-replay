"""Fixed saved-path analysis; importing this module performs no analysis or replay."""

import os, sys, pathlib, json, hashlib, resource, threading, time, itertools, math, argparse
import numpy as np
import pandas as pd
from ledger_consistent_etf_trading.v2 import (
    batch,
    bootstrap,
    registry,
    contrasts,
    factorial,
    metrics,
)

PERIODS = metrics.SUBPERIODS
reg = {c.name: c for c in registry.build()}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_seconds": u.ru_utime + u.ru_stime,
        "wall_seconds": time.monotonic() - start,
        "peak_rss_mib": u.ru_maxrss / (1048576 if sys.platform == "darwin" else 1024),
    }


def write(name, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(z) for z in x]
        if isinstance(x, float):
            return x if math.isfinite(x) else None
        if hasattr(x, "item"):
            return clean(x.item())
        return x

    p = A / name
    p.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n")


def watch():
    while True:
        u = usage()
        if u["cpu_seconds"] > 230 or u["peak_rss_mib"] > 1900:
            write("resources.json", u | {"status": "resource_stopped"})
            os._exit(3)
        time.sleep(0.05)


def returns(v):
    return v[1:] / v[:-1] - 1


def point(v, ref):
    assert len(v) == len(ref) and np.all(v > 0) and np.all(ref > 0)
    ra, rb = returns(v), returns(ref)
    n = len(ra)
    sa, sb = ra.std(), rb.std()
    return {
        "n_returns": n,
        "terminal_gap_bps": 1e4 * (ref[-1] - v[-1]) / ref[-1],
        "terminal_gap_ann_bps": 1e4 * np.expm1((np.log(ref[-1] / v[-1])) * 252 / n),
        "active_return_ann": 252 * float(np.mean(ra - rb)),
        "return_te_ann": float(np.std(ra - rb, ddof=1) * np.sqrt(252))
        if n > 1
        else None,
        "sharpe_difference": float(np.sqrt(252) * (ra.mean() / sa - rb.mean() / sb))
        if sa > 0 and sb > 0
        else None,
    }


def boot(ra, rb, ix):
    n = len(ra)
    B = len(ix)
    draws = {
        k: [] for k in ("active_return_ann", "sharpe_difference", "terminal_gap_ann")
    }
    for off in range(0, B, 64):
        j = ix[off : off + 64]
        x, y = ra[j], rb[j]
        sx, sy = x.std(axis=1), y.std(axis=1)
        draws["active_return_ann"].extend((252 * (x - y).mean(axis=1)).tolist())
        with np.errstate(divide="ignore", invalid="ignore"):
            draws["sharpe_difference"].extend(
                (np.sqrt(252) * (x.mean(axis=1) / sx - y.mean(axis=1) / sy)).tolist()
            )
        draws["terminal_gap_ann"].extend(
            np.expm1(
                252 / n * (np.log1p(y).sum(axis=1) - np.log1p(x).sum(axis=1))
            ).tolist()
        )
    pointvalues = {
        "active_return_ann": float(252 * (ra - rb).mean()),
        "sharpe_difference": float(
            np.sqrt(252) * (ra.mean() / ra.std() - rb.mean() / rb.std())
        )
        if ra.std() > 0 and rb.std() > 0
        else None,
        "terminal_gap_ann": float(
            np.expm1(252 / n * (np.log1p(rb).sum() - np.log1p(ra).sum()))
        ),
    }
    out = {}
    for k, d in draws.items():
        z = np.array(d)
        bad = int((~np.isfinite(z)).sum())
        p = pointvalues[k]
        out[k] = {
            "point": p,
            "undefined_replicates": bad,
            "lower": float(np.quantile(z, 0.025))
            if not bad and p is not None
            else None,
            "upper": float(np.quantile(z, 0.975))
            if not bad and p is not None
            else None,
            "bootstrap_sd": float(z.std(ddof=1)) if not bad and p is not None else None,
        }
    return {
        "replicates": 2000,
        "seed": 20261119,
        "level": 0.95,
        "pointwise": True,
        "statistics": out,
    }


def factor(cells, k):
    beta = factorial.effects(cells, k)
    return {
        "cells": {"".join(map(str, c)): v for c, v in cells.items()},
        "beta": {
            ",".join(map(str, s)) if s else "grand_mean": v for s, v in beta.items()
        },
        "factorial_effects_2beta": {
            ",".join(map(str, s)): 2 * v for s, v in beta.items() if s
        },
        "max_reconstruction_error": factorial.reconstruction_error(cells, k),
    }


def main():
    # The saved manifests establish identity; no private historical authorization is imported.
    candidates = sorted((OUT / "runs").glob("*/manifest.json"))
    if not candidates:
        raise ValueError("no saved manifests")
    first = json.loads(candidates[0].read_text())["run_spec"]
    auth = {
        k: first[k]
        for k in ["code_commit", "design_fingerprint", "market_id", "data_input_sha256"]
    }
    auth["authorized"] = {c.scenario_id: 1 for c in reg.values()}
    runs = {}
    for p in (OUT / "runs").iterdir():
        m = json.loads((p / "manifest.json").read_text())
        sp = m["run_spec"]
        if (
            sp["code_commit"] != auth["code_commit"]
            or sp["design_fingerprint"] != auth["design_fingerprint"]
        ):
            continue
        ok, why = batch.verify_completed(p, sp)
        assert ok, why
        assert (
            sp["scenario_id"] in auth["authorized"]
            and sp["market_id"] == auth["market_id"]
            and sp["data_input_sha256"] == auth["data_input_sha256"]
        )
        name = sp["config"]["name"]
        assert name not in runs
        d = pd.read_csv(
            p / "daily.csv", index_col=0, parse_dates=True, float_precision="round_trip"
        )
        met = json.loads((p / "metrics.json").read_text())
        runs[name] = {
            "path": p,
            "manifest": m,
            "metrics": met,
            "daily": d,
            "nav": d.nav.to_numpy(),
            "id": sp["run_spec_id"],
        }
    assert set(runs) == set(reg), (
        "Every final scenario required; missing cells remain missing"
    )
    idx = next(iter(runs.values()))["daily"].index
    assert all(r["daily"].index.equals(idx) for r in runs.values())
    score = [
        {
            "name": n,
            "experiment": reg[n].experiment,
            "run_spec_id": r["id"],
            "run_directory": str(r["path"]),
            "code_commit": auth["code_commit"],
            "design_fingerprint": auth["design_fingerprint"],
            **r["metrics"],
        }
        for n, r in sorted(runs.items())
    ]
    write("scorecard.json", score)
    pd.json_normalize(score).drop(columns=["subperiods"], errors="ignore").to_csv(
        A / "scorecard.csv", index=False
    )
    subscore = [
        {"name": n, "run_spec_id": r["id"], "period": j, **z}
        for n, r in runs.items()
        for j, z in enumerate(r["metrics"]["subperiods"])
    ]
    write("subperiod_scorecard.json", subscore)
    pd.json_normalize(subscore).to_csv(A / "subperiod_scorecard.csv", index=False)
    post = {}
    for name, r in runs.items():
        if reg[name].experiment == "E2" and name.endswith("cost0_cap0"):
            p = OUT / "derived" / f"{name}.posthoc.csv"
            d = pd.read_csv(
                p, index_col=0, parse_dates=True, float_precision="round_trip"
            )
            assert d.index.equals(idx)
            meta = json.loads(p.with_suffix(".json").read_text())
            assert (
                sha(p) == meta["output_sha256"]
                and meta["reference_run_spec_id"] == r["id"]
            )
            post[name] = d.nav.to_numpy()
    assert len(post) == 6
    # Same indices across every arm/contrast, generated once per block length; n is fixed before reading outcomes.
    index_cache = {
        L: bootstrap.stationary_indices(len(idx) - 1, 2000, L, 20261119)
        for L in (20, 60)
    }
    crows = []
    for j, recipe in enumerate(contrasts.build(), 1):
        arm, ref = recipe["arm"], recipe["reference"]
        ra, rref = runs[arm], runs[ref]
        ispost = recipe["kind"] == "vs_posthoc"
        v = ra["nav"]
        b = post[ref] if ispost else rref["nav"]
        contrasts.check_matched(reg[arm], reg[ref], set(recipe["allowed"]))
        assert reg[arm].initial_cash == reg[ref].initial_cash
        d0a = ra["metrics"]["deployment_session"]
        d0b = rref["metrics"]["deployment_session"]
        assert d0a == d0b
        row = recipe | {
            "contrast_id": f"C{j:03d}",
            "reference_source": ref,
            "reference": "posthoc::" + ref if ispost else ref,
            "arm_run_spec_id": ra["id"],
            "reference_source_run_spec_id": rref["id"],
            "effect_class": (
                "cost_feedback"
                if not reg[arm].cap_enabled
                else "feedback_plus_liquidity"
            )
            if ispost
            else "registered_replay_contrast",
            **point(v, b),
        }
        row["material_active_return_5bps"] = abs(row["active_return_ann"]) >= 0.0005
        row["material_catchup_ratio_5bps"] = abs(row["terminal_gap_ann_bps"]) >= 5
        row["arm_completion"] = ra["metrics"]["orders"]["completion_aggregate"]
        row["reference_completion"] = (
            None if ispost else rref["metrics"]["orders"]["completion_aggregate"]
        )
        row["completion_own_demand_only"] = True
        row["bootstrap"] = {
            str(L): boot(returns(v), returns(b), ix) | {"mean_block": L}
            for L, ix in index_cache.items()
        }
        row["subperiods"] = []
        for first, last in PERIODS:
            mask = (idx >= pd.Timestamp(first, tz="UTC")) & (
                idx <= pd.Timestamp(last, tz="UTC")
            )
            loc = np.flatnonzero(mask)
            # Arithmetic active returns from saved preceding-close intervals; no unmatched subperiod wealth-level gap.
            r1 = returns(v)
            r2 = returns(b)
            keep = mask[1:]
            row["subperiods"].append(
                {
                    "first": first,
                    "last": last,
                    "n_returns": int(keep.sum()),
                    "active_return_ann": float(252 * (r1 - r2)[keep].mean()),
                    "return_te_ann": float(
                        np.std((r1 - r2)[keep], ddof=1) * np.sqrt(252)
                    ),
                    "basis": "daily returns assigned to their end session, common predecessor NAV; no raw unmatched-start wealth ratio",
                }
            )
        if recipe["experiment"] == "E1":
            oa = pd.read_csv(ra["path"] / "orders.csv", float_precision="round_trip")
            ob = pd.read_csv(rref["path"] / "orders.csv", float_precision="round_trip")
            xa = oa.groupby(["session", "ticker"]).filled_units.sum()
            xb = ob.groupby(["session", "ticker"]).filled_units.sum()
            xa, xb = xa.align(xb, fill_value=0)
            prices = (
                oa.groupby(["session", "ticker"])
                .fill_price.first()
                .combine_first(ob.groupby(["session", "ticker"]).fill_price.first())
                .reindex(xa.index)
            )
            row["differing_asset_session_fills"] = int((xa != xb).sum())
            row["absolute_fill_difference_notional"] = float(
                ((xa - xb).abs() * prices).sum()
            )
        crows.append(row)
        if j % 6 == 0:
            write("contrasts.partial.json", crows)
            print("contrasts", j, "CPU", usage()["cpu_seconds"], flush=True)
    write("contrasts.json", crows)
    flat = [
        {k: v for k, v in x.items() if k not in ("bootstrap", "subperiods", "allowed")}
        for x in crows
    ]
    pd.json_normalize(flat).to_csv(A / "contrasts.csv", index=False)
    write("primary_contrasts.json", [x for x in crows if x["tier"] == "primary"])
    # Independently validated chosen bootstrap implementation is exercised once end-to-end as a cross-check of vectorized recipe.
    ex = crows[0]
    ar = runs[ex["arm"]]["nav"]
    br = runs[ex["reference_source"]]["nav"]
    p = bootstrap.paired_bootstrap(pd.Series(ar, index=idx), pd.Series(br, index=idx))
    for stat, z in p["statistics"].items():
        for key in ("point", "lower", "upper"):
            u = ex["bootstrap"]["20"]["statistics"][stat][key]
            v = z[key]
            assert (u is None and v is None) or np.isclose(u, v, rtol=1e-9, atol=1e-11)
    write(
        "bootstrap_crosscheck.json",
        {
            "contrast": ex["contrast_id"],
            "production_result": p,
            "vectorized_match": True,
        },
    )
    e2 = []
    for s in ("A", "C"):
        for cap in ("10m", "100m", "1b"):
            base = runs[f"E2_{s}_{cap}_cost0_cap0"]
            cellruns = {
                (cost, lim): runs[f"E2_{s}_{cap}_cost{cost}_cap{lim}"]
                for cost, lim in itertools.product((0, 1), repeat=2)
            }
            outcomes = {}
            for outcome in (
                "active_return_ann",
                "catchup_ratio",
                "wte_risky_mean",
                "completion_aggregate",
                "cost_dollars",
            ):
                cells = {}
                for levels, r in cellruns.items():
                    if outcome == "active_return_ann":
                        y = 252 * (returns(r["nav"]) - returns(base["nav"])).mean()
                    elif outcome == "catchup_ratio":
                        y = point(r["nav"], base["nav"])["terminal_gap_ann_bps"] / 10000
                    elif outcome == "completion_aggregate":
                        y = r["metrics"]["orders"]["completion_aggregate"]
                    else:
                        y = r["metrics"][outcome]
                    cells[levels] = float(y)
                fac = factor(cells, 2)
                raw = cells[1, 1] - cells[1, 0] - cells[0, 1] + cells[0, 0]
                eps = 1e-12 + 1e-9 * max(abs(x) for x in cells.values())
                fac.update(
                    raw_interaction=raw,
                    numerical_zero_tolerance=eps,
                    numerically_zero=abs(raw) <= eps,
                    economic_materiality_5bps=(
                        abs(raw) >= 0.0005 if outcome == "active_return_ann" else None
                    ),
                    interpretation="primary H2c annual arithmetic active return"
                    if outcome == "active_return_ann"
                    else "descriptive only",
                )
                outcomes[outcome] = fac
            e2.append(
                {
                    "strategy": s,
                    "initial_capital_label": cap,
                    "reference_run_spec_id": base["id"],
                    "run_spec_ids": {
                        "".join(map(str, k)): r["id"] for k, r in cellruns.items()
                    },
                    "outcomes": outcomes,
                }
            )
    base = runs["E2b_cost0_min0_borrow1_fractional"]
    brows = {}
    for n, r in runs.items():
        c = reg[n]
        if c.experiment == "E2b":
            brows[
                (
                    int(c.cost.enabled),
                    int(c.min_trade_notional > 0),
                    int(not c.allow_borrowing),
                    int(c.units == "integer"),
                )
            ] = r
    bfac = {}
    for outcome in (
        "terminal_nav",
        "terminal_gap_bps",
        "active_return_ann",
        "sharpe_rf0",
        "cost_dollars",
    ):
        cells = {}
        for lv, r in brows.items():
            if outcome == "terminal_nav":
                y = r["nav"][-1]
            elif outcome == "terminal_gap_bps":
                y = point(r["nav"], base["nav"])["terminal_gap_bps"]
            elif outcome == "active_return_ann":
                y = point(r["nav"], base["nav"])["active_return_ann"]
            else:
                y = r["metrics"][outcome]
            cells[lv] = float(y)
        bfac[outcome] = factor(cells, 4)
    write(
        "factorials.json",
        {
            "e2": e2,
            "e2b": {
                "factors": [
                    "cost_enabled",
                    "minimum_1000",
                    "borrowing_forbidden",
                    "integer_units",
                ],
                "baseline_run_spec_id": base["id"],
                "run_spec_ids": {
                    "".join(map(str, k)): r["id"] for k, r in brows.items()
                },
                "outcomes": bfac,
            },
            "coefficient_convention": "beta uses -1/+1 coding, factorial effect=2*beta for nonempty subset; E2 four-cell difference=4*beta_(0,1)",
        },
    )
    write(
        "runtime_profile.json",
        {
            "scope": "Execution ledger intentionally not exported; local saved-result analysis only"
        },
    )
    manifest = {
        "code_commit": auth["code_commit"],
        "design_fingerprint": auth["design_fingerprint"],
        "scenario_runs": {n: r["id"] for n, r in runs.items()},
        "generated_by_sha256": sha(pathlib.Path(__file__)),
        "bootstrap": {
            "B": 2000,
            "mean_blocks": [20, 60],
            "seed": 20261119,
            "interval": "pointwise95%percentile",
        },
        "output_sha256": {
            p.name: sha(p)
            for p in A.iterdir()
            if p.is_file()
            and p.name not in ("analysis_manifest.json", "resources.json")
        },
        "resources": usage(),
    }
    write("analysis_manifest.json", manifest)
    write(
        "resources.json",
        usage()
        | {
            "status": "completed",
            "cpu_limit": 240,
            "process_cpu_limit_seconds": 240,
            "memory_monitor_mib": 1900,
        },
    )
    print("ANALYSIS COMPLETE", usage(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    W = args.runs_root.resolve()
    OUT = W
    A = args.out.resolve()
    if A.exists():
        parser.error("output must be new")
    A.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_CPU, (240, 240))
    start = time.monotonic()
    threading.Thread(target=watch, daemon=True).start()
    main()
