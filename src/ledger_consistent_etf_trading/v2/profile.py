"""Runtime profile of one scenario by phase (E6 support). CPU and wall seconds per phase over
repetitions, event counts, and peak resident memory in MiB. Peak RSS is the process-lifetime
high-water mark at the time of reading, not the incremental memory of a phase."""
from __future__ import annotations

import resource
import statistics
import sys
import time

from . import checks, engine, metrics
from .data import build_features


def peak_rss_mib() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1048576 if sys.platform == "darwin" else r / 1024


def profile(mkt, cfg, repetitions: int = 3, production: bool = False) -> dict:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    phases = {k: {"cpu": [], "wall": []} for k in ("features", "replay", "validation", "metrics", "total")}
    events = None
    for _ in range(repetitions):
        c0, w0 = time.process_time(), time.perf_counter()

        def lap(name, c1, w1):
            phases[name]["cpu"].append(time.process_time() - c1); phases[name]["wall"].append(time.perf_counter() - w1)

        c, w = time.process_time(), time.perf_counter()
        feats = build_features(mkt, cfg.features, cfg.adv_estimator); lap("features", c, w)
        c, w = time.process_time(), time.perf_counter()
        res = engine.run(mkt, cfg, feats=feats); lap("replay", c, w)
        c, w = time.process_time(), time.perf_counter()
        rep = checks.validate_result(res, mkt, feats, production=production); lap("validation", c, w)
        c, w = time.process_time(), time.perf_counter()
        metrics.summarize(res, mkt.close); lap("metrics", c, w)
        lap("total", c0, w0)
        events = {"sessions": int(len(res.daily)), "assets": len(res.tickers), "order_rows": int(len(res.orders)), "parents": int(len(res.parents)),
                  "demand_rows": int(len(res.demand)), "validation_checks": len(rep.checks), "valid": rep.ok}
    out = {"scenario_id": cfg.scenario_id, "repetitions": repetitions, "events": events, "peak_rss_mib_process_high_water": peak_rss_mib(), "phases": {}}
    for name, d in phases.items():
        out["phases"][name] = {f"{k}_seconds_{s}": fn(v) for k, v in d.items() for s, fn in (("median", statistics.median), ("min", min), ("max", max))}
    return out
