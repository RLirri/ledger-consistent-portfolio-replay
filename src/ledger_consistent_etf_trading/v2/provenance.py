from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import SPEC_VERSION


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_identity(repo: Path) -> dict:
    git = lambda *a: subprocess.check_output(["git", "-C", str(repo), *a], text=True)
    diff = git("diff", "HEAD")
    untracked = git("ls-files", "--others", "--exclude-standard").split()
    h = hashlib.sha256(diff.encode())
    for u in sorted(untracked):
        h.update(u.encode()); h.update((repo / u).read_bytes())
    return {"commit": git("rev-parse", "HEAD").strip(), "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(diff.strip() or untracked), "dirty_sha256": h.hexdigest()}


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def write_run(out_root: Path, res, summary: dict, mkt, repo: Path, timings: dict, authorization: dict) -> Path:
    """Single-run writer kept for the Sprint 3 pilot format. Batch execution (v2/batch.py) is the
    production path: it validates, reads back and publishes atomically under a run-spec identity."""
    out = Path(out_root) / f"{res.cfg.name}-{res.cfg.scenario_id}"
    if out.exists():
        raise FileExistsError(f"run directory exists; outputs are immutable: {out}")
    out.mkdir(parents=True)
    res.daily.to_csv(out / "daily.csv")
    res.orders.to_csv(out / "orders.csv", index=False)
    res.parents.to_csv(out / "parents.csv", index=False)
    res.demand.to_csv(out / "demand.csv", index=False)
    pd.DataFrame(res.positions, index=res.dates, columns=res.tickers).to_csv(out / "positions.csv")
    pd.DataFrame(res.target_eff, index=res.dates, columns=res.tickers).to_csv(out / "target_weights.csv")
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    files = {p.name: _sha(p) for p in sorted(out.iterdir())}
    manifest = {"spec_version": SPEC_VERSION, "scenario_id": res.cfg.scenario_id, "config": res.cfg.to_dict(),
                "information_class": res.info["information"], "run_info": json.loads(json.dumps(res.info)), "code": code_identity(repo), "data_input_sha256": mkt.input_hashes,
                "environment": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__, "platform": platform.platform()},
                "timings_seconds": timings, "peak_rss_mb": peak_rss_mb(), "authorization": authorization,
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "output_sha256": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for p in out.iterdir():
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return out
