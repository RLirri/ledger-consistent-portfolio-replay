"""Render saved publication evidence into a new local directory; never replay."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", default="F2")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    root, diagnostics, out = (
        args.evidence_root.resolve(),
        args.diagnostics.resolve(),
        args.out.resolve(),
    )
    if out.exists():
        parser.error("output must be a new directory; saved inputs are immutable")
    # Explicitly named, locally held saved evidence only. No data fetching.
    required = [
        root / "experiments/v2_grid/analysis/scorecard.json",
        root / "experiments/v2_grid/analysis/contrasts.json",
        root / "experiments/v2_grid/analysis/factorials.json",
        diagnostics / "manifests/evidence_verification.json",
    ]
    if any(not p.is_file() for p in required):
        parser.error("required local saved-evidence inputs are missing")
    out.mkdir(parents=True)
    for name in ["data", "figures", "tables", "manifests", "review"]:
        (out / name).mkdir()
    for source in (diagnostics / "data").iterdir():
        if source.is_file() and source.suffix in {".csv", ".json"}:
            shutil.copyfile(source, out / "data" / source.name)
    shutil.copyfile(
        diagnostics / "manifests/evidence_verification.json",
        out / "manifests/evidence_verification.json",
    )
    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        repo / "figures/paper_v2_ieee.mplstyle", out / "paper_v2_ieee.mplstyle"
    )
    env = dict(os.environ, REPLAY_EVIDENCE_ROOT=str(root), REPLAY_FIGURE_ROOT=str(out))
    if args.font:
        env["REPLAY_FONT_PATH"] = str(args.font.resolve())
    command = [
        sys.executable,
        "-B",
        str(repo / "scripts/publication_figures.py"),
        "--figure",
        args.figure,
    ]
    subprocess.run(command, env=env, check=True)
    if args.tables:
        subprocess.run(
            [sys.executable, "-B", str(repo / "scripts/publication_tables.py")],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
