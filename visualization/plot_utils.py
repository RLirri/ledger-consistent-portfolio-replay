from __future__ import annotations

from pathlib import Path
from matplotlib.figure import Figure


def save_fig(fig: Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save both for paper + quick preview
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
