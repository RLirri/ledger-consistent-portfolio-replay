# visualization/plot_style.py
from __future__ import annotations

import matplotlib as mpl


def set_pub_style() -> None:
    mpl.rcParams.update({
        # Resolution / saving
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # Fonts
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,

        # Axes look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,

        # Lines
        "lines.linewidth": 2.2,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fancybox": True,
        "legend.borderpad": 0.5,
    })
