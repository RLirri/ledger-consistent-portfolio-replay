from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ledger_consistent_etf_trading.utils.paths import ARTIFACTS_DIR

def set_pub_style():
    mpl.rcParams.update({
        # fonts
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # lines
        "lines.linewidth": 2.25,
        "lines.markersize": 7,

        # axes aesthetics
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",

        # figure
        "figure.figsize": (8.2, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig: plt.Figure, name: str):
    fig.savefig(ARTIFACTS_DIR / f"{name}.pdf")
    fig.savefig(ARTIFACTS_DIR / f"{name}.png")
    plt.close(fig)


def money_fmt(x, _):
    if x >= 1_000_000:
        return f"{x/1_000_000:g}M"
    if x >= 1_000:
        return f"{x/1_000:g}k"
    return f"{x:g}"


def cap_label(x: float) -> str:
    if x >= 1_000_000:
        return f"${x/1_000_000:g}M"
    if x >= 1_000:
        return f"${x/1_000:g}k"
    return f"${x:g}"


def main():
    set_pub_style()

    df = pd.read_csv(ARTIFACTS_DIR / "capacity_stress_weekly_with_te_is.csv").copy()

    # Keep one mp to avoid duplicates
    if "max_participation" in df.columns:
        df = df[df["max_participation"] == 0.01].copy()

    # Ensure numeric
    for c in ["initial_cash", "max_trade_notional", "pct_clipped", "avg_weight_te", "impl_shortfall_bps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


    # Capacity curve
    fig, ax = plt.subplots()

    for maxN, g in df.groupby("max_trade_notional"):
        g = g.sort_values("initial_cash")
        ax.plot(g["initial_cash"], g["pct_clipped"], marker="o", label=f"Max trade {cap_label(maxN)}")

    ax.set_title("Capacity constraint binding vs capital")
    ax.set_xlabel("Initial capital")
    ax.set_ylabel("Clipping rate")

    ax.xaxis.set_major_formatter(FuncFormatter(money_fmt))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.legend(frameon=True, title="Notional cap", loc="upper left")
    save_fig(fig, "fig_pub_capacity_curve")


    # 2-panel scatter (TE & IS vs clip)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))


    caps = sorted(df["max_trade_notional"].unique())
    cap_to_color = {c: None for c in caps}  # let matplotlib choose cycle

    # Panel A: Tracking error vs clipping
    ax = axes[0]
    for maxN, g in df.groupby("max_trade_notional"):
        ax.scatter(g["pct_clipped"], g["avg_weight_te"], alpha=0.85, label=cap_label(maxN))
    ax.set_title("Portfolio distortion vs capacity binding")
    ax.set_xlabel("Clipping rate")
    ax.set_ylabel("Average weight tracking error")

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Panel B: Implementation shortfall vs clipping
    ax = axes[1]
    for maxN, g in df.groupby("max_trade_notional"):
        ax.scatter(g["pct_clipped"], g["impl_shortfall_bps"], alpha=0.85, label=cap_label(maxN))
    ax.set_title("Implementation shortfall vs capacity binding")
    ax.set_xlabel("Clipping rate")
    ax.set_ylabel("Implementation shortfall (bps)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # One legend for both
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Max trade", ncol=4, loc="upper center", frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    save_fig(fig, "fig_pub_te_is_vs_clipping")

    print("Saved publication figures (PDF + PNG) to artifacts/:")
    print(" - fig_pub_capacity_curve.(pdf|png)")
    print(" - fig_pub_te_is_vs_clipping.(pdf|png)")


if __name__ == "__main__":
    main()
