# scripts/14_make_sensitivity_figures.py
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def _set_pub_style_capacity_consistent() -> None:
    """
    Match your capacity_binding figure style:
    - Serif font
    - Larger text
    - Thicker lines and markers
    - Default tab colors
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "lines.markersize": 8,
        "savefig.dpi": 300,
    })


def plot_sharpe_sensitivity(
    sensitivity_csv: str,
    out_dir: str,
    *,
    filename_stem: str = "fig_sensitivity_sharpe_cost_model",
) -> None:
    """
    Paper figure: Sharpe sensitivity to transaction cost assumptions.
    Small multiples by max_participation; colors encode half-spread levels.
    Style matches capacity_binding figure (tab colors + serif + thicker lines).
    """
    _set_pub_style_capacity_consistent()
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(sensitivity_csv)

    # ---- Validate schema (project safety check) ----
    required = {"max_participation", "half_spread_bps", "slippage_a_bps", "sharpe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sensitivity_scorecards_weekly.csv missing columns: {sorted(missing)}")

    # ---- Types ----
    df["max_participation"] = df["max_participation"].astype(float)
    df["half_spread_bps"] = df["half_spread_bps"].astype(float)
    df["slippage_a_bps"] = df["slippage_a_bps"].astype(float)

    max_parts = sorted(df["max_participation"].unique())
    spread_levels = sorted(df["half_spread_bps"].unique())
    slip_levels = sorted(df["slippage_a_bps"].unique())

    # ---- Match your paper’s color convention (tab colors) ----
    tab_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    color_map = {hs: tab_colors[i % len(tab_colors)] for i, hs in enumerate(spread_levels)}

    fig, axes = plt.subplots(1, len(max_parts), figsize=(11, 3.6), sharey=True)
    if len(max_parts) == 1:
        axes = [axes]

    for ax, mp in zip(axes, max_parts):
        sub = df[df["max_participation"] == mp]

        for hs in spread_levels:
            ssub = sub[sub["half_spread_bps"] == hs].sort_values("slippage_a_bps")
            ax.plot(
                ssub["slippage_a_bps"],
                ssub["sharpe"],
                label=f"Half-spread {hs:.1f} bps",
                color=color_map[hs],
                marker="o",
                markerfacecolor="white",
                markeredgewidth=1.2,
            )

        ax.set_title(f"Max participation {mp:.0%}", pad=10)
        ax.set_xlabel(r"Slippage coefficient $a$ (bps)")
        ax.set_xticks(slip_levels)
        ax.grid(True, axis="y", alpha=0.25, linewidth=1)

    axes[0].set_ylabel("Sharpe ratio")

    # Shared legend like your other figure (with frame)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=min(3, len(spread_levels)),
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.12),
    )

    fig.suptitle("Sensitivity of Sharpe to transaction-cost model", y=1.22, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(out_dir, f"{filename_stem}.png")
    out_pdf = os.path.join(out_dir, f"{filename_stem}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))  # repo root
    artifacts = os.path.join(base, "artifacts")
    out_dir = os.path.join(artifacts, "figures_pub")

    plot_sharpe_sensitivity(
        sensitivity_csv=os.path.join(artifacts, "sensitivity_scorecards_weekly.csv"),
        out_dir=out_dir,
        filename_stem="fig_sensitivity_sharpe_cost_model",
    )
    print("Wrote sensitivity figure to", out_dir)
