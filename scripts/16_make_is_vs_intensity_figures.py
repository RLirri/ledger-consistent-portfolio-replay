from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _set_style_capacity_consistent() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "savefig.dpi": 300,
    })


def plot_is_vs_trading_intensity(
    fills_weekly_csv: str,
    out_dir: str,
    *,
    filename_stem: str = "fig_is_vs_trading_intensity",
    nbins: int = 10,
    regime_cut: str = "2020-01-01",
) -> None:
    """
    Paper figure: Implementation shortfall vs gross traded notional (weekly).

    IS (bps) = total_cost_$ / gross traded notional * 1e4

    Visual encoding:
      - points: weekly observations (faint), marker shape separates pre/post-2020
      - line + band: binned median + IQR band (main inference layer)
      - x-axis capped at 99th percentile for readability
    """
    _set_style_capacity_consistent()
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(fills_weekly_csv)
    required = {"date", "notional_$", "total_cost_$"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df["gross_notional_$"] = df["notional_$"].abs()

    g = df.groupby("date", as_index=True).agg(
        gross_notional_usd=("gross_notional_$", "sum"),
        is_usd=("total_cost_$", "sum"),
    )

    g["is_bps"] = (g["is_usd"] / g["gross_notional_usd"] * 1e4).astype(float)
    g["notional_m"] = (g["gross_notional_usd"] / 1e6).astype(float)

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["is_bps", "notional_m"])
    g = g[g["notional_m"] > 0]

    cut = pd.Timestamp(regime_cut)
    pre = g[g.index < cut]
    post = g[g.index >= cut]

    bins = pd.qcut(g["notional_m"], q=nbins, duplicates="drop")
    binned = g.groupby(bins).agg(
        x_med=("notional_m", "median"),
        y_med=("is_bps", "median"),
        y_q25=("is_bps", lambda s: s.quantile(0.25)),
        y_q75=("is_bps", lambda s: s.quantile(0.75)),
        n=("is_bps", "size"),
    ).sort_values("x_med")

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.6))

    # Scatter: faint texture
    ax.scatter(pre["notional_m"], pre["is_bps"], s=18, marker="o",
               color="tab:red", alpha=0.18, linewidths=0)
    ax.scatter(post["notional_m"], post["is_bps"], s=18, marker="^",
               color="tab:red", alpha=0.18, linewidths=0)

    # Binned median + IQR band (inference layer)
    ax.fill_between(
        binned["x_med"].values,
        binned["y_q25"].values,
        binned["y_q75"].values,
        color="tab:red", alpha=0.12, linewidth=0
    )
    ax.plot(
        binned["x_med"].values,
        binned["y_med"].values,
        color="tab:red", marker="o",
        markerfacecolor="white", markeredgewidth=1.0
    )

    # Reference: long-run median
    long_med = float(g["is_bps"].median())
    ax.axhline(long_med, color="0.35", linestyle="--", linewidth=1.2)
    ax.text(
        0.995, 0.06, f"Long-run median: {long_med:.3f} bps",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="0.35"
    )

    # Cap x-axis to avoid early outliers dominating
    x99 = float(g["notional_m"].quantile(0.99))
    ax.set_xlim(0, x99 * 1.05)
    ax.text(
        0.995, 0.02, "X-axis capped at 99th pct.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="0.35"
    )

    ax.set_title("Implementation shortfall vs trading intensity (weekly)")
    ax.set_xlabel("Gross traded notional ($M) per rebalance week")
    ax.set_ylabel("IS (bps of gross traded)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=1)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="tab:red", alpha=0.35, markersize=6,
               label="Pre-2020 (weekly points)"),
        Line2D([0], [0], marker="^", color="none",
               markerfacecolor="tab:red", alpha=0.35, markersize=6,
               label="2020+ (weekly points)"),
        Line2D([0], [0], color="tab:red", linewidth=3.0,
               label="Binned median (IQR band)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fancybox=True)

    fig.tight_layout()

    out_png = os.path.join(out_dir, f"{filename_stem}.png")
    out_pdf = os.path.join(out_dir, f"{filename_stem}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    artifacts = os.path.join(base, "artifacts")
    out_dir = os.path.join(artifacts, "figures_pub")

    plot_is_vs_trading_intensity(
        fills_weekly_csv=os.path.join(artifacts, "fills_equal_weight_weekly.csv"),
        out_dir=out_dir,
        filename_stem="fig_is_vs_trading_intensity",
        nbins=10,
        regime_cut="2020-01-01",
    )
    print("Wrote IS vs intensity figure to", out_dir)
