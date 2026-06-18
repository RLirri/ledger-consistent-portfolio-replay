# scripts/16_make_implementation_shortfall_figures.py
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter


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


def plot_implementation_shortfall_time(
    fills_weekly_csv: str,
    out_dir: str,
    *,
    filename_stem: str = "fig_implementation_shortfall_time",
    rolling_weeks: int = 13,
) -> None:
    """
    Paper figure: implementation shortfall over time (weekly rebalances).

    IS (bps) = total_cost_$ / gross traded notional * 1e4
    Uses zoomed y-scale for readability + long-run median reference line.
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

    roll = rolling_weeks
    is_med = g["is_bps"].rolling(roll, min_periods=1).median()
    is_q10 = g["is_bps"].rolling(roll, min_periods=1).quantile(0.1)
    is_q90 = g["is_bps"].rolling(roll, min_periods=1).quantile(0.9)

    not_m = (g["gross_notional_usd"] / 1e6).astype(float)
    not_med = not_m.rolling(roll, min_periods=1).median()
    not_q10 = not_m.rolling(roll, min_periods=1).quantile(0.1)
    not_q90 = not_m.rolling(roll, min_periods=1).quantile(0.9)

    x = g.index.to_pydatetime()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1]}
    )

    # Panel A: IS (raw faint + band + median)
    ax1.plot(x, g["is_bps"].values, color="tab:red", alpha=0.15, linewidth=1.4)
    ax1.fill_between(x, is_q10.values, is_q90.values, alpha=0.18, color="tab:red", linewidth=0)
    ax1.plot(x, is_med.values, color="tab:red", label=f"IS ({roll}w median)")

    # Zoom y-limits around actual variation
    ymin = float(is_q10.min())
    ymax = float(is_q90.max())
    pad = (ymax - ymin) * 0.25 if ymax > ymin else 0.01
    ax1.set_ylim(ymin - pad, ymax + pad)

    # Long-run median reference line (more meaningful than 0 here)
    long_med = float(g["is_bps"].median())
    ax1.axhline(long_med, color="0.35", linestyle="--", linewidth=1.2)
    ax1.text(
        0.995, 0.06, f"Long-run median: {long_med:.3f} bps",
        transform=ax1.transAxes, ha="right", va="bottom",
        fontsize=9, color="0.35"
    )

    ax1.set_ylabel("IS (bps of gross traded)")
    ax1.set_title("Implementation shortfall over time (weekly rebalances)", pad=12)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=1)
    ax1.legend(loc="upper left", frameon=True, fancybox=True)

    # Panel B: gross traded notional (cap outlier)
    ax2.fill_between(x, not_q10.values, not_q90.values, alpha=0.12, color="0.2", linewidth=0)
    ax2.plot(x, not_med.values, color="0.2", linewidth=2.6,
             label=f"Gross traded notional ({roll}w median)")

    p99 = float(not_m.quantile(0.99))
    ax2.set_ylim(0, max(0.05, p99 * 1.15))
    ax2.set_ylabel("Notional ($M)")
    ax2.grid(True, axis="y", alpha=0.25, linewidth=1)
    ax2.legend(loc="upper left", frameon=True, fancybox=True)

    ax2.text(
        0.995, 0.05, "Y-axis capped at 99th pct.",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=9, color="0.35"
    )

    ax2.xaxis.set_major_locator(YearLocator(2))
    ax2.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax2.set_xlabel("Date")

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

    plot_implementation_shortfall_time(
        fills_weekly_csv=os.path.join(artifacts, "fills_equal_weight_weekly.csv"),
        out_dir=out_dir,
        filename_stem="fig_implementation_shortfall_time",
        rolling_weeks=13,
    )
    print("Wrote IS figure to", out_dir)
