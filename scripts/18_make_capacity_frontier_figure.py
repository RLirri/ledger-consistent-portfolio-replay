from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# Keep consistent with your paper universe definitions (used elsewhere)
UNIVERSE = ["SPY","QQQ","IWM","XLF","XLK","XLE","XLY","XLI","XLV","XLP","XLU","TLT","IEF","GLD"]


# ---------- Style (match your accepted figures) ----------
def _set_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.8,
        "lines.markersize": 6.0,
        "savefig.dpi": 300,
    })


# ---------- Utilities ----------
def _fmt_millions(x: float, _pos: int) -> str:
    # x is dollars
    if x >= 1e9:
        return f"{x/1e9:.0f}B"
    return f"{x/1e6:.0f}M"


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find required column. Tried candidates:\n"
        f"  {candidates}\n"
        f"Available columns:\n  {list(df.columns)}"
    )


def _coerce_max_trade_label(v: float) -> str:
    # v is dollars
    if v >= 1e6:
        return f"${v/1e6:.0f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}k"
    return f"${v:.0f}"


def _style_map_for_max_trade(levels: List[float]) -> Dict[float, Dict]:
    """
    Consistent with your capacity binding figure:
      25k -> C0, 50k -> C1, 100k -> C2, 250k -> C3 (if present).
    Also grayscale-safe via linestyle/marker.
    """
    # Sort ascending to keep “tighter constraint” first
    levels_sorted = sorted(levels)

    # Default style cycle
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    linestyles = ["-", "--", ":", "-.", "-", "--"]
    markers = ["o", "s", "^", "D", "v", "P"]

    styles: Dict[float, Dict] = {}
    for i, lv in enumerate(levels_sorted):
        styles[lv] = dict(
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
    return styles


# ---------- Main plotting ----------
def plot_capacity_frontier(
    capacity_csv: str,
    out_dir: str,
    *,
    filename_stem: str = "fig_capacity_frontier_pub",
    title: str = "Capacity frontier (weekly rebalances)",
) -> None:
    _set_pub_style()
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(capacity_csv)

    # --- Column detection (robust across your pipeline variants) ---
    cap_col = _pick_col(df, [
        "initial_cash", "initial_capital", "capital", "aum", "AUM", "notional", "initial_notional"
    ])
    max_trade_col = _pick_col(df, [
        "max_trade_notional", "max_trade", "max_trade_dollars", "trade_cap", "max_trade_usd", "max_trade_$"
    ])
    sharpe_col = _pick_col(df, [
        "sharpe", "sharpe_net", "sharpe_executed", "net_sharpe", "sharpe_ratio"
    ])
    is_col = _pick_col(df, [
        "impl_shortfall_bps", "impl_shortfall_bp", "implementation_shortfall_bps",
        "is_bps", "implementation_shortfall", "implementation_shortfall_bp"
    ])

    _ensure_numeric(df, [cap_col, max_trade_col, sharpe_col, is_col])

    # Drop incomplete rows
    df = df.dropna(subset=[cap_col, max_trade_col, sharpe_col, is_col]).copy()

    # Ensure sensible types
    df[cap_col] = df[cap_col].astype(float)
    df[max_trade_col] = df[max_trade_col].astype(float)
    df[sharpe_col] = df[sharpe_col].astype(float)
    df[is_col] = df[is_col].astype(float)

    # Sort for clean lines
    df = df.sort_values([max_trade_col, cap_col])

    max_trade_levels = sorted(df[max_trade_col].unique())
    styles = _style_map_for_max_trade(max_trade_levels)

    # --- Layout ---
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.2, 4.0), gridspec_kw={"wspace": 0.18}
    )

    # --- Panel A: Sharpe vs capital ---
    for mt in max_trade_levels:
        sub = df[df[max_trade_col] == mt].sort_values(cap_col)
        axL.plot(
            sub[cap_col].values,
            sub[sharpe_col].values,
            label=f"Max trade {_coerce_max_trade_label(mt)}",
            **styles[mt],
        )

    axL.set_title("Risk-adjusted performance", pad=8)
    axL.set_xlabel("Initial capital")
    axL.set_ylabel("Sharpe ratio")
    axL.xaxis.set_major_formatter(FuncFormatter(_fmt_millions))
    axL.grid(True, axis="y", alpha=0.25, linewidth=1.0)

    # --- Panel B: IS vs capital ---
    for mt in max_trade_levels:
        sub = df[df[max_trade_col] == mt].sort_values(cap_col)
        axR.plot(
            sub[cap_col].values,
            sub[is_col].values,
            **styles[mt],
        )

    axR.set_title("Execution cost", pad=8)
    axR.set_xlabel("Initial capital")
    axR.set_ylabel("Implementation shortfall (bps)")
    axR.xaxis.set_major_formatter(FuncFormatter(_fmt_millions))
    axR.grid(True, axis="y", alpha=0.25, linewidth=1.0)

    # --- Shared legend (compact, paper style) ---
    handles, labels = axL.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=True,
        fancybox=True,
        bbox_to_anchor=(0.5, 1.10),
    )

    fig.suptitle(title, y=1.16)

    # Small disclosure note (optional but reviewer-friendly)
    fig.text(
        0.995, 0.02,
        "Lines grouped by max-trade constraint; marker/linestyle redundant for grayscale print.",
        ha="right", va="bottom", fontsize=9, color="0.35"
    )

    out_png = os.path.join(out_dir, f"{filename_stem}.png")
    out_pdf = os.path.join(out_dir, f"{filename_stem}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    out_dir = artifacts / "figures_pub"
    os.makedirs(out_dir, exist_ok=True)

    # Try a sensible default filename (edit if your project uses a different one)
    # Common patterns in your repo: sensitivity_scorecards_weekly.csv, etc.
    default_candidates = [
        artifacts / "capacity_stress_weekly_with_te_is.csv",
        artifacts / "capacity_stress_weekly.csv",
        artifacts / "capacity_stress.csv",
        artifacts / "capacity_grid.csv",
        artifacts / "capacity_results.csv",
    ]
    capacity_csv = None
    for p in default_candidates:
        if p.exists():
            capacity_csv = p
            break
    if capacity_csv is None:
        raise FileNotFoundError(
            "Could not find a capacity CSV in artifacts. Tried:\n"
            + "\n".join([str(x) for x in default_candidates])
            + "\n\nPlease set capacity_csv to your actual file path."
        )

    plot_capacity_frontier(
        capacity_csv=str(capacity_csv),
        out_dir=str(out_dir),
        filename_stem="fig_capacity_frontier_pub",
        title="Capacity frontier (weekly rebalances)",
    )
    print("Wrote capacity frontier figure to", out_dir)
