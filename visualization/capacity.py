from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter, FuncFormatter

def _money_tick(x, _):
    if x >= 1_000_000:
        return f"{x/1_000_000:g}M"
    if x >= 1_000:
        return f"{x/1_000:g}k"
    return f"{x:g}"

def _cap_label(x: float) -> str:
    if x >= 1_000_000:
        return f"${x/1_000_000:g}M"
    if x >= 1_000:
        return f"${x/1_000:g}k"
    return f"${x:g}"

def plot_capacity_curve(ax: Axes, df: pd.DataFrame) -> None:
    """
    df columns required:
    initial_cash, max_trade_notional, pct_clipped
    """
    for maxN, g in df.groupby("max_trade_notional"):
        g = g.sort_values("initial_cash")
        ax.plot(g["initial_cash"], g["pct_clipped"], marker="o",
                label=f"Max trade {_cap_label(float(maxN))}")

    ax.set_title("Capacity constraint binding vs capital")
    ax.set_xlabel("Initial capital")
    ax.set_ylabel("Clipping rate")
    ax.xaxis.set_major_formatter(FuncFormatter(_money_tick))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

def plot_te_is_vs_clipping(ax_te: Axes, ax_is: Axes, df: pd.DataFrame) -> None:
    """
    df columns required:
    pct_clipped, avg_weight_te, impl_shortfall_bps, max_trade_notional
    """
    # scatter colored by max_trade_notional (legend becomes "Max trade")
    for maxN, g in df.groupby("max_trade_notional"):
        label = _cap_label(float(maxN))
        ax_te.scatter(g["pct_clipped"], g["avg_weight_te"], alpha=0.9, label=label)
        ax_is.scatter(g["pct_clipped"], g["impl_shortfall_bps"], alpha=0.9, label=label)

    ax_te.set_title("Portfolio distortion vs capacity binding")
    ax_te.set_xlabel("Clipping rate")
    ax_te.set_ylabel("Average weight tracking error")
    ax_te.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax_is.set_title("Implementation shortfall vs capacity binding")
    ax_is.set_xlabel("Clipping rate")
    ax_is.set_ylabel("Implementation shortfall (bps)")
    ax_is.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

def plot_capacity_heatmap(ax: Axes, df: pd.DataFrame, value_col: str = "pct_clipped") -> None:
    """
    Publication-grade heatmap: capacity regime map.
    value_col: pct_clipped OR avg_weight_te OR impl_shortfall_bps
    """
    pivot = df.pivot_table(
        index="max_trade_notional",
        columns="initial_cash",
        values=value_col,
        aggfunc="mean"
    ).sort_index().sort_index(axis=1)

    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_title(f"Capacity regime map ({value_col})")
    ax.set_xlabel("Initial capital")
    ax.set_ylabel("Max trade notional")

    # ticks
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([_money_tick(x, None) for x in pivot.columns], rotation=0)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([_cap_label(float(x)) for x in pivot.index])

    return im
