from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.transforms import blended_transform_factory


@dataclass(frozen=True)
class RegimeBand:
    start: str  # "YYYY-MM-DD"
    end: str
    label: str


DEFAULT_REGIMES = [
    RegimeBand("2018-10-01", "2019-01-31", "Q4-2018"),
    RegimeBand("2020-02-15", "2020-05-31", "COVID"),
    RegimeBand("2022-01-01", "2022-12-31", "Rates"),
]


def _to_dt_index(s: pd.Series) -> pd.Series:
    s = s.dropna()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # make tz-naive (matplotlib dates like this)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def _format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))
    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)


def shade_regimes(
    ax: plt.Axes,
    regimes: Iterable[RegimeBand],
    *,
    alpha: float = 0.08,
    label_y_levels: Optional[list[float]] = None,
) -> None:
    """
    Very light vertical shading + non-overlapping labels.
    Labels are placed in axes-y coordinates (so they don't collide with data),
    but x is in date (data) coordinates.
    """
    regimes = list(regimes)
    if not regimes:
        return

    if label_y_levels is None:
        # Stagger labels (top of axes). More levels -> less overlap.
        label_y_levels = [0.98, 0.92, 0.86, 0.80]

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    for i, r in enumerate(regimes):
        s = pd.to_datetime(r.start)
        e = pd.to_datetime(r.end)

        ax.axvspan(s, e, alpha=alpha, zorder=0)

        # label at the band midpoint; staggered vertically
        mid = s + (e - s) * 0.5
        y = label_y_levels[i % len(label_y_levels)]

        ax.text(
            mid,
            y,
            r.label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=11,
            color="black",
            alpha=0.65,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.60),
            zorder=5,
        )


def drawdown(eq: pd.Series) -> pd.Series:
    eq = _to_dt_index(eq)
    if eq.empty:
        return eq
    peak = eq.cummax()
    return eq / peak - 1.0


def plot_drawdown_exec_vs_ideal(
    executed_eq: pd.Series,
    ideal_eq: pd.Series,
    outpath_dir,
    *,
    title: str = "Drawdown (out-of-sample)",
    regimes: Optional[Iterable[RegimeBand]] = DEFAULT_REGIMES,
) -> None:
    executed_eq = _to_dt_index(executed_eq)
    ideal_eq = _to_dt_index(ideal_eq)

    # align for fair visual comparison
    idx = executed_eq.index.union(ideal_eq.index)
    executed_eq = executed_eq.reindex(idx).ffill()
    ideal_eq = ideal_eq.reindex(idx).ffill()

    dd_exec = drawdown(executed_eq)
    dd_ideal = drawdown(ideal_eq)

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.7, zorder=1)

    # Plot ideal first (behind), then executed (on top)
    ax.plot(
        dd_ideal.index, dd_ideal.values,
        label="Ideal (frictionless)",
        linestyle="--",
        linewidth=1.9,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        dd_exec.index, dd_exec.values,
        label="Executed",
        linestyle="-",
        linewidth=2.4,
        alpha=0.95,
        zorder=3,
    )

    # Optional: max-DD guides (subtle)
    mdd_exec = float(dd_exec.min())
    mdd_ideal = float(dd_ideal.min())
    ax.axhline(mdd_exec, linestyle=":", linewidth=1.0, alpha=0.55, zorder=1)
    ax.axhline(mdd_ideal, linestyle=":", linewidth=1.0, alpha=0.40, zorder=1)

    ax.set_title(title, pad=10)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    if regimes:
        shade_regimes(ax, regimes, alpha=0.08)

    _format_date_axis(ax)
    ax.legend(loc="lower left", ncol=1)
    fig.tight_layout()

    from .plot_utils import save_fig
    save_fig(fig, outpath_dir, "fig_drawdown_exec_vs_ideal")
    plt.close(fig)


def plot_equity_log_exec_vs_ideal(
    executed_eq: pd.Series,
    ideal_eq: pd.Series,
    benchmark_eq: Optional[pd.Series],
    outpath_dir,
    *,
    title: str = "Cumulative performance (out-of-sample)",
    regimes: Optional[Iterable[RegimeBand]] = DEFAULT_REGIMES,
) -> None:
    executed_eq = _to_dt_index(executed_eq)
    ideal_eq = _to_dt_index(ideal_eq)
    bench = _to_dt_index(benchmark_eq) if benchmark_eq is not None else None

    # Normalize to 1.0
    def _norm(s: pd.Series) -> pd.Series:
        s = s.dropna()
        return s / float(s.iloc[0]) if not s.empty else s

    executed_eq = _norm(executed_eq)
    ideal_eq = _norm(ideal_eq)
    if bench is not None and not bench.empty:
        bench = _norm(bench)

    # align
    idx = executed_eq.index.union(ideal_eq.index)
    if bench is not None:
        idx = idx.union(bench.index)
    executed_eq = executed_eq.reindex(idx).ffill()
    ideal_eq = ideal_eq.reindex(idx).ffill()
    if bench is not None:
        bench = bench.reindex(idx).ffill()

    fig, ax = plt.subplots(figsize=(10.8, 4.9))

    # Ideal behind; executed on top. Benchmark distinct + slightly thicker.
    if bench is not None:
        ax.plot(
            bench.index, np.log(bench.values),
            label="Benchmark (SPY)",
            linewidth=2.3,
            alpha=0.95,
            zorder=2,
        )

    ax.plot(
        ideal_eq.index, np.log(ideal_eq.values),
        label="Ideal (frictionless)",
        linestyle="--",
        linewidth=1.9,
        alpha=0.85,
        zorder=3,
    )
    ax.plot(
        executed_eq.index, np.log(executed_eq.values),
        label="Executed",
        linestyle="-",
        linewidth=2.4,
        alpha=0.95,
        zorder=4,
    )

    ax.set_title(title, pad=10)
    ax.set_ylabel("log Normalized equity")

    if regimes:
        shade_regimes(ax, regimes, alpha=0.08)

    _format_date_axis(ax)
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()

    from .plot_utils import save_fig
    save_fig(fig, outpath_dir, "fig_equity_log_exec_vs_ideal")
    plt.close(fig)


def plot_tracking_error_over_time(
    te_series: pd.Series,
    outpath_dir,
    *,
    title: str = "Tracking error over time",
    smooth: int = 63,
    regimes: Optional[Iterable[RegimeBand]] = DEFAULT_REGIMES,
    add_refs: bool = True,
) -> None:
    te = _to_dt_index(te_series)
    if te.empty:
        return

    if smooth and smooth > 1:
        te = te.rolling(smooth).mean()

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.plot(te.index, te.values, label=f"Rolling TE ({smooth}d mean)", linewidth=2.6, alpha=0.95)

    if add_refs:
        # Reference bands: median + (median +/- IQR/2) style guides (robust)
        med = float(te.median())
        q25 = float(te.quantile(0.25))
        q75 = float(te.quantile(0.75))
        ax.axhline(med, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.axhline(q25, linestyle="--", linewidth=1.0, alpha=0.35)
        ax.axhline(q75, linestyle="--", linewidth=1.0, alpha=0.35)

    ax.set_title(title, pad=10)
    ax.set_ylabel("Weight tracking error")

    if regimes:
        shade_regimes(ax, regimes, alpha=0.06)

    _format_date_axis(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()

    from .plot_utils import save_fig
    save_fig(fig, outpath_dir, "fig_te_time")
    plt.close(fig)
