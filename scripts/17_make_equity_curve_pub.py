from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter


# Universe used to build the equal-weight benchmark (daily rebalanced)
UNIVERSE = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLY", "XLI", "XLV", "XLP", "XLU", "TLT", "IEF", "GLD"]


def _set_pub_style() -> None:
    """Publication defaults consistent with your accepted figures."""
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
        "savefig.dpi": 300,
    })


def _normalize(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(float)
    if len(s) == 0:
        raise ValueError("Series is empty after dropping NaNs.")
    return s / float(s.iloc[0])


def _load_clean_close(parquet_path: Path) -> pd.Series:
    """Load close/adj_close from *_1d_clean.parquet produced by your pipeline."""
    df = pd.read_parquet(parquet_path)

    # Date handling (robust to either date column or datetime index)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
        df = df.set_index("date")

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)

    for col in ["adj_close", "adjClose", "Adj Close", "close", "Close"]:
        if col in df.columns:
            s = df[col].astype(float).sort_index().dropna()
            if len(s) == 0:
                raise ValueError(f"{parquet_path.name}: close series is empty after dropna().")
            return s

    raise ValueError(f"No close/adj_close column found in {parquet_path.name}")


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> tuple[pd.Series, float, float]:
    """Clip extremes to stabilize the y-scale for paper readability."""
    s2 = s.dropna()
    if len(s2) == 0:
        return s.copy(), float("nan"), float("nan")
    ql, qh = s2.quantile([lo, hi])
    return s.clip(lower=float(ql), upper=float(qh)), float(ql), float(qh)


def _build_equal_weight_benchmark(
    processed_dir: Path,
    dates: pd.DatetimeIndex,
    universe: list[str],
) -> tuple[pd.Series, int]:
    """
    Daily rebalanced equal-weight benchmark of available universe assets.
    Returns (benchmark_equity_curve, n_assets_used).
    """
    closes: dict[str, pd.Series] = {}
    for sym in universe:
        p = processed_dir / f"{sym}_1d_clean.parquet"
        if p.exists():
            closes[sym] = _load_clean_close(p)

    if len(closes) == 0:
        raise ValueError("No universe parquet files found to build the benchmark.")

    # Align to strategy dates with forward-fill (benchmark should exist for all strategy dates)
    px = pd.DataFrame({s: closes[s].reindex(dates, method="ffill") for s in closes}).dropna(how="all")
    px = px.reindex(dates, method="ffill")

    # Daily equal-weight rebalanced: mean of constituent daily returns
    ret = px.pct_change().fillna(0.0)
    bench_ret = ret.mean(axis=1)
    benchmark = (1.0 + bench_ret).cumprod()

    return benchmark, len(closes)


def plot_cumperf_two_panel_pub(
    equity_csv: str,
    processed_dir: str,
    out_dir: str,
    *,
    filename_stem: str = "fig_cumperf_two_panel_pub",
    smooth_window_days: int = 65,   # ~13 weeks trading days
) -> None:
    """
    Paper figure (publication-quality, consistent palette):

    Panel A: Cumulative log return (Executed vs Ideal vs Benchmark)
      - Executed: blue (C0), solid
      - Ideal: orange (C1), dashed
      - Benchmark: green (C2), dotted

    Panel B: Gaps in bps (smoothed + winsorized), two y-axes
      - Relative performance vs benchmark: green dashed
      - Friction drag vs ideal: orange solid

    Notes:
      - Bottom panel uses rolling median for robustness and winsorization to avoid a single spike
        forcing an unreadable scale. This is disclosed in-figure.
    """
    _set_pub_style()
    os.makedirs(out_dir, exist_ok=True)

    # --- Load strategy series from your artifacts ---
    eq = pd.read_csv(equity_csv)
    if "date" not in eq.columns:
        raise ValueError("equity_csv must include a 'date' column.")
    eq["date"] = pd.to_datetime(eq["date"], utc=True).dt.tz_convert(None)
    eq = eq.sort_values("date").set_index("date")

    required = {"equity", "realized_costs"}
    missing = required - set(eq.columns)
    if missing:
        raise ValueError(f"Missing columns in equity CSV: {sorted(missing)}")

    executed = eq["equity"].astype(float)
    ideal = eq["equity"].astype(float) + eq["realized_costs"].astype(float)

    # --- Build EW universe benchmark from your processed parquet files ---
    processed = Path(processed_dir)
    dates = executed.index
    benchmark, n_assets_used = _build_equal_weight_benchmark(processed, dates, UNIVERSE)

    # --- Normalize and compute log curves (paper-friendly y-axis) ---
    E = _normalize(executed)
    I = _normalize(ideal)
    B = _normalize(benchmark)

    E_lr = np.log(E)
    I_lr = np.log(I)
    B_lr = np.log(B)

    # --- Bottom-panel gaps in bps ---
    # friction drag: ideal vs executed (positive means costs hurt executed)
    drag_bps = np.log(I / E) * 1e4
    # relative performance: executed vs benchmark (positive means E > B)
    rel_bps = np.log(E / B) * 1e4

    # Smooth (rolling median) for paper readability
    drag_sm = drag_bps.rolling(smooth_window_days, min_periods=max(10, smooth_window_days // 5)).median()
    rel_sm = rel_bps.rolling(smooth_window_days, min_periods=max(10, smooth_window_days // 5)).median()

    # Winsorize for stable scale
    drag_plot, drag_lo, drag_hi = _winsorize(drag_sm, 0.01, 0.99)
    rel_plot, rel_lo, rel_hi = _winsorize(rel_sm, 0.01, 0.99)

    # --- Figure layout ---
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Panel A: cumulative log return
    ax1.plot(E_lr.index, E_lr.values, color="C0", linestyle="-", linewidth=3.2, label="Executed (net)")
    ax1.plot(I_lr.index, I_lr.values, color="C1", linestyle="--", linewidth=2.8, label="Ideal (frictionless)")
    ax1.plot(B_lr.index, B_lr.values, color="C2", linestyle=":", linewidth=2.8, label="Benchmark (EW universe)")

    ax1.set_title("Cumulative performance (out-of-sample)", pad=10)
    ax1.set_ylabel("Cumulative log return")
    ax1.grid(True, axis="y", alpha=0.25, linewidth=1.0)
    leg1 = ax1.legend(loc="upper left", frameon=True, fancybox=True)
    leg1.get_frame().set_alpha(0.95)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Panel B: gaps (two axes, same black axis styling)
    ax2.plot(rel_plot.index, rel_plot.values, color="C2", linestyle="--", linewidth=2.6,
             label=r"Relative perf.: $\ln(E/B)\times 10{,}000$ (smoothed)")
    ax2.axhline(0.0, color="0.55", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Relative performance (bps)")
    ax2.grid(True, axis="y", alpha=0.25, linewidth=1.0)

    ax2b = ax2.twinx()
    ax2b.plot(drag_plot.index, drag_plot.values, color="C1", linestyle="-", linewidth=2.4,
              label=r"Friction drag: $\ln(I/E)\times 10{,}000$ (smoothed)")
    ax2b.set_ylabel("Friction drag (bps)")
    ax2b.tick_params(axis="y", colors="black")
    ax2b.yaxis.label.set_color("black")

    # Merge legends for bottom panel
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    leg2 = ax2.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fancybox=True)
    leg2.get_frame().set_alpha(0.95)

    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_locator(YearLocator(2))
    ax2.xaxis.set_major_formatter(DateFormatter("%Y"))

    # Small disclosure note (reviewer-friendly)
    ax2.text(
        0.995, 0.05,
        f"Bottom: {smooth_window_days}d rolling median; winsorized to 1–99th pct. "
        f"Benchmark uses {n_assets_used}/{len(UNIVERSE)} assets.",
        transform=ax2.transAxes, ha="right", va="bottom",
        fontsize=9, color="0.35"
    )

    # Save
    out_png = os.path.join(out_dir, f"{filename_stem}.png")
    out_pdf = os.path.join(out_dir, f"{filename_stem}.pdf")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    artifacts = base / "artifacts"
    processed_dir = base / "data" / "processed"
    out_dir = artifacts / "figures_pub"

    plot_cumperf_two_panel_pub(
        equity_csv=str(artifacts / "equity_equal_weight.csv"),
        processed_dir=str(processed_dir),
        out_dir=str(out_dir),
        filename_stem="fig_cumperf_two_panel_pub",
        smooth_window_days=65,
    )
    print("Wrote two-panel cumulative performance figure to", out_dir)
