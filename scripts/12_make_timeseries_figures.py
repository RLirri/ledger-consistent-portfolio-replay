# scripts/12_make_timeseries_figures.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Config
# -----------------------------
ROOT = Path(".")
ARTIFACTS = ROOT / "artifacts"
DATA_PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures" / "timeseries"

# Equal-weight artifacts you showed in `find ...`
EXEC_EQUITY_CSV = ARTIFACTS / "equity_equal_weight.csv"
TARGET_WEIGHTS_CSV = ARTIFACTS / "weights_equal_weight.csv"          # target weights (should sum to 1)
EXEC_POSITIONS_CSV = ARTIFACTS / "positions_equal_weight.csv"        # executed holdings/positions (if available)

# If you have a panel (aligned prices) we can use it
PANEL_ALIGNED = DATA_PROCESSED / "panel_1d_aligned.parquet"
PANEL_FALLBACK = DATA_PROCESSED / "panel_1d.parquet"

# Fallback for SPY
SPY_CLEAN = DATA_PROCESSED / "SPY_1d_clean.parquet"


# -----------------------------
# Publication style
# -----------------------------
def set_pub_style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Helpers: IO
# -----------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    # try common date columns
    for c in ["date", "Date", "timestamp", "time", "datetime"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        # if first column looks like dates
        maybe = df.columns[0]
        try:
            tmp = pd.to_datetime(df[maybe])
            df = df.set_index(tmp).drop(columns=[maybe])
        except Exception:
            pass
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    return df


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    # ensure datetime index when possible
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["date", "Date", "timestamp", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    return df


def _to_series(df: pd.DataFrame, candidates: list[str], name: str) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(float).copy()
            s.name = name
            s.index = pd.to_datetime(s.index)
            return s.sort_index().dropna()
    raise KeyError(f"Could not find any of columns {candidates} in dataframe columns={list(df.columns)[:20]}...")


# -----------------------------
# Prices loading
# -----------------------------
def load_spy_close() -> pd.Series:
    # preferred: SPY clean parquet
    if SPY_CLEAN.exists():
        df = _read_parquet(SPY_CLEAN)
        for col in ["close", "Close", "adj_close", "Adj Close", "adjClose"]:
            if col in df.columns:
                s = df[col].astype(float)
                s.name = "SPY"
                return s.sort_index().dropna()
    raise FileNotFoundError("Could not load SPY close price from data/processed/SPY_1d_clean.parquet")


def load_panel_close(symbols: list[str]) -> pd.DataFrame | None:
    """
    Tries to load a panel of closes from panel_1d_aligned.parquet or panel_1d.parquet.
    This function is defensive because panel format varies by implementation.
    Returns: DataFrame indexed by date, columns = symbols, values = close
    """
    panel_path = PANEL_ALIGNED if PANEL_ALIGNED.exists() else PANEL_FALLBACK
    if not panel_path.exists():
        return None

    df = _read_parquet(panel_path)

    # Case A: columns like 'SPY_close', 'QQQ_close', etc.
    cols = {}
    for sym in symbols:
        for cand in [f"{sym}_close", f"{sym}.close", f"{sym}:close", f"{sym}_Close"]:
            if cand in df.columns:
                cols[sym] = cand
                break
    if cols:
        out = df[list(cols.values())].copy()
        out.columns = list(cols.keys())
        out.index = pd.to_datetime(out.index)
        return out.sort_index().dropna(how="all")

    # Case B: MultiIndex columns (symbol, field)
    if isinstance(df.columns, pd.MultiIndex):
        # try (sym, 'close')
        frames = []
        for sym in symbols:
            for field in ["close", "Close", "adj_close", "Adj Close"]:
                if (sym, field) in df.columns:
                    frames.append(df[(sym, field)].rename(sym))
                    break
        if frames:
            out = pd.concat(frames, axis=1)
            out.index = pd.to_datetime(out.index)
            return out.sort_index().dropna(how="all")

    # Case C: long format with columns ['symbol','close']
    if {"symbol", "close"}.issubset(df.columns):
        out = df.reset_index()
        out["date"] = pd.to_datetime(out[out.columns[0]])
        out = out[out["symbol"].isin(symbols)]
        out = out.pivot(index="date", columns="symbol", values="close").sort_index()
        return out.dropna(how="all")

    return None


def load_symbol_close(sym: str) -> pd.Series:
    # fallback: per-symbol parquet exists in data/processed/{SYM}_1d_clean.parquet
    p = DATA_PROCESSED / f"{sym}_1d_clean.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing price file for {sym}: {p}")
    df = _read_parquet(p)
    for col in ["close", "Close", "adj_close", "Adj Close", "adjClose"]:
        if col in df.columns:
            s = df[col].astype(float).rename(sym)
            return s.sort_index().dropna()
    raise KeyError(f"No close column found in {p}")


# -----------------------------
# Finance math
# -----------------------------
def normalize_to_1(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    if s.empty:
        return s
    return s / float(s.iloc[0])


def equity_from_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """
    Frictionless equity curve from target weights.
    Uses daily close-to-close returns and applies weights at t-1 to returns at t.
    """
    weights = weights.copy()
    prices = prices.copy()

    weights.index = pd.to_datetime(weights.index)
    prices.index = pd.to_datetime(prices.index)

    # align
    idx = weights.index.intersection(prices.index)
    weights = weights.loc[idx].sort_index()
    prices = prices.loc[idx].sort_index()

    # ensure columns overlap
    common = [c for c in weights.columns if c in prices.columns]
    if not common:
        raise ValueError("No common symbols between weights and prices.")
    weights = weights[common].astype(float)
    prices = prices[common].astype(float)

    # normalize weights rows to sum 1 (defensive)
    row_sum = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(row_sum, axis=0).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    port_ret = (weights.shift(1).fillna(method="bfill") * rets).sum(axis=1)

    eq = (1.0 + port_ret).cumprod()
    eq.name = "ideal_eq"
    return eq


def drawdown(eq: pd.Series) -> pd.Series:
    eq = eq.dropna().sort_index()
    peak = eq.cummax()
    return eq / peak - 1.0


def weight_tracking_error(target_w: pd.DataFrame, exec_w: pd.DataFrame) -> pd.Series:
    """
    Weight tracking error = 0.5 * sum_i |w_exec - w_target|
    (Total variation distance between weight vectors.)
    """
    target_w = target_w.copy()
    exec_w = exec_w.copy()
    target_w.index = pd.to_datetime(target_w.index)
    exec_w.index = pd.to_datetime(exec_w.index)

    idx = target_w.index.intersection(exec_w.index)
    target_w = target_w.loc[idx].sort_index()
    exec_w = exec_w.loc[idx].sort_index()

    common = [c for c in target_w.columns if c in exec_w.columns]
    if not common:
        raise ValueError("No common symbols between target and executed weights.")
    target_w = target_w[common].astype(float)
    exec_w = exec_w[common].astype(float)

    # normalize rows
    target_w = target_w.div(target_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    exec_w = exec_w.div(exec_w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    te = 0.5 * (exec_w - target_w).abs().sum(axis=1)
    te.name = "TE"
    return te


# -----------------------------
# Regime shading (no overlap)
# -----------------------------
@dataclass(frozen=True)
class Regime:
    start: str
    end: str
    label: str


REGIMES = [
    Regime("2018-10-01", "2018-12-31", "Q4-2018"),
    Regime("2020-02-15", "2020-05-30", "COVID"),
    Regime("2022-01-01", "2022-12-31", "Rates"),
]


def add_regime_shading(ax: plt.Axes, regimes=REGIMES, alpha=0.10):
    """
    Publication-safe regime shading:
    - uses datetime spans
    - labels placed in axes coordinates to avoid title collision
    - staggers y positions to avoid label overlap
    """
    y_levels = [0.98, 0.92, 0.86, 0.80]  # enough for a few regimes
    for i, r in enumerate(regimes):
        s = pd.Timestamp(r.start)
        e = pd.Timestamp(r.end)
        ax.axvspan(s, e, alpha=alpha, zorder=0)

        x = s + (e - s) / 2
        y = y_levels[i % len(y_levels)]
        ax.text(
            x,
            y,
            r.label,
            transform=ax.get_xaxis_transform(),  # x in data, y in axes coords
            ha="center",
            va="center",
            fontsize=12,
            color="0.35",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=2.0),
            zorder=5,
        )


# -----------------------------
# Plotting
# -----------------------------
def percent_fmt(x, _pos):
    return f"{x*100:.0f}%"


def plot_equity_log_exec_vs_ideal(executed_eq: pd.Series, ideal_eq: pd.Series, bench_eq: pd.Series | None):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    # Normalize
    executed = normalize_to_1(executed_eq)
    ideal = normalize_to_1(ideal_eq)

    # log scale by log of normalized equity (keeps shapes comparable)
    ax.plot(executed.index, np.log(executed.values), label="Executed", linewidth=2.6, zorder=3)
    ax.plot(ideal.index, np.log(ideal.values), label="Ideal (frictionless)", linewidth=2.2, linestyle="--", zorder=2)

    if bench_eq is not None and not bench_eq.empty:
        bench = normalize_to_1(bench_eq)
        ax.plot(bench.index, np.log(bench.values), label="Benchmark (SPY)", linewidth=2.2, alpha=0.85, zorder=1)

    add_regime_shading(ax)

    ax.set_title("Cumulative performance (out-of-sample)")
    ax.set_ylabel("log Normalized equity")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=True)
    return fig


def plot_drawdown_exec_vs_ideal(executed_eq: pd.Series, ideal_eq: pd.Series):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    dd_exec = drawdown(executed_eq)
    dd_ideal = drawdown(ideal_eq)

    # make executed visually dominant, ideal dashed + slightly transparent
    ax.plot(dd_exec.index, dd_exec.values, label="Executed", linewidth=2.6, zorder=3)
    ax.plot(dd_ideal.index, dd_ideal.values, label="Ideal (frictionless)", linewidth=2.2, linestyle="--", alpha=0.9, zorder=2)

    add_regime_shading(ax)

    ax.axhline(0, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_title("Drawdown (out-of-sample)")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(FuncFormatter(percent_fmt))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.legend(loc="lower left", frameon=True)
    return fig


def plot_te_over_time(te: pd.Series, window: int = 63):
    set_pub_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.0))

    te = te.dropna().sort_index()
    te_roll = te.rolling(window).mean()

    ax.plot(te_roll.index, te_roll.values, linewidth=2.6, label=f"Rolling TE ({window}d mean)", zorder=3)

    # guide bands (quantiles) – better than random dashed lines
    q25, q50, q75 = np.nanquantile(te_roll.values, [0.25, 0.50, 0.75])
    for q, ls in [(q25, ":"), (q50, "--"), (q75, ":")]:
        ax.axhline(q, linestyle=ls, linewidth=1.2, alpha=0.5)

    add_regime_shading(ax)

    ax.set_title("Tracking error over time")
    ax.set_ylabel("Weight tracking error")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=True)
    return fig


# -----------------------------
# Main: load executed_eq, ideal_eq, bench_eq, te_series
# -----------------------------
def load_executed_equity() -> pd.Series:
    df = _read_csv(EXEC_EQUITY_CSV)
    # common column names to try
    return _to_series(
        df,
        candidates=["equity", "executed_equity", "equity_executed", "portfolio_value", "nav", "value"],
        name="executed_eq",
    )


def load_target_weights() -> pd.DataFrame:
    w_raw = _read_csv(TARGET_WEIGHTS_CSV).copy()
    w_raw.index = pd.to_datetime(w_raw.index)
    w_raw = w_raw.sort_index()

    cols_lower = {c.lower(): c for c in w_raw.columns}

    # Case A: Long format (symbol + weight column)
    if ("symbol" in cols_lower) or ("ticker" in cols_lower):
        sym_col = cols_lower.get("symbol", cols_lower.get("ticker"))

        # choose weight column (common patterns)
        weight_candidates = [
            "target_weight", "weight", "w", "target_w", "target", "desired_weight", "targetWeight"
        ]
        weight_col = None
        for k in weight_candidates:
            if k.lower() in cols_lower:
                weight_col = cols_lower[k.lower()]
                break

        if weight_col is None:
            # fallback: pick first numeric column (excluding obvious ids)
            numeric_cols = w_raw.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                raise ValueError(
                    "weights_equal_weight.csv looks like long format but no numeric weight column found."
                )
            weight_col = numeric_cols[0]

        tmp = w_raw.reset_index().rename(columns={w_raw.index.name or "index": "date"})
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp[sym_col] = tmp[sym_col].astype(str).str.upper()

        wide = (
            tmp.pivot(index="date", columns=sym_col, values=weight_col)
            .sort_index()
            .fillna(0.0)
        )

        # keep only columns that look like tickers (2-6 uppercase letters)
        wide = wide.loc[:, wide.columns.astype(str).str.match(r"^[A-Z]{2,6}$")]

        # normalize rows (defensive)
        rs = wide.sum(axis=1).replace(0, np.nan)
        wide = wide.div(rs, axis=0).fillna(0.0)

        return wide

    # ---------
    # Case B: Wide format (tickers already as columns)
    # ---------
    # Drop obvious non-ticker columns
    drop_like = {"cash", "turnover", "target_weight", "weight", "date"}
    keep_cols = []
    for c in w_raw.columns:
        cl = c.lower()
        if cl in drop_like:
            continue
        # keep columns that look like tickers
        if str(c).upper().match if False else True:
            pass
        keep_cols.append(c)

    w = w_raw[keep_cols].select_dtypes(include=[np.number]).copy()

    # also filter by ticker regex to avoid columns like "target_weight"
    w = w.loc[:, w.columns.astype(str).str.match(r"^[A-Z]{2,6}$")]

    if w.shape[1] == 0:
        raise ValueError(
            "weights_equal_weight.csv could not be parsed as wide format (no ticker-like columns). "
            "It might be long format but missing 'symbol'/'ticker' column."
        )

    rs = w.sum(axis=1).replace(0, np.nan)
    w = w.div(rs, axis=0).fillna(0.0)
    return w.sort_index()



def load_executed_weights_from_positions(prices: pd.DataFrame, positions_path: Path) -> pd.DataFrame:
    """
    Attempts to build executed weights from positions file.
    Supports both:
      - wide format: columns are tickers with values as shares or dollars
      - long format: columns include symbol + shares/qty/position
    """
    pos = _read_csv(positions_path)
    pos.index = pd.to_datetime(pos.index)

    # long format
    if {"symbol"}.issubset(pos.columns):
        qty_col = None
        for c in ["shares", "qty", "quantity", "position", "units"]:
            if c in pos.columns:
                qty_col = c
                break
        if qty_col is None:
            raise ValueError("positions_equal_weight.csv long format detected but no shares/qty column found.")
        tmp = pos.reset_index().rename(columns={pos.index.name or "index": "date"})
        tmp["date"] = pd.to_datetime(tmp["date"])
        wide_qty = tmp.pivot(index="date", columns="symbol", values=qty_col).fillna(0.0)
        wide_qty = wide_qty.sort_index()
        common = [c for c in wide_qty.columns if c in prices.columns]
        wide_qty = wide_qty[common]
        value = wide_qty * prices.reindex(wide_qty.index)[common]
        w_exec = value.div(value.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        return w_exec

    # wide format
    wide = pos.select_dtypes(include=[np.number]).copy()
    wide = wide.sort_index()
    common = [c for c in wide.columns if c in prices.columns]
    if not common:
        raise ValueError("positions file is wide format but no columns match price symbols.")
    value = wide[common] * prices.reindex(wide.index)[common]
    w_exec = value.div(value.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return w_exec


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ---- executed_eq ----
    executed_eq = load_executed_equity()

    # ---- target weights + prices => ideal_eq ----
    target_w = load_target_weights()
    symbols = list(target_w.columns)

    # prices: try panel first, else per-symbol fallbacks
    prices = load_panel_close(symbols)
    if prices is None:
        series = [load_symbol_close(sym) for sym in symbols]
        prices = pd.concat(series, axis=1).dropna(how="all")

    ideal_eq = equity_from_weights(target_w, prices)

    # ---- benchmark (SPY) ----
    try:
        bench_eq = load_spy_close()
    except Exception:
        bench_eq = pd.Series(dtype=float)

    # ---- TE series ----
    # Prefer to compute from executed weights if positions file exists
    if EXEC_POSITIONS_CSV.exists():
        exec_w = load_executed_weights_from_positions(prices, EXEC_POSITIONS_CSV)
        te_series = weight_tracking_error(target_w, exec_w)
    else:
        # fallback: compute TE against equal weights based on price coverage (rough)
        te_series = pd.Series(dtype=float)

    # align all by intersection (keeps plots clean)
    idx = executed_eq.index.intersection(ideal_eq.index)
    if not bench_eq.empty:
        idx = idx.intersection(bench_eq.index)
    executed_eq = executed_eq.loc[idx]
    ideal_eq = ideal_eq.loc[idx]
    bench_eq = bench_eq.loc[idx] if not bench_eq.empty else bench_eq

    if not te_series.empty:
        te_series = te_series.loc[te_series.index.intersection(idx)]

    # ---- plots ----
    fig1 = plot_equity_log_exec_vs_ideal(executed_eq, ideal_eq, bench_eq if not bench_eq.empty else None)
    save_fig(fig1, FIG_DIR, "fig_equity_log_exec_vs_ideal")

    fig2 = plot_drawdown_exec_vs_ideal(executed_eq, ideal_eq)
    save_fig(fig2, FIG_DIR, "fig_drawdown_exec_vs_ideal")

    if not te_series.empty:
        fig3 = plot_te_over_time(te_series, window=63)
        save_fig(fig3, FIG_DIR, "fig_te_time")

    print(f"[OK] Saved figures to: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
