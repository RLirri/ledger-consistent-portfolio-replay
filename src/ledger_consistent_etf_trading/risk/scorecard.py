from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.risk.metrics import (
    equity_to_returns, annualized_return, annualized_vol, sharpe_ratio,
    max_drawdown, hist_var, beta_to_benchmark
)

def build_scorecard(
    equity_df: pd.DataFrame,
    fills_df: pd.DataFrame,
    panel: pd.DataFrame,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    # ---- Fix 1: ensure equity is indexed by date ----
    eq = equity_df.copy()

    # equity CSV might come with either:
    # (a) a 'date' column, or (b) first column unnamed that is date index.
    if "date" in eq.columns:
        eq["date"] = pd.to_datetime(eq["date"], utc=True)
        eq = eq.set_index("date")
    else:
        # common case: first column is the index saved by to_csv
        first = eq.columns[0]
        eq[first] = pd.to_datetime(eq[first], utc=True, errors="coerce")
        eq = eq.dropna(subset=[first]).set_index(first)
        eq.index.name = "date"

    equity = eq["equity"].astype(float).sort_index()
    rets = equity_to_returns(equity)

    # ---- Benchmark returns on same date index ----
    bench_close = panel.xs(benchmark, level="ticker")["close"].astype(float)
    bench_ret = bench_close.pct_change().dropna()

    # Align returns for beta
    beta = beta_to_benchmark(rets, bench_ret)

    # ---- Turnover: sum abs notional per day / equity (mean over ALL days) ----
    fills = fills_df.copy()
    fills["date"] = pd.to_datetime(fills["date"], utc=True)
    fills["abs_notional"] = fills["notional_$"].abs()

    daily_traded = fills.groupby("date")["abs_notional"].sum().sort_index()

    # Reindex to full equity calendar (zeros on non-trade days)
    daily_traded = daily_traded.reindex(equity.index).fillna(0.0)
    turnover = (daily_traded / equity).fillna(0.0)

    # ---- Costs ----
    total_cost = float(fills.get("total_cost_$", pd.Series(dtype=float)).sum())
    spread_cost = float(fills.get("spread_cost_$", pd.Series(dtype=float)).sum())
    slip_cost = float(fills.get("slippage_cost_$", pd.Series(dtype=float)).sum())
    fee_cost = float(fills.get("fee_cost_$", pd.Series(dtype=float)).sum())

    out = {
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "cagr": annualized_return(rets),
        "ann_vol": annualized_vol(rets),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(equity),
        "var_95": hist_var(rets, 0.05),
        "var_99": hist_var(rets, 0.01),
        "beta_to_spy": beta,
        "avg_daily_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "total_cost_$": total_cost,
        "fee_cost_$": fee_cost,
        "spread_cost_$": spread_cost,
        "slippage_cost_$": slip_cost,
        "num_fills": int(len(fills)),
    }
    return pd.DataFrame([out])
