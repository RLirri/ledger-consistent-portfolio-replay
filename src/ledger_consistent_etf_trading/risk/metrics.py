from __future__ import annotations
import numpy as np
import pandas as pd

def equity_to_returns(equity: pd.Series) -> pd.Series:
    r = equity.pct_change().dropna()
    r.name = "ret"
    return r

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    growth = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    return float(growth ** (1.0 / years) - 1.0) if years > 0 else 0.0

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return 0.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    ex = returns - rf_daily
    denom = ex.std(ddof=0)
    return float((ex.mean() / denom) * np.sqrt(periods_per_year)) if denom > 0 else 0.0

def hist_var(returns: pd.Series, alpha: float) -> float:
    # alpha=0.01 gives 99% VaR (a negative number)
    if len(returns) == 0:
        return 0.0
    return float(np.quantile(returns.dropna(), alpha))

def beta_to_benchmark(returns: pd.Series, bench_returns: pd.Series) -> float:
    df = pd.concat([returns, bench_returns], axis=1).dropna()
    if df.shape[0] < 10:
        return 0.0
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    varx = np.var(x)
    return float(np.cov(x, y, ddof=0)[0, 1] / varx) if varx > 0 else 0.0
