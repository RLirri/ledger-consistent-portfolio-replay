import numpy as np
import pandas as pd

from ledger_consistent_etf_trading.v2.config import CostParams, ScenarioConfig
from ledger_consistent_etf_trading.v2.data import market_from_frames

NOCOST = CostParams(enabled=False)


def synth(T=70, tickers=("AAA", "BBB", "CCC", "DDD", "EEE"), seed=7, dollar_volume=5e6, flat=False):
    """Synthetic sessions (future-dated on purpose). Random walk closes, gapped opens, noisy volume."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2030-01-01", periods=T, freq="B", tz="UTC")
    n = len(tickers)
    if flat:
        close = np.full((T, n), 100.0); open_ = close.copy()
    else:
        close = 100.0 * np.cumprod(1 + 0.012 * rng.standard_normal((T, n)), axis=0)
        prev = np.vstack([close[:1], close[:-1]])
        open_ = prev * (1 + 0.004 * rng.standard_normal((T, n)))
    vol = dollar_volume / 100.0 * (1 + 0.3 * rng.random((T, n)))
    f = lambda a: pd.DataFrame(a, index=idx, columns=list(tickers))
    return market_from_frames(f(open_), f(close), f(vol))


def cfg(mkt, start=25, **kw):
    kw.setdefault("name", "t")
    kw.setdefault("eval_start", str(mkt.dates[start].date()))
    return ScenarioConfig(**kw)


def weights(mkt, **w):
    v = np.zeros(len(mkt.tickers))
    for k, x in w.items():
        v[mkt.tickers.index(k)] = x
    return v
