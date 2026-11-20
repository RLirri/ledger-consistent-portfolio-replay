from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market


def equal_weight_weekly(mkt: Market, start: int, end: int) -> dict:
    """Signal on the first evaluation session of each calendar week; execute at the next session.

    Returns {execution_session_index: weight_vector}. Matches the legacy tape convention
    (pandas weekly periods ending Sunday). The signal carries no market information.
    """
    naive = mkt.dates[start:end].tz_localize(None)
    week = pd.Series(naive.to_period("W"))
    first = week.drop_duplicates().index.to_numpy() + start
    n = len(mkt.tickers)
    return {int(s) + 1: np.full(n, 1.0 / n) for s in first if s + 1 < end}


def momentum_top4(mkt: Market, start: int, end: int, lookback: int = 20, top: int = 4) -> dict:
    """Daily relative momentum, a turnover generator with no alpha claim.

    For execution session d the score is close[d-1] / close[d-1-lookback] - 1, known before the open.
    Eligible: finite, strictly positive score. Ranking: score descending, ties broken by ticker
    (alphabetical, i.e. column order). The selected assets share weight equally; all cash if none.
    A missing or non-positive price in either endpoint makes the asset ineligible that day.
    The first target is issued at start+1, the same first execution session as the weekly strategy.
    """
    out = {}
    n = len(mkt.tickers)
    for d in range(start + 1, end):
        s = d - 1
        if s - lookback < 0:
            raise ValueError("insufficient history for momentum signal")
        a, b = mkt.close[s], mkt.close[s - lookback]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        score = np.where(ok, a / np.where(ok, b, 1.0) - 1.0, -np.inf)
        order = sorted(range(n), key=lambda i: (-score[i], i))
        chosen = [i for i in order if score[i] > 0][:top]
        w = np.zeros(n)
        if chosen:
            w[chosen] = 1.0 / len(chosen)
        out[d] = w
    return out


def from_tape(mkt: Market, tape: pd.DataFrame, start: int, end: int) -> dict:
    """Targets from a (date, ticker, target_weight) signal tape; execution at the next session."""
    t = tape.copy()
    t["date"] = pd.to_datetime(t["date"], utc=True)
    col = {k: i for i, k in enumerate(mkt.tickers)}
    out = {}
    for sig, g in t.groupby("date"):
        pos = int(mkt.dates.searchsorted(sig))
        e = pos + 1 if pos < len(mkt.dates) and mkt.dates[pos] == sig else pos
        if start < e < end:
            w = np.zeros(len(mkt.tickers))
            for tk, x in zip(g["ticker"], g["target_weight"]):
                w[col[tk]] = float(x)
            out[e] = w
    return out


def build_targets(mkt: Market, cfg, start: int, end: int, tape=None) -> dict:
    if cfg.strategy == "equal_weight_weekly":
        return equal_weight_weekly(mkt, start, end)
    if cfg.strategy == "momentum_top4":
        return momentum_top4(mkt, start, end, cfg.momentum_lookback, cfg.momentum_top)
    if tape is None:
        raise ValueError("strategy 'tape' needs a tape")
    return from_tape(mkt, tape, start, end)
