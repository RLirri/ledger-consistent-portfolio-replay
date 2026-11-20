"""Paired stationary block bootstrap (Politis and Romano, 1994) for matched arms.

Index generation: each replicate is a sequence of n session indices built from blocks. A block starts
at a uniformly drawn session and has a geometric length with mean `mean_block` (success probability
1 / mean_block); indices advance by one and wrap around the end of the sample (circular). The same
index matrix is applied to every arm of a contrast, so pairing across arms is preserved.

Random numbers: numpy.random.Generator(PCG64(seed)); the draw order is fixed and documented in
`stationary_indices`, so a result is reproducible from (n, replicates, mean_block, seed).

Intervals: percentile intervals at a stated level. A replicate whose statistic is undefined (for
example zero variance in a Sharpe ratio) is counted; if any replicate is undefined the interval is
not reported (None) and the count is returned. Nothing is dropped silently.

These intervals describe the stability of a contrast over this one history. They are not tests.
"""
from __future__ import annotations

import numpy as np

from .errors import ScenarioInputError

K = 252


def stationary_indices(n: int, replicates: int, mean_block: float, seed: int) -> np.ndarray:
    if not (isinstance(n, (int, np.integer)) and n >= 2 and isinstance(replicates, (int, np.integer)) and replicates >= 1 and mean_block >= 1):
        raise ScenarioInputError("bootstrap needs n >= 2, replicates >= 1 and mean_block >= 1")
    rng = np.random.Generator(np.random.PCG64(seed))
    p = 1.0 / mean_block
    out = np.empty((replicates, n), dtype=np.int64)
    for r in range(replicates):                             # draw order per replicate: start, then one uniform per later position
        starts = rng.integers(0, n, size=n)
        new_block = rng.random(n) < p
        new_block[0] = True
        idx = np.empty(n, dtype=np.int64)
        cur = 0
        for t in range(n):
            cur = starts[t] if new_block[t] else (cur + 1) % n
            idx[t] = cur
        out[r] = idx
    return out


def _returns(nav) -> np.ndarray:
    v = np.asarray(nav, float)
    if v.ndim != 1 or len(v) < 3 or not (np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ScenarioInputError("bootstrap needs a finite positive NAV series with at least two returns")
    return v[1:] / v[:-1] - 1.0


def stat_active_return(ra, rb):
    """Annualized arithmetic active return, arm a minus arm b."""
    return K * float(np.mean(ra - rb))


def stat_sharpe_difference(ra, rb):
    """Sharpe (rf = 0, population SD) of arm a minus that of arm b; NaN if either variance is zero."""
    sa, sb = ra.std(ddof=0), rb.std(ddof=0)
    if not (sa > 0 and sb > 0):
        return float("nan")
    return float(np.sqrt(K) * (ra.mean() / sa - rb.mean() / sb))


def stat_terminal_gap_ann(ra, rb):
    """Geometric annualized wealth gap in rate terms: (W_b / W_a) ** (K / n) - 1; positive = arm a poorer."""
    if np.any(ra <= -1) or np.any(rb <= -1):
        return float("nan")
    return float(np.exp((np.log1p(rb).sum() - np.log1p(ra).sum()) * K / len(ra)) - 1.0)


STATISTICS = {"active_return_ann": stat_active_return, "sharpe_difference": stat_sharpe_difference, "terminal_gap_ann": stat_terminal_gap_ann}


def paired_bootstrap(nav_a, nav_b, replicates: int = 2000, mean_block: float = 20, seed: int = 20261119, level: float = 0.95,
                     statistics=("active_return_ann", "sharpe_difference", "terminal_gap_ann")) -> dict:
    """Percentile intervals for statistics of arm a relative to arm b on an identical calendar."""
    if hasattr(nav_a, "index") and hasattr(nav_b, "index") and not nav_a.index.equals(nav_b.index):
        raise ScenarioInputError("arms must share an identical calendar")
    ra, rb = _returns(nav_a), _returns(nav_b)
    if len(ra) != len(rb):
        raise ScenarioInputError("arms must have the same number of returns")
    if not 0 < level < 1:
        raise ScenarioInputError("level must be in (0, 1)")
    n = len(ra)
    idx = stationary_indices(n, replicates, mean_block, seed)
    out = {"n_returns": n, "replicates": replicates, "mean_block": mean_block, "seed": seed, "level": level, "method": "paired stationary bootstrap, circular, percentile interval", "statistics": {}}
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    for name in statistics:
        fn = STATISTICS[name]
        point = fn(ra, rb)
        draws = np.array([fn(ra[i], rb[i]) for i in idx])
        undefined = int(np.sum(~np.isfinite(draws)))
        rec = {"point": point if np.isfinite(point) else None, "undefined_replicates": undefined}
        if undefined == 0 and np.isfinite(point):
            rec.update(lower=float(np.quantile(draws, lo_q)), upper=float(np.quantile(draws, hi_q)), bootstrap_sd=float(draws.std(ddof=1)) if replicates > 1 else 0.0)
        else:
            rec.update(lower=None, upper=None, bootstrap_sd=None, reason="statistic undefined in the sample or in at least one replicate")
        out["statistics"][name] = rec
    return out
