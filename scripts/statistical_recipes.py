"""Fixed saved-path reporting recipes; importing this module performs no computation."""

import numpy as np
from ledger_consistent_etf_trading.v2 import factorial


def returns(v):
    return v[1:] / v[:-1] - 1


def point(v, ref):
    assert len(v) == len(ref) and np.all(v > 0) and np.all(ref > 0)
    ra, rb = returns(v), returns(ref)
    n = len(ra)
    sa, sb = ra.std(), rb.std()
    return {
        "n_returns": n,
        "terminal_gap_bps": 1e4 * (ref[-1] - v[-1]) / ref[-1],
        "terminal_gap_ann_bps": 1e4 * np.expm1((np.log(ref[-1] / v[-1])) * 252 / n),
        "active_return_ann": 252 * float(np.mean(ra - rb)),
        "return_te_ann": float(np.std(ra - rb, ddof=1) * np.sqrt(252))
        if n > 1
        else None,
        "sharpe_difference": float(np.sqrt(252) * (ra.mean() / sa - rb.mean() / sb))
        if sa > 0 and sb > 0
        else None,
    }


def boot(ra, rb, ix):
    n = len(ra)
    B = len(ix)
    draws = {
        k: [] for k in ("active_return_ann", "sharpe_difference", "terminal_gap_ann")
    }
    for off in range(0, B, 64):
        j = ix[off : off + 64]
        x, y = ra[j], rb[j]
        sx, sy = x.std(axis=1), y.std(axis=1)
        draws["active_return_ann"].extend((252 * (x - y).mean(axis=1)).tolist())
        with np.errstate(divide="ignore", invalid="ignore"):
            draws["sharpe_difference"].extend(
                (np.sqrt(252) * (x.mean(axis=1) / sx - y.mean(axis=1) / sy)).tolist()
            )
        draws["terminal_gap_ann"].extend(
            np.expm1(
                252 / n * (np.log1p(y).sum(axis=1) - np.log1p(x).sum(axis=1))
            ).tolist()
        )
    pointvalues = {
        "active_return_ann": float(252 * (ra - rb).mean()),
        "sharpe_difference": float(
            np.sqrt(252) * (ra.mean() / ra.std() - rb.mean() / rb.std())
        )
        if ra.std() > 0 and rb.std() > 0
        else None,
        "terminal_gap_ann": float(
            np.expm1(252 / n * (np.log1p(rb).sum() - np.log1p(ra).sum()))
        ),
    }
    out = {}
    for k, d in draws.items():
        z = np.array(d)
        bad = int((~np.isfinite(z)).sum())
        p = pointvalues[k]
        out[k] = {
            "point": p,
            "undefined_replicates": bad,
            "lower": float(np.quantile(z, 0.025))
            if not bad and p is not None
            else None,
            "upper": float(np.quantile(z, 0.975))
            if not bad and p is not None
            else None,
            "bootstrap_sd": float(z.std(ddof=1)) if not bad and p is not None else None,
        }
    return {
        "replicates": 2000,
        "seed": 20261119,
        "level": 0.95,
        "pointwise": True,
        "statistics": out,
    }


def factor(cells, k):
    beta = factorial.effects(cells, k)
    return {
        "cells": {"".join(map(str, c)): v for c, v in cells.items()},
        "beta": {
            ",".join(map(str, s)) if s else "grand_mean": v for s, v in beta.items()
        },
        "factorial_effects_2beta": {
            ",".join(map(str, s)): 2 * v for s, v in beta.items() if s
        },
        "max_reconstruction_error": factorial.reconstruction_error(cells, k),
    }
