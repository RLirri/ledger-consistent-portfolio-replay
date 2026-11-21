"""Post-hoc cost adjustment: the simplified treatment the stateful replay is compared against.

Take the unconstrained reference run (no costs, no liquidity limit). Price its own trade tape with
the cost model, and subtract that cost from its return series. Trade sizes, holdings and targets
are the reference's and never respond to the costs (no feedback).

    c_d       = sum_i cost(|x_{i,d}| * O_{i,d}; ADV_{i,d}, sigma_{i,d})        (same inputs as the engine)
    r_adj[d]  = r_ref[d] - c_d / NAV_ref[d-1]
    NAV_adj[d] = NAV_adj[d-1] * (1 + r_adj[d]),  NAV_adj[0] = NAV_ref[0] - c_0

Costs therefore scale with adjusted wealth (a rate convention); the reference trade tape is not
re-sized. The result is a return-only series: no holdings, weights, completion or order metrics
exist for it and none may be reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Features, Market
from .errors import ScenarioInputError
from .engine import RunResult, _cost


ALLOWED_DIFFERENCES = {"name", "experiment", "cap_enabled", "max_participation"}      # besides the cost block


def posthoc_cost_adjusted(ref: RunResult, mkt: Market, feats: Features, cost_cfg) -> pd.DataFrame:
    """`cost_cfg` is the configuration of the stateful arm the series will be compared with. Apart from the
    cost block and the liquidity limit, it must equal the reference's configuration: same strategy, timing,
    capital, cash, unit, minimum-trade and residual rules and the same evaluation range."""
    if ref.cfg.cost.enabled or ref.cfg.cap_enabled or ref.cfg.max_trade_notional is not None:
        raise ScenarioInputError("the post-hoc comparator must be derived from the unconstrained reference (no cost, no cap)")
    if not cost_cfg.cost.enabled:
        raise ScenarioInputError("cost model must be enabled for the adjustment")
    if (cost_cfg.features, cost_cfg.adv_estimator) != (feats.mode, feats.estimator):
        raise ScenarioInputError("feature basis of the adjustment must match the stateful arm it is compared with")
    a, b = ref.cfg.to_dict(), cost_cfg.to_dict()
    extra = sorted(k for k in a if k != "cost" and a[k] != b[k] and k not in ALLOWED_DIFFERENCES)
    if extra:
        raise ScenarioInputError(f"reference and stateful arm differ in fields that must match: {extra}")
    if ref.info.get("market_id") != mkt.identity():
        raise ScenarioInputError("reference run was produced from different market data")
    if feats.market_id is not None and feats.market_id != mkt.identity():
        raise ScenarioInputError("features were computed from different market data")
    if feats.vol.shape != mkt.close.shape:
        raise ScenarioInputError("features do not match the market calendar")
    col = {t: i for i, t in enumerate(mkt.tickers)}
    n = len(mkt.tickers)
    cost_by_session = {}
    for d, g in ref.orders.groupby("session"):
        notional = np.zeros(n)
        for t, x in zip(g["ticker"], g["notional"]):
            notional[col[t]] += abs(x)
        cost_by_session[int(d)] = float(_cost(cost_cfg, notional, feats.adv[d], feats.vol[d]).sum())
    nav_ref = ref.daily["nav"].to_numpy(float)
    if not (np.all(np.isfinite(nav_ref)) and np.all(nav_ref > 0)):
        raise ScenarioInputError("reference NAV must be finite and positive")
    start = ref.info["start"]
    c = np.array([cost_by_session.get(start + k, 0.0) for k in range(len(nav_ref))])
    adj = np.empty_like(nav_ref)
    adj[0] = nav_ref[0] - c[0]
    growth = np.empty_like(nav_ref); growth[0] = adj[0] / ref.cfg.initial_cash
    for k in range(1, len(nav_ref)):
        growth[k] = nav_ref[k] / nav_ref[k - 1] - c[k] / nav_ref[k - 1]
        adj[k] = adj[k - 1] * growth[k]
    if not (np.all(np.isfinite(growth)) and np.all(growth > 0) and np.all(adj > 0)):
        raise ScenarioInputError("post-hoc adjustment produced a non-positive or non-finite growth factor; no series is returned")
    out = pd.DataFrame({"nav": adj, "posthoc_cost": c}, index=ref.dates)
    out.attrs.update(reference_scenario_id=ref.cfg.scenario_id, cost_parameters=cost_cfg.cost.__dict__.copy(),
                     rate_convention="r_adj[d] = r_ref[d] - c_d / NAV_ref[d-1]; first session c_0 subtracted in level", market_id=mkt.identity())
    return out
