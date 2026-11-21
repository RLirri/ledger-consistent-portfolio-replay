"""V2 metrics. Each has one name, one formula, one sign convention and one denominator.

Positive shortfall / gap / cost = adverse. Annualization K = 252. Sharpe uses rf = 0 and the
population standard deviation (legacy continuity) and is named sharpe_rf0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ScenarioInputError

K = 252


def performance(nav: pd.Series) -> dict:
    """Undefined statistics are returned as NaN here and serialized as null with a reason by `summarize`."""
    v = nav.to_numpy(float)
    if not (np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ScenarioInputError("NAV series must be finite and positive")
    r = v[1:] / v[:-1] - 1.0
    n = len(r)
    if n == 0:
        return dict(n_returns=0, cagr=np.nan, ann_vol=np.nan, sharpe_rf0=np.nan, max_drawdown=np.nan)
    sd = float(r.std(ddof=0))
    return dict(n_returns=n, cagr=float((v[-1] / v[0]) ** (K / n) - 1), ann_vol=sd * np.sqrt(K),
                sharpe_rf0=float(r.mean() / sd * np.sqrt(K)) if sd > 0 else np.nan,
                max_drawdown=float((nav / nav.cummax() - 1).min()))


def direct_cost(res) -> dict:
    """Modelled cost relative to executed notional (the legacy manuscript's Eq. 13 quantity)."""
    g = float(res.orders["notional"].abs().sum())
    c = float(res.orders["cost"].sum())
    return dict(cost_dollars=c, gross_executed_notional=g, direct_cost_bps_exec=1e4 * c / g if g > 0 else np.nan,
                cost_bps_of_mean_nav_per_year=1e4 * c / float(res.daily["nav"].mean()) / (len(res.daily) / K))


def order_shortfall(parents: pd.DataFrame) -> pd.DataFrame:
    """Per-parent implementation shortfall in dollars (Perold paper-versus-real, undiscounted).

    IS = sum x (p_fill - p_D) + u (p_H - p_D) + cost, with signed q = sum x + u.
    Exact split: delay q (p_A - p_D) + execution sum x (p_fill - p_A) + opportunity u (p_H - p_A) + cost.
    A missing mark is an error, never a silently skipped term.
    """
    p = parents.copy()
    need = ["q", "filled", "unfilled", "p_D", "p_A", "p_H", "fill_value", "cost"]
    if len(p) and not np.all(np.isfinite(p[need].to_numpy(float))):
        raise ScenarioInputError("order shortfall needs finite quantities, prices and marks for every parent")
    p["delay"] = p["q"] * (p["p_A"] - p["p_D"])
    p["execution"] = p["fill_value"] - p["filled"] * p["p_A"]
    p["opportunity"] = p["unfilled"] * (p["p_H"] - p["p_A"])
    p["is_dollars"] = p["fill_value"] - p["filled"] * p["p_D"] + p["unfilled"] * (p["p_H"] - p["p_D"]) + p["cost"]
    p["intended_notional"] = p["q"].abs() * p["p_D"]
    p["age_sessions"] = p["close_session"] - p["open_session"]
    return p


def demand_summary(demand: pd.DataFrame) -> dict:
    """Totals of the demand ledger at basis prices. originated = executed + withdrawn + outstanding_end."""
    v = demand.groupby("event")["value"].sum() if len(demand) else pd.Series(dtype=float)
    g = lambda k: float(v.get(k, 0.0))
    orig = g("originated")
    out = dict(originated_value=orig, executed_value=g("executed"), withdrawn_value=g("withdrawn"),
               outstanding_end_value=g("outstanding_end"), overshoot_value=g("overshoot"))
    out["lineage_identity_residual"] = orig - out["executed_value"] - out["withdrawn_value"] - out["outstanding_end_value"]
    for k in ("executed", "withdrawn", "outstanding_end"):
        out[f"{k}_share"] = out[f"{k}_value"] / orig if orig > 0 else None
    return out


def order_summary(parents: pd.DataFrame) -> dict:
    """Aggregate order statistics.

    completion_aggregate = executed demand / originated demand, both valued at the basis price of the
    demand cohort they belong to (see engine docstring). It is a proportion in [0, 1] by accounting,
    is independent of how prices move after the demand arises, and does not depend on the residual
    policy when demand and fills are the same. It is None when no demand was originated.
    episode_served_ratio is the per-target-event analogue (served units over that event's gap, valued
    at the event's sizing price); it counts continued demand again at every event and is therefore
    not comparable across re-targeting frequencies.
    """
    if len(parents) == 0:
        return dict(parents=0, completion_aggregate=None, completion_undefined_reason="no orders")
    p = order_shortfall(parents)
    lineage = "demand_new_value" in p.columns
    if not lineage:
        raise ScenarioInputError("parents carry no demand lineage; completion cannot be computed")
    originated = float(p["demand_new_value"].sum())
    executed = float(p["demand_executed_value"].sum())
    gap_value = float((p["demand_gap"].abs() * p["p_D"]).sum())
    served_value = float((p["demand_executed_units"] * p["p_D"]).sum())
    nz = p[p["q"] != 0]
    comp = nz["filled"].abs() / nz["q"].abs()
    out = dict(parents=int(len(p)), is_dollars=float(p["is_dollars"].sum()), delay=float(p["delay"].sum()),
               execution=float(p["execution"].sum()), opportunity=float(p["opportunity"].sum()), cost=float(p["cost"].sum()),
               intended_notional_gross=float(p["intended_notional"].sum()), new_demand_notional=originated,
               executed_demand_value=executed,
               is_bps_of_new_demand=1e4 * float(p["is_dollars"].sum()) / originated if originated > 0 else None,
               completion_aggregate=executed / originated if originated > 0 else None,
               episode_served_ratio=served_value / gap_value if gap_value > 0 else None,
               completion_parent_mean=float(comp.mean()) if len(comp) else None,
               completion_parent_p05=float(comp.quantile(0.05)) if len(comp) else None,
               parents_incomplete=int((p["unfilled"].abs() > 0).sum()),
               cancelled_or_replaced_notional=float((p["unfilled"].abs() * p["p_D"]).sum()),
               overshoot_value=float(p["overshoot_value"].sum()),
               age_mean=float(p["age_sessions"].mean()), age_p95=float(p["age_sessions"].quantile(0.95)), age_max=int(p["age_sessions"].max()))
    if originated <= 0:
        out["completion_undefined_reason"] = "no demand originated"
    return out


def weight_tracking(res, mkt_close: np.ndarray) -> pd.DataFrame:
    """Daily L2 distance between close-marked realized weights and the target effective that session.

    Targets count from their execution session; sessions before the first target are excluded.
    WTE_risky: assets only. WTE_with_cash: adds the cash coordinate (target cash = 1 - sum of weights).
    """
    s, e = res.info["start"], res.info["end"]
    close = mkt_close[s:e]
    nav = res.daily["nav"].to_numpy()
    w = np.where(res.positions != 0, res.positions * close, 0.0) / nav[:, None]
    active = ~np.isnan(res.target_eff).any(axis=1)
    diff = w - np.nan_to_num(res.target_eff)
    risky = np.sqrt((diff ** 2).sum(axis=1))
    cash_diff = res.daily["cash"].to_numpy() / nav - (1 - np.nan_to_num(res.target_eff).sum(axis=1))
    with_cash = np.sqrt((diff ** 2).sum(axis=1) + cash_diff ** 2)
    out = pd.DataFrame({"wte_risky": risky, "wte_with_cash": with_cash, "cash_weight": res.daily["cash"].to_numpy() / nav}, index=res.dates)
    return out[active]


def relative(nav: pd.Series, nav_ref: pd.Series) -> dict:
    """Arm versus matched reference on an identical calendar. Positive gap = arm finished poorer.

    terminal_gap_bps:      1e4 * (V_ref - V_arm) / V_ref at the last session (not annualized).
    terminal_gap_ann_bps:  1e4 * ((V_ref / V_arm) ** (K / n) - 1), the geometric annual rate equivalent to the
                           terminal wealth ratio over n daily returns; the materiality rule uses this one.
    active_return_ann:     K * mean(r_arm - r_ref), arithmetic.
    """
    if not nav.index.equals(nav_ref.index):
        raise ScenarioInputError("arms must share an identical calendar; refusing to inner-join")
    a, b = nav.to_numpy(float), nav_ref.to_numpy(float)
    if len(a) < 2 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(a > 0) and np.all(b > 0)):
        raise ScenarioInputError("relative metrics need at least two finite positive NAV observations per arm")
    ra, rb = a[1:] / a[:-1] - 1, b[1:] / b[:-1] - 1
    dlt = ra - rb
    n = len(dlt)
    return dict(terminal_gap_bps=1e4 * float((b[-1] - a[-1]) / b[-1]),
                terminal_gap_ann_bps=1e4 * float((b[-1] / a[-1]) ** (K / n) - 1),
                active_return_ann=float(dlt.mean() * K),
                return_te_ann=float(dlt.std(ddof=1) * np.sqrt(K)) if n > 1 else np.nan)


def _clean(x):
    """JSON-safe: NaN/inf become None so that no success artifact contains a non-finite number."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def deployment_session(res) -> int | None:
    """The scheduled deployment session d0: the first target-execution session of the strategy calendar.
    It is fixed by the strategy and evaluation range before any fill exists, so matched arms share it
    whether or not either of them trades on it."""
    ts = res.info.get("target_sessions") or []
    return int(ts[0]) if ts else None


def require_common_deployment(*runs) -> int | None:
    """Matched arms must share calendar and scheduled deployment session before turnover is compared."""
    d0 = {deployment_session(r) for r in runs}
    cal = {(r.info["start"], r.info["end"]) for r in runs}
    if len(d0) != 1 or len(cal) != 1 or any(not runs[0].dates.equals(r.dates) for r in runs):
        raise ScenarioInputError("matched arms do not share an evaluation calendar and scheduled deployment session")
    return d0.pop()


def turnover(res, mask=None, d0="scheduled") -> dict:
    """Gross executed notional / opening pre-trade NAV, averaged over a declared set of sessions.

    daily mean:            mean over the set S of G_d / NAV_open,d
    excluding deployment:  the same mean over S without the scheduled deployment session d0 (N - 1 observations
                           when d0 is in S, N otherwise). d0 is scheduled, not the first actual fill, so a delayed
                           initial buy stays in this figure: it excludes the scheduled deployment session, not all
                           initial deployment trading.
    annualized:            252 x the daily mean, in gross NAV multiples per 252-session year (not compounded).
    """
    d = res.daily
    ratio = (d["gross_traded"] / d["nav_open"]).to_numpy(float)
    keep = np.ones(len(d), bool) if mask is None else np.asarray(mask, bool)
    k0 = deployment_session(res) if d0 == "scheduled" else d0
    k = None if k0 is None else k0 - res.info["start"]
    ex = keep.copy()
    inside = k is not None and 0 <= k < len(d) and keep[k]
    if inside:
        ex[k] = False
    mean = float(ratio[keep].mean()) if keep.any() else None
    mean_ex = float(ratio[ex].mean()) if ex.any() else None
    return dict(turnover_gross_mean_daily=mean, turnover_gross_annualized=None if mean is None else K * mean,
                turnover_observations=int(keep.sum()),
                turnover_gross_mean_daily_excl_deployment_session=mean_ex,
                turnover_gross_annualized_excl_deployment_session=None if mean_ex is None else K * mean_ex,
                turnover_observations_excl_deployment_session=int(ex.sum()),
                deployment_session=k0, deployment_session_date=None if k is None or not (0 <= k < len(d)) else str(d.index[k].date()),
                deployment_session_in_period=bool(inside),
                turnover_denominator="opening pre-trade NAV; all sessions of the declared set")


def period_summary(res, mkt_close: np.ndarray, first=None, last=None) -> dict:
    """Reporting block for a date range [first, last] (inclusive, UTC dates) of one run."""
    d = res.daily
    mask = np.ones(len(d), bool)
    if first is not None:
        mask &= d.index >= pd.Timestamp(first, tz="UTC")
    if last is not None:
        mask &= d.index <= pd.Timestamp(last, tz="UTC")
    dd = d[mask]
    if len(dd) == 0:
        return dict(sessions=0)
    s = res.info["start"]
    sess = np.nonzero(mask)[0] + s
    o = res.orders[res.orders["session"].isin(sess)]
    n_assets = len(res.tickers)
    liq = o["limited_by"].str.contains("liquidity") if len(o) else pd.Series(dtype=bool)
    w = weight_tracking(res, mkt_close)
    w = w[w.index.isin(dd.index)]
    traded_days = int((dd["gross_traded"] > 0).sum())
    return dict(sessions=int(len(dd)), first=str(dd.index[0].date()), last=str(dd.index[-1].date()),
                nav_min=float(dd["nav"].min()), nav_max=float(dd["nav"].max()),
                **turnover(res, mask),
                order_rows=int(len(o)), liquidity_limited_order_rows=int(liq.sum()),
                liquidity_limited_share_of_order_rows=float(liq.mean()) if len(o) else None,
                liquidity_limited_share_of_ticker_sessions=float(liq.sum() / (len(dd) * n_assets)),
                sessions_with_trades=traded_days,
                sessions_with_a_liquidity_limited_order=int(o.loc[liq, "session"].nunique()) if len(o) else 0,
                cash_weight_mean=float((dd["cash"] / dd["nav"]).mean()),
                wte_risky_mean=float(w["wte_risky"].mean()) if len(w) else None,
                wte_with_cash_mean=float(w["wte_with_cash"].mean()) if len(w) else None)


SUBPERIODS = (("2010-04-01", "2014-12-31"), ("2015-01-01", "2019-12-31"), ("2020-01-01", "2026-06-15"))


def summarize(res, mkt_close: np.ndarray) -> dict:
    """Run summary. Denominators are named. Turnover = gross executed notional / opening pre-trade NAV
    (cash + units x that session's open, before trading); see `turnover` for the deployment-session rule."""
    d = res.daily
    w = weight_tracking(res, mkt_close)
    o = res.orders
    out = dict(scenario_id=res.cfg.scenario_id, name=res.cfg.name, sessions=int(len(d)), nav_start=float(d["nav"].iloc[0]),
               nav_end=float(d["nav"].iloc[-1]), nav_min=float(d["nav"].min()), nav_max=float(d["nav"].max()),
               initial_capital=float(res.cfg.initial_cash))
    out.update(performance(d["nav"]))
    out.update(direct_cost(res))
    out["orders"] = order_summary(res.parents)
    out["demand"] = demand_summary(res.demand)
    out.update(turnover(res))
    out["cash_weight_mean"] = float((d["cash"] / d["nav"]).mean())
    out["wte_risky_mean"] = float(w["wte_risky"].mean()) if len(w) else None
    out["wte_risky_p95"] = float(w["wte_risky"].quantile(0.95)) if len(w) else None
    out["wte_with_cash_mean"] = float(w["wte_with_cash"].mean()) if len(w) else None
    out["order_rows"] = int(len(o))
    out["liquidity_limited_share"] = float(o["limited_by"].str.contains("liquidity").mean()) if len(o) else 0.0
    out["liquidity_limited_share_denominator"] = "order rows"
    out["cash_limited_share"] = float(o["limited_by"].str.contains("cash").mean()) if len(o) else 0.0
    traded_rows = o.loc[o["notional"] != 0, "participation"] if len(o) else pd.Series(dtype=float)
    out["participation_median"] = float(traded_rows.median()) if len(traded_rows) else None
    out["participation_max"] = float(traded_rows.max()) if len(traded_rows) else None
    out["full_period"] = period_summary(res, mkt_close)
    out["subperiods"] = [period_summary(res, mkt_close, a, b) for a, b in SUBPERIODS]
    out = _clean(out)
    undefined = {}
    for k in ("cagr", "ann_vol", "sharpe_rf0", "max_drawdown"):
        if out.get(k) is None:
            undefined[k] = "fewer than one return" if out["n_returns"] == 0 else "zero return variance"
    if out["orders"].get("completion_aggregate") is None:
        undefined["completion_aggregate"] = out["orders"].get("completion_undefined_reason", "no demand originated")
    if out.get("direct_cost_bps_exec") is None:
        undefined["direct_cost_bps_exec"] = "no executed notional"
    out["undefined"] = undefined
    return out


def wte_material(arm_mean: float, ref_mean: float, rel=0.10, floor=1e-4, abs_thr=0.005) -> bool:
    """Materiality of a change in mean WTE. Relative rule when the reference is not near zero,
    absolute rule (0.005 in weight-norm units) when the reference mean is below `floor`."""
    if ref_mean < floor:
        return abs(arm_mean - ref_mean) >= abs_thr
    return abs(arm_mean - ref_mean) / ref_mean >= rel
