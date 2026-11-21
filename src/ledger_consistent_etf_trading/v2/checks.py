"""Automatic per-run scientific validation.

`validate` recomputes every accounting identity from the run's tapes and the market data, without
calling the engine. It is applied to the in-memory result before anything is published and again to
the files read back from disk. A run with any failed check is not scientifically valid.

Tolerances are stated in the unit of the quantity checked:
  dollars: 1e-9 x initial capital (x10 for cumulative paths)
  units:   dollar tolerance / smallest price used
  ratios:  1e-9 relative
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from types import SimpleNamespace

from .config import ScenarioConfig, information_class
from .data import FEATURE_WINDOW, Features, Market, build_features
from .errors import RunValidationError
from .strategies import build_targets

REL = 1e-9
CLOSE_REASONS = {"completed", "replaced", "cancelled_eod", "end_of_sample"}

# Diagnostic settings and the experiments in which the protocol authorizes them.
DIAGNOSTIC_RULES = (
    ("negative-control timing", lambda c: not information_class(c.valuation, c.features)["valid"], {"E1"}),
    ("legacy additive carry", lambda c: c.residual_policy == "legacy_additive", {"E3"}),
    ("legacy sequential cash", lambda c: c.cash_rule == "legacy_sequential", {"E4"}),
    ("borrowing", lambda c: c.allow_borrowing, {"E2b"}),
    ("absolute dollar cap", lambda c: c.max_trade_notional is not None, {"E1"}),
    ("integer adjusted units", lambda c: c.units == "integer", {"E1", "E2b"}),
    ("minimum trade floor", lambda c: c.min_trade_notional > 0, {"E1", "E2b"}),
    ("legacy ADV estimator", lambda c: c.adv_estimator == "shares_x_close", {"E1"}),
)


def diagnostic_exceptions(cfg: ScenarioConfig) -> tuple:
    """(authorized, unauthorized) lists of diagnostic settings used by this configuration."""
    ok, bad = [], []
    for name, used, where in DIAGNOSTIC_RULES:
        if used(cfg):
            (ok if cfg.experiment in where else bad).append(f"{name} (allowed in {sorted(where)}, experiment is {cfg.experiment!r})")
    return ok, bad


@dataclass
class Report:
    scenario_id: str
    checks: list = field(default_factory=list)
    diagnostics_authorized: list = field(default_factory=list)

    def add(self, name, passed, value=None, tolerance=None, detail=""):
        v = None if value is None else (float(value) if np.isfinite(value) else str(value))
        self.checks.append(dict(check=name, passed=bool(passed), value=v, tolerance=tolerance, detail=detail))

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["passed"] for c in self.checks)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if not c["passed"]]

    def to_dict(self) -> dict:
        return dict(scenario_id=self.scenario_id, valid=self.ok, n_checks=len(self.checks), n_failed=len(self.failures),
                    diagnostics_authorized=self.diagnostics_authorized, checks=self.checks)

    def require(self):
        if not self.ok:
            raise RunValidationError("run failed validation: " + "; ".join(f"{c['check']} ({c['detail'] or c['value']})" for c in self.failures))


def _maxabs(a) -> float:
    a = np.asarray(a, float)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a))) if np.all(np.isfinite(a)) else float("nan")


def _model_cost(cfg, notional, adv, vol):
    c = cfg.cost
    if not c.enabled:
        return np.zeros_like(notional)
    x = np.minimum(notional / adv, 1.0)
    return notional * (c.fees_bps + c.half_spread_bps + c.a_bps * x ** c.b * (vol / c.vol_ref) ** c.c) / 1e4


def validate(cfg: ScenarioConfig, info: dict, daily: pd.DataFrame, orders: pd.DataFrame, parents: pd.DataFrame, demand: pd.DataFrame,
             positions: np.ndarray, target_eff: np.ndarray, mkt: Market, feats: Features, production: bool = True) -> Report:
    rep = Report(cfg.scenario_id)
    s, e = int(info["start"]), int(info["end"])
    T, N = e - s, len(mkt.tickers)
    col = {t: i for i, t in enumerate(mkt.tickers)}
    tol = 1e-9 * max(1.0, cfg.initial_cash)
    O, C = mkt.open[s:e], mkt.close[s:e]
    held = np.abs(positions) > 0

    # configuration and timing metadata
    authorized, unauthorized = diagnostic_exceptions(cfg)
    rep.diagnostics_authorized = authorized
    if production:
        rep.add("diagnostic settings authorized for this experiment", not unauthorized, detail="; ".join(unauthorized))
        rep.add("features bound to the run's market data", info.get("features_binding") == "bound" and info.get("features_market_id") == mkt.identity(),
                detail=str(info.get("features_binding")))
    rep.add("information class matches configuration", info.get("information") == information_class(cfg.valuation, cfg.features))
    rep.add("feature mode and estimator match configuration", (feats.mode, feats.estimator) == (cfg.features, cfg.adv_estimator))
    rep.add("market identity matches", info.get("market_id") == mkt.identity())
    shapes = len(daily) == T and positions.shape == (T, N) and target_eff.shape == (T, N)
    rep.add("tape dimensions match the evaluation calendar", shapes, detail=f"{len(daily)} daily rows, positions {positions.shape}, expected {(T, N)}")
    if not shapes:
        return rep
    rep.add("daily dates equal the market sessions", bool(pd.DatetimeIndex(daily.index).equals(mkt.dates[s:e])))

    # finiteness
    dnum = daily[["nav", "cash", "holdings_value", "cost", "gross_traded", "nav_open"]].to_numpy(float)
    rep.add("daily tape finite", bool(np.all(np.isfinite(dnum))))
    rep.add("NAV positive", bool(np.all(daily["nav"].to_numpy(float) > 0)))
    onum = orders[["intent_units", "liq_eligible_units", "cash_eligible_units", "filled_units", "fill_price", "notional", "cost"]].to_numpy(float)
    pnum = parents[["q", "p_D", "p_A", "filled", "fill_value", "cost", "p_H", "unfilled", "demand_gap", "demand_new_value", "demand_executed_value"]].to_numpy(float)
    rep.add("order tape finite", bool(np.all(np.isfinite(onum))))
    rep.add("parent tape finite, including every unfilled-opportunity mark", bool(np.all(np.isfinite(pnum))))
    rep.add("positions finite", bool(np.all(np.isfinite(positions))))
    rep.add("demand ledger finite", bool(np.all(np.isfinite(demand[["units", "basis_price", "value"]].to_numpy(float)))))
    if not all(c["passed"] for c in rep.checks if "finite" in c["check"]):
        return rep

    sess = orders["session"].to_numpy(int)
    k = sess - s
    ti = np.array([col.get(t, -1) for t in orders["ticker"]], int)
    in_range = bool(np.all((k >= 0) & (k < T)) and np.all(ti >= 0)) if len(orders) else True
    rep.add("order sessions and tickers lie in the run", in_range)
    if not in_range:
        return rep
    fu, fp, nt, oc = (orders[c].to_numpy(float) for c in ("filled_units", "fill_price", "notional", "cost"))
    pmin = float(np.min(O[held])) if held.any() else (float(fp.min()) if len(fp) else 1.0)
    utol = tol / max(pmin, 1e-12)

    # fills: price, notional, dates
    rep.add("fill price equals the session open", _maxabs(fp - O[k, ti]) <= REL * max(1.0, _maxabs(fp)), _maxabs(fp - O[k, ti]), REL)
    rep.add("notional equals units x price", _maxabs(nt - fu * fp) <= tol, _maxabs(nt - fu * fp), tol)
    rep.add("order dates equal the market sessions", bool((pd.to_datetime(orders["date"], utc=True).to_numpy() == mkt.dates[sess].to_numpy()).all()) if len(orders) else True)

    # cash conservation, cost charged once
    flow = np.zeros(T); np.add.at(flow, k, nt + oc)
    cash_path = cfg.initial_cash - np.cumsum(flow)
    err = _maxabs(cash_path - daily["cash"].to_numpy(float))
    rep.add("cash conservation (every session)", err <= 10 * tol, err, 10 * tol)
    adv, vol = feats.adv[sess, ti], feats.vol[sess, ti]
    traded = np.abs(nt) > 0
    if cfg.cost.enabled:
        good = bool(np.all(np.isfinite(adv[traded]) & (adv[traded] > 0) & np.isfinite(vol[traded]) & (vol[traded] > 0)))
        rep.add("liquidity inputs valid for every costed fill", good)
        model = np.zeros_like(nt)
        if good and traded.any():
            model[traded] = _model_cost(cfg, np.abs(nt[traded]), adv[traded], vol[traded])
        cerr = _maxabs(model - oc)
        rep.add("cost equals the independent model recalculation", cerr <= max(tol, REL * _maxabs(oc)), cerr, tol)
    else:
        rep.add("no cost charged when costs are disabled", _maxabs(oc) == 0.0, _maxabs(oc), 0.0)
    rep.add("costs non-negative", bool(np.all(oc >= 0)))
    day_cost = np.zeros(T); np.add.at(day_cost, k, oc)
    rep.add("cost charged exactly once (orders = daily = parents)",
            _maxabs(day_cost - daily["cost"].to_numpy(float)) <= tol and abs(oc.sum() - parents["cost"].sum()) <= 10 * tol,
            max(_maxabs(day_cost - daily["cost"].to_numpy(float)), abs(oc.sum() - parents["cost"].sum())), 10 * tol)
    day_gross = np.zeros(T); np.add.at(day_gross, k, np.abs(nt))
    rep.add("gross traded equals summed fills", _maxabs(day_gross - daily["gross_traded"].to_numpy(float)) <= 10 * tol)

    # positions and valuation
    dpos = np.zeros((T, N)); np.add.at(dpos, (k, ti), fu)
    perr = _maxabs(np.cumsum(dpos, axis=0) - positions)
    rep.add("position reconciliation", perr <= 10 * utol * max(1, T ** 0.5), perr, 10 * utol)
    marks_ok = bool(np.all(np.isfinite(C[held]) & (C[held] > 0)))
    rep.add("close marks exist for every held position", marks_ok)
    hv = (np.where(held, positions * np.where(held, C, 0.0), 0.0)).sum(axis=1)
    rep.add("holdings value at current close marks", _maxabs(hv - daily["holdings_value"].to_numpy(float)) <= 10 * tol, _maxabs(hv - daily["holdings_value"].to_numpy(float)), 10 * tol)
    rep.add("NAV = cash + holdings", _maxabs(daily["cash"].to_numpy(float) + hv - daily["nav"].to_numpy(float)) <= 10 * tol)
    prev_pos = np.vstack([np.zeros((1, N)), positions[:-1]])
    prev_cash = np.concatenate([[cfg.initial_cash], daily["cash"].to_numpy(float)[:-1]])
    ph = np.abs(prev_pos) > 0
    nav_open = prev_cash + np.where(ph, prev_pos * np.where(ph, O, 0.0), 0.0).sum(axis=1)
    rep.add("opening pre-trade NAV", _maxabs(nav_open - daily["nav_open"].to_numpy(float)) <= 10 * tol)

    # feasibility
    mincash = float(daily["cash"].min())
    rep.add("no unauthorized borrowing", cfg.allow_borrowing or mincash >= -tol, mincash, -tol)
    rep.add("no short positions", float(positions.min()) >= -utol, float(positions.min()), -utol)
    iu, lu, cu = (orders[c].to_numpy(float) for c in ("intent_units", "liq_eligible_units", "cash_eligible_units"))
    hier = np.all(np.abs(fu) <= np.abs(cu) + utol) and np.all(np.abs(cu) <= np.abs(lu) + utol) and np.all(np.abs(lu) <= np.abs(iu) + utol)
    rep.add("gate hierarchy: fill <= cash-eligible <= liquidity-eligible <= intent", bool(hier))
    rep.add("fills have the sign of the intent", bool(np.all(fu * iu >= 0)))
    if cfg.cap_enabled:
        cap_ok = bool(np.all(np.isfinite(adv[traded]) & (adv[traded] > 0)))
        lim = cfg.max_participation * adv
        viol = np.abs(nt)[traded] - lim[traded] * (1 + REL) if cap_ok else np.array([np.inf])
        rep.add("liquidity-budget compliance", cap_ok and bool(np.all(viol <= tol)), float(np.max(viol)) if viol.size else 0.0, tol)
    if cfg.max_trade_notional is not None:
        rep.add("absolute cap compliance", bool(np.all(np.abs(nt) <= cfg.max_trade_notional * (1 + REL) + tol)))
    if cfg.min_trade_notional > 0:
        rep.add("executed fills respect the minimum-trade floor", bool(np.all(np.abs(nt)[traded] >= cfg.min_trade_notional * (1 - 1e-9))))
    if cfg.units == "integer":
        rep.add("integer units", bool(np.all(fu == np.round(fu))))

    # parents and order lineage
    p = parents.set_index("parent_id")
    rep.add("parent ids unique", bool(p.index.is_unique))
    linked = bool(orders["parent_id"].isin(p.index).all()) if len(orders) else True
    rep.add("every order belongs to one parent", linked)
    if linked and p.index.is_unique:
        if len(orders):
            op = p.loc[orders["parent_id"]]
            rep.add("order and parent agree on the asset", bool((op["ticker"].to_numpy() == orders["ticker"].to_numpy()).all()))
            rep.add("order sessions lie within the parent's life", bool(((sess >= op["open_session"].to_numpy()) & (sess <= op["close_session"].to_numpy())).all()))
            agg = orders.assign(v=fu * fp).groupby("parent_id").agg(f=("filled_units", "sum"), v=("v", "sum"), c=("cost", "sum")).reindex(p.index, fill_value=0.0)
        else:
            agg = pd.DataFrame(0.0, index=p.index, columns=["f", "v", "c"])
        rep.add("parent fills, value and cost equal their orders",
                _maxabs(agg["f"] - p["filled"]) <= 10 * utol and _maxabs(agg["v"] - p["fill_value"]) <= 10 * tol and _maxabs(agg["c"] - p["cost"]) <= 10 * tol,
                max(_maxabs(agg["v"] - p["fill_value"]), _maxabs(agg["c"] - p["cost"])), 10 * tol)
    rep.add("parent quantity = filled + unfilled", _maxabs(p["q"] - p["filled"] - p["unfilled"]) <= 10 * utol, _maxabs(p["q"] - p["filled"] - p["unfilled"]), 10 * utol)
    rep.add("parent life and close reason valid", bool((p["close_session"] >= p["open_session"]).all() and p["close_reason"].isin(CLOSE_REASONS).all()))
    rep.add("parent prices and marks positive", bool((p[["p_D", "p_A", "p_H"]].to_numpy(float) > 0).all()))
    tsess = set(int(x) for x in info.get("target_sessions", []))
    rep.add("parents open only at target events", set(p["open_session"].astype(int)) <= tsess)
    te = daily["target_event"].to_numpy(bool)
    rep.add("target-event flags equal the target sessions", set((np.nonzero(te)[0] + s).tolist()) == {t for t in tsess if s <= t < e})
    if cfg.residual_policy == "cancel":
        rep.add("cancel policy: orders only on target sessions, all closed the same session",
                set(sess.tolist()) <= tsess and bool((p["close_session"] == p["open_session"]).all()))

    # demand lineage and completion
    dm = metrics.demand_summary(demand)
    scale = max(1.0, dm["originated_value"])
    rep.add("demand lineage: originated = executed + withdrawn + outstanding", abs(dm["lineage_identity_residual"]) <= REL * scale + tol, dm["lineage_identity_residual"], REL * scale + tol)
    rep.add("parents and demand ledger agree on originated and executed demand",
            abs(parents["demand_new_value"].sum() - dm["originated_value"]) <= REL * scale + tol and abs(parents["demand_executed_value"].sum() - dm["executed_value"]) <= REL * scale + tol)
    rep.add("demand ledger events and units valid", bool(demand["event"].isin({"originated", "executed", "withdrawn", "overshoot", "outstanding_end"}).all() and (demand["units"] > 0).all() and (demand["basis_price"] > 0).all()))
    comp = dm["executed_share"]
    rep.add("completion is a proportion in [0, 1] or undefined", comp is None or (-REL <= comp <= 1 + REL), comp if comp is not None else 0.0)
    if cfg.residual_policy != "legacy_additive":
        rep.add("no overshoot outside the legacy additive diagnostic", dm["overshoot_value"] <= tol, dm["overshoot_value"], tol)

    # --- marks, lineage and targets recomputed from external prices and the tapes (Sprint 3E, D02) ---
    def sizing_price(d, i):                                  # the price a target event at session d sizes asset i with
        return mkt.close[d - 1, i] if cfg.valuation == "prior_close" else mkt.open[d, i]

    close = lambda a, b: bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= REL * max(1.0, abs(b)))
    pi = np.array([col.get(t, -1) for t in parents["ticker"]], int)
    pos_, pcs = parents["open_session"].to_numpy(int), parents["close_session"].to_numpy(int)
    life_ok = bool(np.all(pi >= 0) and np.all((pos_ >= s) & (pos_ <= pcs) & (pcs < e))) if len(parents) else True
    rep.add("parent sessions and assets lie in the run", life_ok)
    if life_ok and len(parents):
        bad_d = [int(r) for r in range(len(parents)) if not close(parents["p_D"].iat[r], sizing_price(pos_[r], pi[r]))]
        bad_a = [int(r) for r in range(len(parents)) if not close(parents["p_A"].iat[r], mkt.open[pos_[r], pi[r]])]
        bad_h = []
        for r in range(len(parents)):
            why, cs_, i = parents["close_reason"].iat[r], pcs[r], pi[r]
            exp = {"replaced": sizing_price(cs_, i), "cancelled_eod": mkt.close[cs_, i], "completed": mkt.open[cs_, i],
                   "end_of_sample": mkt.close[e - 1, i]}.get(why, np.nan)
            if not close(parents["p_H"].iat[r], exp) or (why == "end_of_sample" and cs_ != e - 1):
                bad_h.append(int(parents["parent_id"].iat[r]))
        rep.add("parent decision price equals the sizing price of its target event", not bad_d, len(bad_d), 0, f"rows {bad_d[:5]}")
        rep.add("parent arrival price equals the open of its target session", not bad_a, len(bad_a), 0, f"rows {bad_a[:5]}")
        rep.add("parent horizon mark equals the price implied by its close reason and session", not bad_h, len(bad_h), 0, f"parents {bad_h[:5]}")

    # demand gap of every target event, recomputed from targets, prior holdings and cash
    gap_bad, navpre_bad = [], []
    pgap = {(int(o), t): g for o, t, g in zip(parents["open_session"], parents["ticker"], parents["demand_gap"])}
    cash_arr = daily["cash"].to_numpy(float)
    for d in sorted(t for t in tsess if s <= t < e):
        k_ = d - s
        hp = positions[k_ - 1] if k_ > 0 else np.zeros(N)
        cp = cash_arr[k_ - 1] if k_ > 0 else cfg.initial_cash
        if cfg.valuation == "prior_close":
            Pv = Ps = mkt.close[d - 1]
        elif cfg.valuation == "open":
            Pv = Ps = mkt.open[d]
        else:
            Pv, Ps = mkt.close[d], mkt.open[d]
        w = target_eff[k_]
        if np.isnan(w).any():
            gap_bad.append((d, "target missing")); continue
        nav_pre = cp + float(hp @ np.where(hp != 0, Pv, 0.0))
        if abs(nav_pre - float(daily["nav_pretrade"].iat[k_])) > 10 * tol:
            navpre_bad.append(d)
        desired = np.zeros(N); pt = w > 0
        desired[pt] = w[pt] * nav_pre / Ps[pt]
        g = desired - hp
        if cfg.units == "integer":
            g = np.trunc(g)
        g[np.abs(g) < 1e-12] = 0.0
        for i, t in enumerate(mkt.tickers):
            have = pgap.get((d, t), 0.0)
            if not abs(have - g[i]) <= utol + REL * max(1.0, abs(g[i])):
                gap_bad.append((d, t))
    rep.add("pre-trade NAV of every target event recomputed", not navpre_bad, len(navpre_bad), 0, f"sessions {navpre_bad[:5]}")
    rep.add("demand gap of every target event equals target units minus prior holdings", not gap_bad, len(gap_bad), 0, f"{gap_bad[:5]}")
    if cfg.residual_policy != "legacy_additive":
        rep.add("order quantity equals the demand gap", _maxabs(parents["q"] - parents["demand_gap"]) <= utol, _maxabs(parents["q"] - parents["demand_gap"]), utol)

    # demand ledger row by row: units x basis, origin prices, FIFO service, newest-first withdrawal, cohort conservation
    du, db, dv = (demand[c].to_numpy(float) for c in ("units", "basis_price", "value"))
    rep.add("demand value equals units x basis price", _maxabs(dv - du * db) <= tol + REL * _maxabs(dv), _maxabs(dv - du * db), tol)
    dsess, dorig = demand["session"].to_numpy(int), demand["origin_session"].to_numpy(int)
    dti = np.array([col.get(t, -1) for t in demand["ticker"]], int)
    dev = demand["event"].to_numpy()
    rows_ok = bool(np.all(dti >= 0) and np.all((dsess >= s) & (dsess < e)) and np.all(dorig <= dsess) and np.all(dorig >= s)) if len(demand) else True
    rep.add("demand ledger sessions, origins and assets lie in the run", rows_ok)
    if rows_ok:
        basis_bad, order_bad = [], []
        queue = {i: [] for i in range(N)}                    # per asset: [origin, remaining units], oldest first
        for r in range(len(demand)):
            i, ev, u, o, d = dti[r], dev[r], du[r], dorig[r], dsess[r]
            exp_basis = mkt.open[d, i] if ev == "overshoot" else sizing_price(o, i)
            if not close(db[r], exp_basis) or (ev in ("originated", "overshoot") and o != d) or (ev != "overshoot" and o not in tsess):
                basis_bad.append(r)
            q_ = queue[i]
            if ev == "originated":
                # A new cohort does not retire live older demand. The comparison
                # tolerance is not a demand-expiry threshold; only engine-scale
                # numerical zero may be removed here.
                while q_ and q_[-1][1] <= 1e-12:
                    q_.pop()
                if q_ and q_[-1][0] >= o:
                    order_bad.append(r)
                q_.append([o, u])
            elif ev == "executed":                           # must serve the oldest outstanding cohort
                while q_ and q_[0][0] != o and q_[0][1] <= utol:      # rounding dust left in an exhausted cohort is not demand
                    q_.pop(0)
                if not q_ or q_[0][0] != o or u > q_[0][1] + utol:
                    order_bad.append(r)
                else:
                    q_[0][1] -= u
                    if q_[0][1] <= 1e-12:
                        q_.pop(0)
            elif ev == "withdrawn":                          # must withdraw the newest outstanding cohort
                while q_ and q_[-1][0] != o and q_[-1][1] <= utol:
                    q_.pop()
                if not q_ or q_[-1][0] != o or u > q_[-1][1] + utol:
                    order_bad.append(r)
                else:
                    q_[-1][1] -= u
                    if q_[-1][1] <= 1e-12:
                        q_.pop()
            elif ev == "outstanding_end":
                hit = [c for c in q_ if c[0] == o]
                if d != e - 1 or not hit or abs(hit[0][1] - u) > utol:
                    order_bad.append(r)
                else:
                    q_.remove(hit[0])
        left = [(mkt.tickers[i], c) for i in range(N) for c in queue[i] if c[1] > utol]
        rep.add("demand basis and origin equal the sizing price of the originating event", not basis_bad, len(basis_bad), 0, f"rows {basis_bad[:5]}")
        rep.add("demand served oldest-first, withdrawn newest-first, within cohort size", not order_bad, len(order_bad), 0, f"rows {order_bad[:5]}")
        rep.add("every cohort conserved: originated = executed + withdrawn + outstanding (units)", not left, len(left), 0, f"{left[:3]}")
        served = np.zeros((T, N)); m_ = np.isin(dev, ["executed", "overshoot"])
        np.add.at(served, (dsess[m_] - s, dti[m_]), du[m_])
        filled_abs = np.zeros((T, N)); np.add.at(filled_abs, (k, ti), np.abs(fu))
        rep.add("executed plus overshoot demand equals the fills of each asset-session", _maxabs(served - filled_abs) <= 10 * utol, _maxabs(served - filled_abs), 10 * utol)
        for ev, colname in (("originated", "demand_new"), ("withdrawn", "demand_withdrawn")):
            led = np.zeros((T, N)); m_ = dev == ev
            np.add.at(led, (dsess[m_] - s, dti[m_]), du[m_])
            par = np.zeros((T, N))
            if life_ok and len(parents):
                np.add.at(par, (pos_ - s, pi), parents[colname].to_numpy(float))
            if ev == "originated":
                rep.add("originated demand per target event agrees between ledger and parents", _maxabs(led - par) <= 10 * utol, _maxabs(led - par), 10 * utol)
            else:                                            # a withdrawal can occur where no order is opened (gap exactly zero)
                rep.add("withdrawn demand in parents never exceeds the ledger", bool(np.all(par <= led + 10 * utol)))

    # features and strategy targets, independently recomputed
    if feats.market_id is not None or production:
        ref = build_features(mkt, cfg.features, cfg.adv_estimator, FEATURE_WINDOW)
        same = feats.window == FEATURE_WINDOW and np.array_equal(ref.vol, feats.vol, equal_nan=True) and np.array_equal(ref.adv, feats.adv, equal_nan=True)
        rep.add("features equal an independent recomputation with the protocol window", bool(same), detail=f"window {feats.window}")
    if production:
        strat = info.get("targets_source") == "strategy" and cfg.strategy != "tape"
        rep.add("targets were generated by the registered strategy", strat, detail=str(info.get("targets_source")))
        if strat:
            want = build_targets(mkt, cfg, s, e)
            exp = np.full((T, N), np.nan); cur = None
            for d in range(s, e):
                cur = want.get(d, cur)
                if cur is not None:
                    exp[d - s] = cur
            rep.add("target tape equals the strategy's targets recomputed from market data",
                    set(want) == tsess and bool(np.array_equal(np.isnan(exp), np.isnan(target_eff)) and np.allclose(np.nan_to_num(exp), np.nan_to_num(target_eff), rtol=0, atol=1e-12)))

    # targets
    act = ~np.isnan(target_eff).any(axis=1)
    tw = target_eff[act]
    rep.add("target weights valid", bool(np.all(tw >= 0) and np.all(tw.sum(axis=1) <= 1 + 1e-9)) if len(tw) else True)
    change = np.nonzero(np.any(np.nan_to_num(target_eff[1:], nan=-1.0) != np.nan_to_num(target_eff[:-1], nan=-1.0), axis=1))[0] + 1
    rep.add("target weights change only at target events", bool(np.all(te[change])) if len(change) else True)
    return rep


def validate_result(res, mkt: Market, feats: Features, production: bool = True) -> Report:
    return validate(res.cfg, res.info, res.daily, res.orders, res.parents, res.demand, res.positions, res.target_eff, mkt, feats, production)


def load_run(run_dir: Path) -> dict:
    """Read a persisted run back from disk, keeping the row and column labels of every tape."""
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    daily = pd.read_csv(run_dir / "daily.csv", parse_dates=["date"], float_precision="round_trip").set_index("date")
    orders = pd.read_csv(run_dir / "orders.csv", keep_default_na=False, na_values=["nan", "NaN"], float_precision="round_trip")
    for c in orders.columns:
        if c not in ("date", "ticker", "limited_by"):
            orders[c] = pd.to_numeric(orders[c])
    parents = pd.read_csv(run_dir / "parents.csv", keep_default_na=False, na_values=["nan", "NaN"], float_precision="round_trip")
    for c in parents.columns:
        if c not in ("ticker", "close_reason"):
            parents[c] = pd.to_numeric(parents[c])
    demand = pd.read_csv(run_dir / "demand.csv", float_precision="round_trip")
    pos_df = pd.read_csv(run_dir / "positions.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    tgt_df = pd.read_csv(run_dir / "target_weights.csv", index_col=0, parse_dates=True, float_precision="round_trip")
    return dict(manifest=man, cfg=ScenarioConfig.from_dict(man["config"]), info=man["run_info"], daily=daily, orders=orders, parents=parents,
                demand=demand, positions=pos_df.to_numpy(float), target_eff=tgt_df.to_numpy(float), positions_frame=pos_df, targets_frame=tgt_df,
                metrics=json.loads((run_dir / "metrics.json").read_text()))


def _same(a, b, path="") -> list:
    """Paths at which two JSON-like values differ (numbers compared to 1e-9 relative)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += _same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [path + "[len]"]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in _same(u, v, f"{path}[{i}]")]
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if num(a) and num(b):
        return [] if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)) else [path]
    return [] if a == b else [path]


def validate_saved(run_dir: Path, mkt: Market, feats: Features, production: bool = True) -> Report:
    """Re-validate from the files on disk; check tape labels; recompute every saved metric from the tapes."""
    r = load_run(run_dir)
    rep = validate(r["cfg"], r["info"], r["daily"], r["orders"], r["parents"], r["demand"], r["positions"], r["target_eff"], mkt, feats, production)
    s, e = int(r["info"]["start"]), int(r["info"]["end"])
    labels = True
    for f in (r["positions_frame"], r["targets_frame"]):
        idx = pd.DatetimeIndex(f.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        labels &= list(f.columns) == list(mkt.tickers) and len(idx) == e - s and bool(idx.equals(mkt.dates[s:e]))
    rep.add("saved position and target tapes carry the run's asset and session labels", bool(labels))
    m = r["metrics"]
    rep.add("saved metrics contain no non-finite number", "NaN" not in json.dumps(m) and "Infinity" not in json.dumps(m))
    rep.add("saved metrics: scenario id", m.get("scenario_id") == r["cfg"].scenario_id == r["manifest"]["scenario_id"])
    try:
        again = metrics.summarize(SimpleNamespace(cfg=r["cfg"], daily=r["daily"], orders=r["orders"], parents=r["parents"], demand=r["demand"],
                                                  positions=r["positions"], target_eff=r["target_eff"], info=r["info"],
                                                  dates=mkt.dates[s:e], tickers=list(mkt.tickers)), mkt.close)
        diff = _same(json.loads(json.dumps(again)), m)
    except Exception as exc:                                 # a tape that cannot be summarized cannot support its saved metrics
        diff = [f"recomputation failed: {type(exc).__name__}: {exc}"]
    rep.add("every saved metric equals its recomputation from the saved tapes", not diff, len(diff), 0, "; ".join(sorted(diff)[:6]))
    return rep
