"""Experiment configuration registry for protocol revision 3. Building it runs nothing."""
from __future__ import annotations

from .config import CostParams, ScenarioConfig, information_class

AUM = {"10m": 1e7, "100m": 1e8, "1b": 1e9}
STRAT = {"A": "equal_weight_weekly", "C": "momentum_top4"}
OFF = CostParams(enabled=False)


def primary(name, experiment, **kw) -> ScenarioConfig:
    return ScenarioConfig(name=name, experiment=experiment, **kw)


def build() -> list:
    reg = []
    # E1 timing: 2 contract-valid arms + 4 negative controls, legacy-like mechanics held fixed, strategy A at $1M
    for val in ("prior_close", "open", "same_close"):
        for feat in ("lagged", "same_day"):
            reg.append(primary(f"E1_{val}_{feat}", "E1", strategy="equal_weight_weekly", initial_cash=1e6, valuation=val, features=feat,
                               adv_estimator="shares_x_close", max_trade_notional=250_000.0, units="integer", min_trade_notional=1000.0))
    # E2 attribution: strategy x AUM x cost x cap (the cost-off/cap-off cell is the unconstrained reference;
    # the post-hoc comparator is derived from it and is not a replay)
    for s, strat in STRAT.items():
        for a, cash in AUM.items():
            for cost in (0, 1):
                for cap in (0, 1):
                    reg.append(primary(f"E2_{s}_{a}_cost{cost}_cap{cap}", "E2", strategy=strat, initial_cash=cash,
                                       cost=CostParams() if cost else OFF, cap_enabled=bool(cap)))
    # E2b legacy comparator assumptions: cost x min trade x borrowing x units at strategy A, $1M, no cap
    for cost in (0, 1):
        for mt in (0.0, 1000.0):
            for borrow in (True, False):
                for units in ("fractional", "integer"):
                    reg.append(primary(f"E2b_cost{cost}_min{int(mt)}_borrow{int(borrow)}_{units}", "E2b", strategy="equal_weight_weekly",
                                       initial_cash=1e6, cost=CostParams() if cost else OFF, cap_enabled=False,
                                       min_trade_notional=mt, allow_borrowing=borrow, units=units))
    # E3 residual policy (daily targets make replace == cancel, so the weekly strategy carries the cancel contrast)
    for s, a in (("A", "1b"), ("C", "100m"), ("C", "1b")):
        for pol in (("cancel", "legacy_additive") if s == "A" else ("legacy_additive",)):
            reg.append(primary(f"E3_{s}_{a}_{pol}", "E3", strategy=STRAT[s], initial_cash=AUM[a], residual_policy=pol))
    # E4 cash rule
    for s, a in (("A", "10m"), ("C", "100m")):
        reg.append(primary(f"E4_{s}_{a}_legacy_sequential", "E4", strategy=STRAT[s], initial_cash=AUM[a], cash_rule="legacy_sequential"))
    # E5 one-at-a-time sensitivity at C, $1B
    base = dict(strategy="momentum_top4", initial_cash=1e9)
    for k, vals in (("a_bps", (5.0, 30.0)), ("b", (0.5, 1.0)), ("c", (0.0, 2.0)), ("half_spread_bps", (1.0, 2.0))):
        for v in vals:
            reg.append(primary(f"E5_{k}_{v}", "E5", cost=CostParams(**{k: v}), **base))
    for phi in (0.01, 0.10):
        reg.append(primary(f"E5_phi_{phi}", "E5", max_participation=phi, **base))
    names = [c.name for c in reg]
    ids = [c.scenario_id for c in reg]
    assert len(set(names)) == len(names) and len(set(ids)) == len(ids), "scenario names and ids must be unique"
    return reg


def pilot() -> ScenarioConfig:
    return primary("pilot_252", "pilot", strategy="momentum_top4", initial_cash=1e9, max_sessions=252)


def as_records() -> list:
    return [dict(name=c.name, experiment=c.experiment, scenario_id=c.scenario_id, information=information_class(c.valuation, c.features),
                 config=c.to_dict()) for c in build() + [pilot()]]
