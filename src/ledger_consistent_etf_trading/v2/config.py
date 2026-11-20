from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import math

from . import SPEC_VERSION
from .errors import ScenarioInputError

VALUATIONS = ("prior_close", "open", "same_close")
FEATURE_MODES = ("lagged", "same_day")
ADV_ESTIMATORS = ("mean_dollar", "shares_x_close")
RESIDUAL_POLICIES = ("replace", "cancel", "legacy_additive")
CASH_RULES = ("sells_first_proportional", "legacy_sequential")
UNITS = ("fractional", "integer")
STRATEGIES = ("equal_weight_weekly", "momentum_top4", "tape")


@dataclass(frozen=True)
class CostParams:
    enabled: bool = True
    fees_bps: float = 0.0
    half_spread_bps: float = 0.5
    a_bps: float = 15.0
    b: float = 0.7
    c: float = 1.0
    vol_ref: float = 0.01

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ScenarioInputError("cost.enabled must be a bool")
        for k in ("fees_bps", "half_spread_bps", "a_bps", "b", "c", "vol_ref"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ScenarioInputError(f"cost.{k} must be a finite non-negative number, got {v!r}")
        if not self.vol_ref > 0:
            raise ScenarioInputError("cost.vol_ref must be positive")


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    strategy: str = "equal_weight_weekly"
    initial_cash: float = 1_000_000.0
    valuation: str = "prior_close"
    features: str = "lagged"
    adv_estimator: str = "mean_dollar"
    cap_enabled: bool = True
    max_participation: float = 0.05
    max_trade_notional: Optional[float] = None      # absolute cap M; legacy diagnostics only
    cost: CostParams = field(default_factory=CostParams)
    residual_policy: str = "replace"
    cash_rule: str = "sells_first_proportional"
    allow_borrowing: bool = False
    units: str = "fractional"
    min_trade_notional: float = 0.0
    eval_start: str = "2010-03-31"                  # first session of the evaluation calendar
    eval_end: Optional[str] = None
    max_sessions: Optional[int] = None              # pilot / budget control
    momentum_lookback: int = 20
    momentum_top: int = 4
    experiment: str = "unassigned"

    def __post_init__(self):
        checks = ((self.valuation, VALUATIONS), (self.features, FEATURE_MODES), (self.adv_estimator, ADV_ESTIMATORS),
                  (self.residual_policy, RESIDUAL_POLICIES), (self.cash_rule, CASH_RULES), (self.units, UNITS),
                  (self.strategy, STRATEGIES))
        for value, allowed in checks:
            if value not in allowed:
                raise ScenarioInputError(f"invalid option {value!r}; allowed {allowed}")
        num = lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
        posint = lambda v: not isinstance(v, bool) and isinstance(v, int) and v > 0
        if not (num(self.initial_cash) and self.initial_cash > 0):
            raise ScenarioInputError("initial_cash must be finite and positive")
        if not (num(self.max_participation) and 0 < self.max_participation <= 1):
            raise ScenarioInputError("max_participation must be in (0, 1]")
        if not (num(self.min_trade_notional) and self.min_trade_notional >= 0):
            raise ScenarioInputError("min_trade_notional must be finite and non-negative")
        if self.max_trade_notional is not None and not (num(self.max_trade_notional) and self.max_trade_notional > 0):
            raise ScenarioInputError("max_trade_notional must be finite and positive, or None")
        if not (posint(self.momentum_lookback) and posint(self.momentum_top)):
            raise ScenarioInputError("momentum_lookback and momentum_top must be positive integers")
        if self.max_sessions is not None and not posint(self.max_sessions):
            raise ScenarioInputError("max_sessions must be a positive integer or None")
        if not isinstance(self.cost, CostParams) or not isinstance(self.cap_enabled, bool) or not isinstance(self.allow_borrowing, bool):
            raise ScenarioInputError("cost must be CostParams; cap_enabled and allow_borrowing must be bools")
        for k in ("name", "experiment", "eval_start"):
            if not isinstance(getattr(self, k), str) or not getattr(self, k):
                raise ScenarioInputError(f"{k} must be a non-empty string")

    def to_dict(self) -> dict:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps({"spec": SPEC_VERSION, "config": self.to_dict()}, sort_keys=True, separators=(",", ":"))

    @property
    def scenario_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def with_(self, **kw) -> "ScenarioConfig":
        return replace(self, **kw)

    @staticmethod
    def from_dict(d: dict) -> "ScenarioConfig":
        d = dict(d)
        cost = d.pop("cost", {})
        return ScenarioConfig(cost=CostParams(**cost), **d)


def information_class(valuation: str, features: str) -> dict:
    """Classify a timing configuration against the V2 information contract.

    Decision event for ``prior_close``: after session d-1 closes, before d opens.
    Decision event for ``open``: immediately after the session-d open is observed.
    Fills are always modelled at the session-d open.
    """
    val_ok = valuation in ("prior_close", "open")
    feat_ok = features == "lagged"
    if valuation == "prior_close" and feat_ok:
        role, note = "primary", "all inputs available before the open; quantities fixed pre-open"
    elif valuation == "open" and feat_ok:
        role, note = "sensitivity", "inputs available at the decision event, but the order is sized with the price it then fills at: an execution approximation no pre-open order can achieve"
    else:
        why = []
        if not val_ok:
            why.append("pre-trade valuation uses the session-d close")
        if not feat_ok:
            why.append("ADV and volatility windows include session d")
        role, note = "negative_control", "; ".join(why)
    return {"valuation": valuation, "features": features, "valid": val_ok and feat_ok, "role": role, "note": note}
