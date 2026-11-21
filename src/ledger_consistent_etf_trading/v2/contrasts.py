"""Explicit contrast recipes for the registered experiments, and the matching rule each must satisfy.

A contrast compares an arm with a reference. `allowed` lists every configuration field permitted to
differ; all other fields must be equal, and both runs must share market identity and calendar. The
list is generated from the registry, so "which contrasts" is fixed before any result exists.
tier: "primary" answers the research question; "secondary" supports it; "diagnostic" characterizes
legacy or invalid conventions and is never a candidate for the default.
"""
from __future__ import annotations

from . import registry
from .config import ScenarioConfig, information_class
from .errors import ScenarioInputError

ALWAYS = {"name"}
POSTHOC = "posthoc"          # reference object derived from the unconstrained arm (return-only)


def differing_fields(a: ScenarioConfig, b: ScenarioConfig) -> set:
    da, db = a.to_dict(), b.to_dict()
    out = {k for k in da if k != "cost" and da[k] != db[k]}
    out |= {"cost." + k for k in da["cost"] if da["cost"][k] != db["cost"][k]}
    return out


def check_matched(a: ScenarioConfig, b: ScenarioConfig, allowed: set) -> None:
    extra = differing_fields(a, b) - set(allowed) - ALWAYS
    if extra:
        raise ScenarioInputError(f"contrast {a.name} vs {b.name}: unintended differences in {sorted(extra)}")


def build() -> list:
    reg = {c.name: c for c in registry.build()}
    out = []

    def add(exp, tier, arm, ref, allowed, kind="replay", note=""):
        out.append(dict(experiment=exp, tier=tier, arm=arm, reference=ref, kind=kind, allowed=sorted(allowed), note=note))

    for val in ("prior_close", "open", "same_close"):
        for feat in ("lagged", "same_day"):
            if (val, feat) != ("prior_close", "lagged"):
                valid = information_class(val, feat)["valid"]
                add("E1", "secondary" if valid else "diagnostic", f"E1_{val}_{feat}", "E1_prior_close_lagged", {"valuation", "features"},
                    note="observed-open sensitivity" if valid else "negative control; reported as a difference from the primary arm only")
    for s in ("A", "C"):
        for a in ("10m", "100m", "1b"):
            g = f"E2_{s}_{a}"
            ref = g + "_cost0_cap0"
            add("E2", "primary", g + "_cost1_cap0", ref, {"cost.enabled"}, kind="vs_posthoc", note="stateful cost against post-hoc cost adjustment of the reference")
            add("E2", "primary", g + "_cost1_cap1", ref, {"cost.enabled", "cap_enabled"}, kind="vs_posthoc", note="stateful cost and limit against post-hoc cost adjustment")
            add("E2", "secondary", g + "_cost1_cap0", ref, {"cost.enabled"})
            add("E2", "secondary", g + "_cost0_cap1", ref, {"cap_enabled"})
            add("E2", "secondary", g + "_cost1_cap1", ref, {"cost.enabled", "cap_enabled"})
    for c in reg.values():
        if c.experiment == "E2b" and c.name != "E2b_cost0_min0_borrow1_fractional":
            add("E2b", "diagnostic", c.name, "E2b_cost0_min0_borrow1_fractional", {"cost.enabled", "min_trade_notional", "allow_borrowing", "units"})
    for c in reg.values():
        if c.experiment == "E3":
            s, a = c.name.split("_")[1:3]
            add("E3", "secondary" if c.residual_policy == "cancel" else "diagnostic", c.name, f"E2_{s}_{a}_cost1_cap1", {"residual_policy", "experiment"})
        if c.experiment == "E4":
            s, a = c.name.split("_")[1:3]
            add("E4", "diagnostic", c.name, f"E2_{s}_{a}_cost1_cap1", {"cash_rule", "experiment"})
        if c.experiment == "E5":
            field = "max_participation" if c.name.startswith("E5_phi") else "cost." + c.name[3:].rsplit("_", 1)[0]
            add("E5", "secondary", c.name, "E2_C_1b_cost1_cap1", {field, "experiment"})
    for r in out:
        check_matched(reg[r["arm"]], reg[r["reference"]], set(r["allowed"]))
        if reg[r["arm"]].name == reg[r["reference"]].name:
            raise ScenarioInputError("a contrast needs two different runs")
    return out
