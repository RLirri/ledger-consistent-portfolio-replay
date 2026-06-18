from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class CapacityConfig:
    max_participation: float        # e.g., 0.05 (5% of ADV notional)
    max_trade_notional: float       # absolute $ cap per asset per day
    carry_unfilled: bool = True


def clip_notional(desired_notional: float, adv_notional: float, cfg: CapacityConfig) -> tuple[float, float]:
    # Returns (executed_notional, remaining_notional).
    if adv_notional is None or adv_notional <= 0:
        # If ADV missing, fall back to absolute cap only
        cap = cfg.max_trade_notional
    else:
        cap_adv = cfg.max_participation * adv_notional
        cap = min(cap_adv, cfg.max_trade_notional)

    executed = max(min(desired_notional, cap), -cap)
    remaining = desired_notional - executed
    return executed, remaining


def notional_to_shares(notional: float, price: float) -> int:
    # Convert notional to integer shares using floor toward zero.
    if price <= 0 or notional == 0:
        return 0
    shares = notional / price
    # floor toward zero:
    return int(math.trunc(shares))
