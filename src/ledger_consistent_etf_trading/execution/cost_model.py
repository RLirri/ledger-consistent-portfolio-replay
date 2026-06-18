from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CostConfig:
    fees_bps: float = 0.0
    half_spread_bps: float = 0.5

    # slippage_bps = a * (trade/ADV)^b * (vol/vol_ref)^c
    slippage_a_bps: float = 15.0
    slippage_b: float = 0.7
    slippage_c: float = 1.0

    vol_ref: float = 0.01  # e.g. 1% daily vol


def compute_costs(
    trade_notional_abs: float,
    adv_notional: float,
    vol: float,
    cfg: CostConfig,
) -> dict:
    """
    Returns a dict with total cost and breakdown in dollars + bps.
    Deterministic and explainable.
    """
    if trade_notional_abs <= 0:
        return {
            "total_cost_$": 0.0,
            "fee_cost_$": 0.0,
            "spread_cost_$": 0.0,
            "slippage_cost_$": 0.0,
            "total_bps": 0.0,
            "slip_bps": 0.0,
        }

    # Participation rate x = N / ADV$
    if adv_notional is None or adv_notional <= 0:
        x = 0.0
    else:
        x = min(trade_notional_abs / adv_notional, 1.0)

    vol_ref = cfg.vol_ref if cfg.vol_ref and cfg.vol_ref > 0 else 0.01
    vol_ratio = (vol / vol_ref) if (vol is not None and vol_ref > 0) else 1.0
    if vol_ratio <= 0:
        vol_ratio = 1.0

    slip_bps = cfg.slippage_a_bps * (x ** cfg.slippage_b) * (vol_ratio ** cfg.slippage_c)

    total_bps = cfg.fees_bps + cfg.half_spread_bps + slip_bps

    fee_cost = trade_notional_abs * (cfg.fees_bps / 10_000.0)
    spread_cost = trade_notional_abs * (cfg.half_spread_bps / 10_000.0)
    slippage_cost = trade_notional_abs * (slip_bps / 10_000.0)

    total_cost = fee_cost + spread_cost + slippage_cost

    return {
        "total_cost_$": float(total_cost),
        "fee_cost_$": float(fee_cost),
        "spread_cost_$": float(spread_cost),
        "slippage_cost_$": float(slippage_cost),
        "total_bps": float(total_bps),
        "slip_bps": float(slip_bps),
        "x_participation": float(x),
        "vol_ratio": float(vol_ratio),
    }
