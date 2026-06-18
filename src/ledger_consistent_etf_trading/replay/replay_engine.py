from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd

from ledger_consistent_etf_trading.ledger.ledger import Ledger
from ledger_consistent_etf_trading.execution.capacity import (
    CapacityConfig,
    clip_notional,
    notional_to_shares,
)
from ledger_consistent_etf_trading.execution.cost_model import CostConfig, compute_costs


@dataclass
class ReplayConfig:
    initial_cash: float = 1_000_000.0
    price_convention: str = "next_open"   # currently implemented: next_open
    use_close_for_mtm: bool = True

    # realism controls
    strict_cash: bool = True              # no margin: buy only if cash allows
    min_trade_notional: float = 1_000.0   # ignore tiny trades below $ threshold
    carry_tiny_trades: bool = False       # if True, tiny trades become carry for next day


def _panel_slice(panel: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    # panel is indexed by [date, ticker]
    return panel.xs(date, level="date")


def _build_exec_calendar(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index.get_level_values("date").unique().sort_values()


def _shift_weights_to_exec_day(weights: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """
    weights: date,ticker,target_weight defined on signal date.
    execute on next trading day in calendar.
    """
    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"], utc=True)

    next_day = {}
    unique_dates = w["date"].dropna().unique()

    for d in unique_dates:
        pos = calendar.searchsorted(d)
        # if d is exactly a trading day -> execute on the next one
        # else execute on the first trading day after d
        exec_pos = pos + (1 if pos < len(calendar) and calendar[pos] == d else 0)
        if exec_pos >= len(calendar):
            continue
        next_day[d] = calendar[exec_pos]

    w["exec_date"] = w["date"].map(next_day)
    w = (
        w.dropna(subset=["exec_date"])
         .drop(columns=["date"])
         .rename(columns={"exec_date": "date"})
    )
    return w


def run_replay(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    cap_cfg: CapacityConfig,
    cost_cfg: CostConfig,
    replay_cfg: ReplayConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Deterministic replay:
      - weights are formed on day t and executed on next trading day (t+1).
      - execution uses OPEN(t_exec) prices
      - equity is marked to CLOSE(t) prices at end of each day

    Returns:
      fills_df: one row per executed fill
      equity_df: daily equity/cash/costs indexed by date
      positions_df: daily positions snapshot (wide), indexed by date
    """
    calendar = _build_exec_calendar(panel)
    weights_exec = _shift_weights_to_exec_day(weights, calendar)

    # index weights by execution date -> {date: {ticker: weight}}
    weights_exec["date"] = pd.to_datetime(weights_exec["date"], utc=True)
    w_by_date = {
        d: g.set_index("ticker")["target_weight"].to_dict()
        for d, g in weights_exec.groupby("date")
    }

    ledger = Ledger(cash=replay_cfg.initial_cash, positions={})

    # carry desired notional remainder per ticker (for clipped trades)
    carry_notional: Dict[str, float] = {}

    fills = []
    equity_rows = []
    positions_rows = []

    for d in calendar:
        day_df = _panel_slice(panel, d)

        open_px = day_df["open"].to_dict()
        close_px = day_df["close"].to_dict()

        # safer dicts if columns missing
        adv_notional = (
            day_df["adv_notional"].to_dict()
            if "adv_notional" in day_df.columns
            else {t: None for t in day_df.index}
        )
        vol = (
            day_df["vol"].to_dict()
            if "vol" in day_df.columns
            else {t: None for t in day_df.index}
        )

        tickers = list(day_df.index)

        # equity before trades: use close-based mark-to-market for determinism
        equity_pre = ledger.mark_to_market(close_px) if replay_cfg.use_close_for_mtm else ledger.cash

        # target weights for this execution date
        targets = w_by_date.get(d, None)

        # desired notional deltas per ticker
        desired_notional_delta: Dict[str, float] = {}

        if targets is not None:
            for t in tickers:
                w_t = float(targets.get(t, 0.0))
                target_dollar = w_t * equity_pre

                current_sh = ledger.get_shares(t)
                current_dollar = current_sh * float(open_px[t])  # value at execution price

                desired_notional_delta[t] = target_dollar - current_dollar

        # always add carry
        for t, rem in list(carry_notional.items()):
            desired_notional_delta[t] = desired_notional_delta.get(t, 0.0) + float(rem)

        # apply trades
        new_carry: Dict[str, float] = {}

        for t, dn in desired_notional_delta.items():
            if t not in open_px:
                continue
            if abs(dn) < 1e-12:
                continue

            # capacity clipping
            exec_notional, rem_notional = clip_notional(
                desired_notional=float(dn),
                adv_notional=adv_notional.get(t),
                cfg=cap_cfg,
            )

            was_clipped = abs(rem_notional) > 1e-9

            # carry clipped remainder
            if cap_cfg.carry_unfilled and was_clipped:
                new_carry[t] = new_carry.get(t, 0.0) + float(rem_notional)

            # convert executed notional -> shares
            px = float(open_px[t])
            sh_delta = notional_to_shares(exec_notional, px)
            if sh_delta == 0:
                continue

            trade_notional = float(sh_delta * px)
            trade_notional_abs = abs(trade_notional)

            # ignore tiny trades
            if trade_notional_abs < replay_cfg.min_trade_notional:
                if replay_cfg.carry_tiny_trades and cap_cfg.carry_unfilled:
                    # if we skip, we can carry the full executed intent forward
                    new_carry[t] = new_carry.get(t, 0.0) + float(exec_notional)
                continue

            # compute costs on executed trade
            cost_breakdown = compute_costs(
                trade_notional_abs=trade_notional_abs,
                adv_notional=adv_notional.get(t),
                vol=vol.get(t),
                cfg=cost_cfg,
            )

            cash_scaled = False

            # strict cash: scale buys down if cash insufficient
            if replay_cfg.strict_cash and sh_delta > 0:
                required = trade_notional_abs + cost_breakdown["total_cost_$"]
                if ledger.cash < required:
                    affordable = max(ledger.cash - cost_breakdown["total_cost_$"], 0.0)
                    sh_aff = int(affordable // px)
                    if sh_aff <= 0:
                        continue

                    cash_scaled = True
                    sh_delta = sh_aff

                    trade_notional = float(sh_delta * px)
                    trade_notional_abs = abs(trade_notional)

                    # recompute costs on resized trade
                    cost_breakdown = compute_costs(
                        trade_notional_abs=trade_notional_abs,
                        adv_notional=adv_notional.get(t),
                        vol=vol.get(t),
                        cfg=cost_cfg,
                    )

            # apply fill to ledger
            ledger.apply_fill(
                ticker=t,
                shares_delta=sh_delta,
                price=px,
                cost_dollars=cost_breakdown["total_cost_$"],
            )

            fills.append({
                "date": d,
                "ticker": t,

                # diagnostics
                "desired_notional_$": float(dn),
                "executed_notional_$": float(exec_notional),
                "remaining_notional_$": float(rem_notional),
                "was_clipped": bool(was_clipped),
                "cash_scaled": bool(cash_scaled),

                # execution
                "shares_delta": int(sh_delta),
                "price": px,
                "notional_$": float(trade_notional),

                # costs + model diagnostics
                **cost_breakdown,
            })

        carry_notional = new_carry

        # end-of-day MTM at close
        equity = ledger.mark_to_market(close_px)
        equity_rows.append({
            "date": d,
            "equity": float(equity),
            "cash": float(ledger.cash),
            "realized_costs": float(ledger.realized_costs),
        })

        pos = ledger.snapshot_positions()
        positions_rows.append({"date": d, **pos})

    fills_df = pd.DataFrame(fills)
    equity_df = pd.DataFrame(equity_rows).set_index("date")
    positions_df = pd.DataFrame(positions_rows).set_index("date").fillna(0).astype(int)

    return fills_df, equity_df, positions_df
