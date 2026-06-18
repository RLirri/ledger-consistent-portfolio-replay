from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd


@dataclass
class Ledger:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)  # shares (int)
    realized_costs: float = 0.0

    def get_shares(self, ticker: str) -> int:
        return int(self.positions.get(ticker, 0))

    def set_shares(self, ticker: str, shares: int) -> None:
        if shares == 0:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = int(shares)

    def apply_fill(self, ticker: str, shares_delta: int, price: float, cost_dollars: float) -> None:
        """
        Buy: shares_delta > 0 -> cash decreases
        Sell: shares_delta < 0 -> cash increases
        cost_dollars always decreases cash (fees+spread+slippage)
        """
        if shares_delta == 0:
            return
        notional = shares_delta * price
        # Buying reduces cash, selling increases cash (since notional negative for sell)
        self.cash -= notional
        self.cash -= cost_dollars
        self.realized_costs += cost_dollars

        new_shares = self.get_shares(ticker) + int(shares_delta)
        self.set_shares(ticker, new_shares)

    def mark_to_market(self, prices: Dict[str, float]) -> float:
     # Equity at given prices (typically close).
        equity = self.cash
        for t, sh in self.positions.items():
            px = float(prices[t])
            equity += sh * px
        return float(equity)

    def snapshot_positions(self) -> Dict[str, int]:
        return {k: int(v) for k, v in self.positions.items()}
