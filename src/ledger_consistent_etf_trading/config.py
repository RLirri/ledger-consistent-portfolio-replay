from __future__ import annotations

UNIVERSE = [
    "SPY", "QQQ", "IWM",
    "XLF", "XLK", "XLE", "XLY", "XLI", "XLV", "XLP", "XLU",
    "TLT", "IEF", "GLD",
]

START_DATE = "2010-01-01"
END_DATE = "2026-06-15"
INTERVAL = "1d"

INITIAL_CASH = 1_000_000.0
BENCHMARK = "SPY"
REBALANCE_FREQ_WEEKLY = "W"
REBALANCE_FREQ_MONTHLY = "M"