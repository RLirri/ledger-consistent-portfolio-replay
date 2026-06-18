from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ARTIFACTS_DIR

def main():
    fills = pd.read_csv(ARTIFACTS_DIR / "fills_equal_weight.csv")
    fills["date"] = pd.to_datetime(fills["date"], utc=True)

    # Capacity utilization
    if "x_participation" in fills.columns:
        cap = {
            "fills": len(fills),
            "pct_clipped": float((fills.get("was_clipped", False) == True).mean()) if "was_clipped" in fills.columns else None,
            "x_mean": float(fills["x_participation"].mean()),
            "x_p95": float(fills["x_participation"].quantile(0.95)),
            "x_max": float(fills["x_participation"].max()),
        }
    else:
        cap = {}

    # Costs
    fills["abs_notional"] = fills["notional_$"].abs()
    total_traded = float(fills["abs_notional"].sum())
    total_cost = float(fills["total_cost_$"].sum())
    cost_bps_on_traded = (total_cost / total_traded) * 10_000 if total_traded > 0 else 0.0

    costs = {
        "total_traded_$": total_traded,
        "total_cost_$": total_cost,
        "cost_bps_on_traded": float(cost_bps_on_traded),
        "spread_cost_$": float(fills.get("spread_cost_$", pd.Series(dtype=float)).sum()),
        "slippage_cost_$": float(fills.get("slippage_cost_$", pd.Series(dtype=float)).sum()), # high-level volatility
        # low level
        "fee_cost_$": float(fills.get("fee_cost_$", pd.Series(dtype=float)).sum()),
    }

    # Cash scaling
    cash_scaled_pct = None
    if "cash_scaled" in fills.columns:
        cash_scaled_pct = float((fills["cash_scaled"] == True).mean())

    print("Capacity")
    print(cap)
    print("\nCosts")
    print(costs)
    print("\nCash scaled")
    print({"pct_cash_scaled_fills": cash_scaled_pct})

if __name__ == "__main__":
    main()
