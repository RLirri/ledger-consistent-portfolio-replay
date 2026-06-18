from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, PROCESSED_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.data.panel import build_aligned_panel

UNIVERSE = ["SPY","QQQ","IWM","XLF","XLK","XLE","XLY","XLI","XLV","XLP","XLU","TLT","IEF","GLD"]
INTERVAL = "1d"

def main():
    ensure_dirs()

    feature_files = {
        t: PROCESSED_DIR / f"{t}_{INTERVAL}_features.parquet"
        for t in UNIVERSE
    }

    panel, rep, dropped = build_aligned_panel(
        feature_files=feature_files,
        interval=INTERVAL,
        min_feature_warmup_days=60,
        policy="intersection",
    )

    out = PROCESSED_DIR / f"panel_{INTERVAL}_aligned.parquet"
    panel.to_parquet(out)

    # Save report + dropped dates
    (ARTIFACTS_DIR / "panel_build_report.json").write_text(
        json.dumps(rep.to_dict(), indent=2), encoding="utf-8"
    )
    dropped_out = ARTIFACTS_DIR / "panel_dropped_dates.csv"
    pd.Series(dropped.astype(str), name="dropped_date").to_csv(dropped_out, index=False)

    print(f"Saved aligned panel: {out}")
    print(f"Saved report: {ARTIFACTS_DIR / 'panel_build_report.json'}")
    print(f"Saved dropped dates: {dropped_out}")

if __name__ == "__main__":
    main()
