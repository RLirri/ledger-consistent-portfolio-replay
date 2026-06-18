from __future__ import annotations
import pandas as pd

from ledger_consistent_etf_trading.utils.paths import ensure_dirs, RAW_DIR, PROCESSED_DIR, QC_DIR
from ledger_consistent_etf_trading.data.clean_qc import qc_report, clean_ohlcv, save_qc
from ledger_consistent_etf_trading.data.features import add_features

UNIVERSE = ["SPY","QQQ","IWM","XLF","XLK","XLE","XLY","XLI","XLV","XLP","XLU","TLT","IEF","GLD"]
INTERVAL = "1d"

def main():
    ensure_dirs()

    frames = []
    for t in UNIVERSE:
        raw_path = RAW_DIR / f"{t}_{INTERVAL}.parquet"
        df = pd.read_parquet(raw_path)

        rep_before = qc_report(df)
        df_clean = clean_ohlcv(df)
        rep_after = qc_report(df_clean)

        save_qc(QC_DIR / f"{t}_qc.json", {"before": rep_before, "after": rep_after})

        out_clean = PROCESSED_DIR / f"{t}_{INTERVAL}_clean.parquet"
        df_clean.to_parquet(out_clean)

        df_feat = add_features(df_clean, vol_window=20, adv_window=20)
        out_feat = PROCESSED_DIR / f"{t}_{INTERVAL}_features.parquet"
        df_feat.to_parquet(out_feat)

        # Build panel
        tmp = df_feat.copy()
        tmp["ticker"] = t
        frames.append(tmp.reset_index().rename(columns={"index": "date"}))

        print(f"Processed: {t}")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()

    panel_out = PROCESSED_DIR / f"panel_{INTERVAL}.parquet"
    panel.to_parquet(panel_out)
    print(f"Saved panel: {panel_out}")

if __name__ == "__main__":
    main()
