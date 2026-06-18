from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import pandas as pd

from ledger_consistent_etf_trading.config import (
    UNIVERSE,
    START_DATE,
    END_DATE,
    INTERVAL,
)
from ledger_consistent_etf_trading.data.clean_qc import standardize_ohlcv
from ledger_consistent_etf_trading.data.openbb_fetch import fetch_ohlcv_openbb
from ledger_consistent_etf_trading.utils.paths import ensure_dirs, RAW_DIR, ARTIFACTS_DIR
from ledger_consistent_etf_trading.utils.time import to_utc_date_index


REQUIRED_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Validate downloaded OHLCV data before saving.

    For publication results, the pipeline should fail loudly instead of
    silently saving empty, incomplete, or corrupted market data.
    """
    if df is None:
        raise ValueError(f"{ticker}: downloaded dataframe is None")

    if df.empty:
        raise ValueError(f"{ticker}: downloaded dataframe is empty")

    missing_cols = REQUIRED_OHLCV_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"{ticker}: missing required columns {sorted(missing_cols)}")

    missing_close_ratio = float(df["close"].isna().mean())
    missing_volume_ratio = float(df["volume"].isna().mean())

    if missing_close_ratio > 0.01:
        raise ValueError(
            f"{ticker}: too many missing close values "
            f"({missing_close_ratio:.2%})"
        )

    if missing_volume_ratio > 0.01:
        raise ValueError(
            f"{ticker}: too many missing volume values "
            f"({missing_volume_ratio:.2%})"
        )

    if (df["close"] <= 0).any():
        raise ValueError(f"{ticker}: non-positive close prices detected")

    if (df["volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume values detected")

    return df


def fetch_with_retry(
    ticker: str,
    retries: int = 3,
    wait_seconds: int = 5,
) -> pd.DataFrame:
    """
    Fetch OHLCV data with retry logic and validation.

    A temporary provider timeout may be retried. However, after all retries,
    the script stops instead of saving invalid data.
    """
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Fetching {ticker} ({attempt}/{retries})...")

            df = fetch_ohlcv_openbb(
                ticker,
                start=START_DATE,
                end=END_DATE,
            )

            df = standardize_ohlcv(df)
            df = validate_ohlcv(df, ticker)

            return df

        except Exception as e:
            last_error = e
            print(f"{ticker}: attempt {attempt}/{retries} failed: {e}")

            if attempt < retries:
                time.sleep(wait_seconds)

    raise RuntimeError(f"{ticker}: failed after {retries} attempts") from last_error


def main() -> None:
    ensure_dirs()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "openbb",
        "start": START_DATE,
        "end": END_DATE,
        "interval": INTERVAL,
        "universe": UNIVERSE,
        "python": platform.python_version(),
    }

    try:
        import openbb

        manifest["openbb_version"] = getattr(openbb, "__version__", "unknown")
    except Exception:
        manifest["openbb_version"] = "unknown"

    for ticker in UNIVERSE:
        df = fetch_with_retry(ticker)
        df = to_utc_date_index(df, col="date")

        out = RAW_DIR / f"{ticker}_{INTERVAL}.parquet"
        df.to_parquet(out)

        print(f"Saved raw: {out}")

    manifest_path = ARTIFACTS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

