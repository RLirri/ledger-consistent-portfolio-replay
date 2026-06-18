from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "openbb"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
QC_DIR = ARTIFACTS_DIR / "qc"

def ensure_dirs() -> None:
    for p in [RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, QC_DIR]:
        p.mkdir(parents=True, exist_ok=True)
