"""Market loaders addressable by name from the batch runner's child process."""
import os

from .data import load_snapshot

UNIVERSE = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLY", "XLI", "XLV", "XLP", "XLU", "TLT", "IEF", "GLD"]


def snapshot():
    """Pinned historical snapshot named by the PV2_SNAPSHOT environment variable. Historical data: use only under authorization."""
    path = os.environ.get("PV2_SNAPSHOT")
    if not path:
        raise RuntimeError("PV2_SNAPSHOT is not set")
    return load_snapshot(path, UNIVERSE)
