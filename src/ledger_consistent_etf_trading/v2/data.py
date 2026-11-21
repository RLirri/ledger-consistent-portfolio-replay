from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ScenarioConfig
from .errors import ScenarioInputError


@dataclass
class Market:
    dates: pd.DatetimeIndex          # sessions, ascending
    tickers: list                    # sorted, fixed order
    open: np.ndarray                 # [T, N]
    close: np.ndarray
    volume: np.ndarray
    input_hashes: dict

    def identity(self) -> str:
        """Content hash of the calendar, universe and price/volume arrays (NaN-stable)."""
        h = hashlib.sha256()
        h.update(np.asarray(self.dates.asi8, dtype="int64").tobytes())
        h.update("|".join(self.tickers).encode())
        for a in (self.open, self.close, self.volume):
            h.update(np.ascontiguousarray(a, dtype="float64").tobytes())
        return h.hexdigest()

    def validate(self) -> None:
        d = self.dates
        if not isinstance(d, pd.DatetimeIndex) or d.tz is None or not (d.is_monotonic_increasing and d.is_unique):
            raise ScenarioInputError("sessions must be a unique, ascending, timezone-aware DatetimeIndex")
        if len(set(self.tickers)) != len(self.tickers) or list(self.tickers) != sorted(self.tickers):
            raise ScenarioInputError("tickers must be unique and sorted")
        shape = (len(d), len(self.tickers))
        for name in ("open", "close", "volume"):
            if getattr(self, name).shape != shape:
                raise ScenarioInputError(f"market array {name} has shape {getattr(self, name).shape}, expected {shape}")

    def validate_domain(self, end: int, strict: bool = False) -> None:
        """Raw-observation domain over every session the run can read, warm-up included (rows 0 .. end-1).

        Policy. Impossible values are rejected always: a price that is zero, negative or infinite, a volume
        that is negative or infinite. A missing value (NaN) is tolerated outside production because a
        synthetic fixture may leave an unused asset blank; it still fails closed wherever the engine needs
        it. In production (`strict`) no value may be missing: every open and close must be finite and
        positive and every volume finite and non-negative. Zero volume is a valid observation.
        """
        for name in ("open", "close"):
            a = getattr(self, name)[:end]
            bad = (~np.isnan(a)) & ((a <= 0) | np.isinf(a))
            if bad.any() or (strict and np.isnan(a).any()):
                raise ScenarioInputError(f"raw {name} prices contain {'missing or ' if strict else ''}impossible values in the sessions this run reads")
        v = self.volume[:end]
        bad = (~np.isnan(v)) & ((v < 0) | np.isinf(v))
        if bad.any() or (strict and np.isnan(v).any()):
            raise ScenarioInputError(f"raw volume contains {'missing or ' if strict else ''}impossible values in the sessions this run reads")

    def copy(self) -> "Market":
        return Market(self.dates, list(self.tickers), self.open.copy(), self.close.copy(), self.volume.copy(), dict(self.input_hashes))


def market_from_frames(open_: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, hashes=None) -> Market:
    tick = sorted(close.columns)
    idx = close.index
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ScenarioInputError("session index must be unique and ascending")
    for f in (open_, volume):
        if not f.index.equals(idx) or sorted(f.columns) != tick:
            raise ScenarioInputError("misaligned market frames")
    return Market(pd.DatetimeIndex(idx), tick, open_[tick].to_numpy(float), close[tick].to_numpy(float), volume[tick].to_numpy(float), hashes or {})


def load_snapshot(snapshot_dir, tickers) -> Market:
    """Read per-ticker clean OHLCV Parquets (full history, before any warm-up drop) on the common calendar."""
    root = Path(snapshot_dir) / "data" / "processed"
    frames, hashes = {}, {}
    for t in sorted(tickers):
        p = root / f"{t}_1d_clean.parquet"
        hashes[str(p.relative_to(snapshot_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        frames[t] = df.sort_index()
    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    idx = pd.DatetimeIndex(common)
    mk = lambda col: pd.DataFrame({t: frames[t].loc[idx, col] for t in frames}, index=idx)
    return market_from_frames(mk("open"), mk("close"), mk("volume"), hashes)


@dataclass
class Features:
    vol: np.ndarray       # [T, N] daily return volatility consumed at session d
    adv: np.ndarray       # [T, N] dollar ADV consumed at session d
    mode: str
    estimator: str
    window: int = 20
    market_id: Optional[str] = None      # identity of the Market these were computed from; None = unbound (externally supplied)


def build_features(mkt: Market, mode: str = "lagged", estimator: str = "mean_dollar", window: int = 20) -> Features:
    """Rolling liquidity features.

    lagged:   value consumed at d is computed from sessions d-window .. d-1 only.
    same_day: window d-window+1 .. d (legacy behaviour; violates the pre-open contract; negative control).
    mean_dollar:     mean(close_k * volume_k).
    shares_x_close:  mean(volume_k) * close at the last session of the window (legacy estimator).
    Volatility is the sample standard deviation (ddof=1) of `window` close-to-close returns.
    """
    close = pd.DataFrame(mkt.close)
    volume = pd.DataFrame(mkt.volume)
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(window, min_periods=window).std(ddof=1)
    if estimator == "mean_dollar":
        adv = (close * volume).rolling(window, min_periods=window).mean()
    elif estimator == "shares_x_close":
        adv = volume.rolling(window, min_periods=window).mean() * close
    else:
        raise ScenarioInputError(estimator)
    if mode == "lagged":
        vol, adv = vol.shift(1), adv.shift(1)
    elif mode != "same_day":
        raise ScenarioInputError(mode)
    return Features(vol.to_numpy(float), adv.to_numpy(float), mode, estimator, window, mkt.identity())


FEATURE_WINDOW = 20      # fixed by the protocol: 20 returns (21 closes) and 20 dollar-volume observations


def check_features(mkt: Market, cfg: ScenarioConfig, feats: Features, production: bool = False) -> str:
    """Pre-replay feature contract.

    Labels alone prove nothing, so features that claim to come from this market are recomputed with the
    protocol window and must be equal element by element. Features with no market identity ('unbound')
    are a synthetic-fixture route: allowed only outside production and reported as such.
    Returns 'bound' (verified by recomputation) or 'unbound'.
    """
    if feats.mode != cfg.features or feats.estimator != cfg.adv_estimator:
        raise ScenarioInputError(f"supplied features are {feats.mode}/{feats.estimator} but the scenario declares {cfg.features}/{cfg.adv_estimator}")
    if feats.window != FEATURE_WINDOW:
        raise ScenarioInputError(f"feature window {feats.window!r} is not the protocol window {FEATURE_WINDOW}")
    if feats.vol.shape != mkt.close.shape or feats.adv.shape != mkt.close.shape:
        raise ScenarioInputError("supplied features do not match the market's sessions and universe")
    if feats.market_id is None:
        if production:
            raise ScenarioInputError("production replay requires features bound to and recomputable from the run's market data")
        return "unbound"
    if feats.market_id != mkt.identity():
        raise ScenarioInputError("supplied features were computed from different market data")
    ref = build_features(mkt, cfg.features, cfg.adv_estimator, FEATURE_WINDOW)
    if not (np.array_equal(ref.vol, feats.vol, equal_nan=True) and np.array_equal(ref.adv, feats.adv, equal_nan=True)):
        raise ScenarioInputError("supplied features do not equal the protocol computation for their declared mode, estimator and window")
    return "bound"


def session_range(mkt: Market, cfg: ScenarioConfig) -> tuple:
    start = int(mkt.dates.searchsorted(pd.Timestamp(cfg.eval_start, tz="UTC")))
    end = len(mkt.dates) if cfg.eval_end is None else int(mkt.dates.searchsorted(pd.Timestamp(cfg.eval_end, tz="UTC"), side="right"))
    if cfg.max_sessions is not None:
        end = min(end, start + int(cfg.max_sessions))
    if start < 1 or end <= start:
        raise ScenarioInputError("evaluation range needs at least one prior session and one evaluation session")
    return start, end
