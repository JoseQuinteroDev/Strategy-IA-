from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True, slots=True)
class DeterministicFeatureConfig:
    atr_window: int = 14
    adx_window: int = 14
    ema_1h_span: int = 200
    intraday_ema_span: int = 50
    realized_vol_window: int = 20
    zscore_window: int = 50
    breakout_window: int = 20
    momentum_window: int = 20


def build_features(
    df: pd.DataFrame,
    *,
    config: DeterministicFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build deterministic multi-timeframe features aligned to the intraday index."""

    cfg = config or DeterministicFeatureConfig()
    frame = _prepare_ohlcv_frame(df)

    features = pd.DataFrame(index=frame.index)
    log_return = np.log(frame["close"]).diff()
    features["log_return"] = log_return

    atr = _atr(frame, window=cfg.atr_window)
    features[f"atr_{cfg.atr_window}"] = atr

    hourly = _resample_to_1h(frame)
    hourly["ema_200_1h"] = hourly["close"].ewm(
        span=cfg.ema_1h_span,
        adjust=False,
        min_periods=1,
    ).mean()
    hourly["adx_1h"] = _adx(hourly, window=cfg.adx_window)
    features["ema_200_1h"] = hourly["ema_200_1h"].reindex(features.index, method="ffill")
    features["adx_1h"] = hourly["adx_1h"].reindex(features.index, method="ffill")

    features["intraday_vwap"] = _intraday_vwap(frame)
    features["ema_50"] = frame["close"].ewm(
        span=cfg.intraday_ema_span,
        adjust=False,
        min_periods=1,
    ).mean()

    realized_vol = log_return.rolling(
        cfg.realized_vol_window,
        min_periods=max(5, cfg.realized_vol_window // 2),
    ).std(ddof=0)
    features[f"realized_volatility_{cfg.realized_vol_window}"] = realized_vol * np.sqrt(
        cfg.realized_vol_window
    )

    features["candle_range"] = frame["high"] - frame["low"]
    features["candle_range_pct"] = features["candle_range"] / frame["close"].replace(0.0, np.nan)
    features["candle_range_atr"] = features["candle_range"] / atr.replace(0.0, np.nan)

    breakout_high = frame["high"].rolling(cfg.breakout_window, min_periods=cfg.breakout_window).max().shift(1)
    breakout_low = frame["low"].rolling(cfg.breakout_window, min_periods=cfg.breakout_window).min().shift(1)
    breakout_range_width = breakout_high - breakout_low
    features[f"breakout_high_{cfg.breakout_window}"] = breakout_high
    features[f"breakout_low_{cfg.breakout_window}"] = breakout_low
    features[f"breakout_range_width_{cfg.breakout_window}"] = breakout_range_width
    features[f"breakout_range_width_pct_{cfg.breakout_window}"] = (
        breakout_range_width / frame["close"].replace(0.0, np.nan)
    )
    features[f"breakout_range_width_atr_{cfg.breakout_window}"] = (
        breakout_range_width / atr.replace(0.0, np.nan)
    )
    features[f"momentum_{cfg.momentum_window}"] = frame["close"].pct_change(
        cfg.momentum_window,
        fill_method=None,
    )
    features["price_vs_ema_200_1h_pct"] = (
        (frame["close"] - features["ema_200_1h"]) / features["ema_200_1h"].replace(0.0, np.nan)
    )

    mean_anchor = features["intraday_vwap"].combine_first(features["ema_50"])
    distance_to_mean = frame["close"] - mean_anchor
    features["distance_to_mean"] = distance_to_mean
    rolling_distance_std = distance_to_mean.rolling(
        cfg.zscore_window,
        min_periods=max(10, cfg.zscore_window // 2),
    ).std(ddof=0)
    features["zscore_distance_to_mean"] = distance_to_mean / rolling_distance_std.replace(0.0, np.nan)

    features = pd.concat([features, _temporal_features(frame.index)], axis=1)
    return features


def _prepare_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(column).lower() for column in frame.columns]

    if "open_time" in frame.columns:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True, errors="coerce")
        frame = frame.set_index("open_time")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("build_features requires a DatetimeIndex or an 'open_time' column.")

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if index.isna().any():
        raise ValueError("Input DataFrame contains invalid timestamps.")

    frame.index = index
    frame.index.name = "open_time"
    frame = frame.sort_index(kind="mergesort")
    if not frame.index.is_unique:
        raise ValueError("Input DataFrame index must be unique before feature construction.")

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _resample_to_1h(frame: pd.DataFrame) -> pd.DataFrame:
    hourly = frame.resample("1h", label="right", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return hourly.dropna(subset=["open", "high", "low", "close"])


def _atr(frame: pd.DataFrame, *, window: int) -> pd.Series:
    true_range = _true_range(frame)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _adx(frame: pd.DataFrame, *, window: int) -> pd.Series:
    high_diff = frame["high"].diff()
    low_diff = -frame["low"].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0.0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0.0), 0.0)

    atr = _atr(frame, window=window)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) * 100.0
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _intraday_vwap(frame: pd.DataFrame) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    trade_date = frame.index.tz_convert("UTC").normalize()
    cumulative_notional = (typical_price * frame["volume"]).groupby(trade_date).cumsum()
    cumulative_volume = frame["volume"].groupby(trade_date).cumsum()
    return cumulative_notional / cumulative_volume.replace(0.0, np.nan)


def _temporal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    utc_index = index.tz_convert("UTC")
    hour = utc_index.hour
    features = pd.DataFrame(index=index)
    features["hour_utc"] = hour.astype(float)
    features["day_of_week"] = utc_index.dayofweek.astype(float)
    features["session_asia"] = ((hour >= 0) & (hour < 8)).astype(float)
    features["session_europe"] = ((hour >= 8) & (hour < 16)).astype(float)
    features["session_us"] = ((hour >= 16) & (hour < 24)).astype(float)
    return features
