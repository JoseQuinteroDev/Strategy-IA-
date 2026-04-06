from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True, slots=True)
class DeterministicFeatureConfig:
    atr_window: int = 14
    adx_window: int = 14
    ema_1h_span: int = 200
    intraday_ema_span: int = 50
    intraday_fast_ema_span: int = 20
    realized_vol_window: int = 20
    zscore_window: int = 50
    breakout_window: int = 20
    momentum_window: int = 20
    ema_slope_lookback_hours: int = 3
    relative_volume_lookback_sessions: int = 20
    opening_range_minutes: int = 30
    session_start_hour_utc: int = 13
    session_start_minute_utc: int = 30
    session_end_hour_utc: int = 20
    session_end_minute_utc: int = 55
    retest_max_bars: int = 3
    opening_range_breakout_buffer_atr: float = 0.0
    use_intraday_vwap_filter: bool = True
    use_intraday_ema50_alignment: bool = True
    use_opening_range_mid_filter: bool = True
    session_trend_structure_lookback_bars: int = 3
    maximum_context_compression_width_atr: float = 1.5
    require_context_vwap_structure: bool = True
    require_context_or_mid_structure: bool = True


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
    hourly["ema_200_1h_slope"] = hourly["ema_200_1h"].diff(cfg.ema_slope_lookback_hours)
    features["ema_200_1h"] = hourly["ema_200_1h"].reindex(features.index, method="ffill")
    features["adx_1h"] = hourly["adx_1h"].reindex(features.index, method="ffill")
    features["ema_200_1h_slope"] = hourly["ema_200_1h_slope"].reindex(features.index, method="ffill")

    features["intraday_vwap"] = _intraday_vwap(frame)
    features["ema_20"] = frame["close"].ewm(
        span=cfg.intraday_fast_ema_span,
        adjust=False,
        min_periods=1,
    ).mean()
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
    features["relative_volume"] = _relative_volume(frame, cfg=cfg)
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
    features = pd.concat([features, _opening_range_features(frame, atr, cfg=cfg)], axis=1)
    features = pd.concat([features, _intraday_contextual_features(frame, features, atr, cfg=cfg)], axis=1)

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


def _session_components(
    index: pd.DatetimeIndex,
    *,
    session_start_hour_utc: int,
    session_start_minute_utc: int,
    session_end_hour_utc: int,
    session_end_minute_utc: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, pd.Series]:
    utc_index = index.tz_convert("UTC")
    minutes = (utc_index.hour * 60) + utc_index.minute
    session_start_minutes = (session_start_hour_utc * 60) + session_start_minute_utc
    session_end_minutes = (session_end_hour_utc * 60) + session_end_minute_utc
    in_session = (minutes >= session_start_minutes) & (minutes <= session_end_minutes)
    session_day = utc_index.normalize().to_numpy()

    session_bar_index = pd.Series(np.nan, index=index, dtype=float)
    session_series = pd.Series(in_session, index=index)
    for _, session_index in session_series[session_series].groupby(utc_index.normalize()[in_session]):
        session_bar_index.loc[session_index.index] = np.arange(len(session_index), dtype=float)

    return utc_index, in_session, session_day, session_bar_index


def _relative_volume(frame: pd.DataFrame, *, cfg: DeterministicFeatureConfig) -> pd.Series:
    _, _, _, session_bar_index = _session_components(
        frame.index,
        session_start_hour_utc=cfg.session_start_hour_utc,
        session_start_minute_utc=cfg.session_start_minute_utc,
        session_end_hour_utc=cfg.session_end_hour_utc,
        session_end_minute_utc=cfg.session_end_minute_utc,
    )
    relative_volume = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = session_bar_index.notna()
    if not valid.any():
        return relative_volume

    session_volume = frame.loc[valid, "volume"]
    slot_index = session_bar_index.loc[valid].astype(int)
    for _, positions in slot_index.groupby(slot_index):
        slot_volume = session_volume.loc[positions.index]
        rolling_reference = slot_volume.shift(1).rolling(
            cfg.relative_volume_lookback_sessions,
            min_periods=max(3, cfg.relative_volume_lookback_sessions // 4),
        ).mean()
        relative_volume.loc[positions.index] = slot_volume / rolling_reference.replace(0.0, np.nan)
    return relative_volume


def _opening_range_features(
    frame: pd.DataFrame,
    atr: pd.Series,
    *,
    cfg: DeterministicFeatureConfig,
) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    defaults = {
        "opening_range_high": np.nan,
        "opening_range_low": np.nan,
        "opening_range_mid": np.nan,
        "opening_range_width": np.nan,
        "opening_range_width_atr": np.nan,
        "opening_range_ready": 0.0,
        "opening_range_long_breakout_entry": 0.0,
        "opening_range_short_breakout_entry": 0.0,
        "opening_range_long_retest_entry": 0.0,
        "opening_range_short_retest_entry": 0.0,
        "opening_range_breakout_count_today": np.nan,
        "opening_range_first_breakout_of_day": np.nan,
        "opening_range_long_continuation_entry": 0.0,
        "opening_range_short_continuation_entry": 0.0,
        "opening_range_long_pullback_entry": 0.0,
        "opening_range_short_pullback_entry": 0.0,
        "opening_range_long_reclaim_entry": 0.0,
        "opening_range_short_reclaim_entry": 0.0,
        "opening_range_bias_side": 0.0,
        "opening_range_breakout_age_bars": np.nan,
        "opening_range_acceptance_bars": np.nan,
        "opening_range_pullback_depth_atr": np.nan,
        "opening_range_reclaim_distance_atr": np.nan,
        "opening_range_intraday_setup_count_today": np.nan,
        "opening_range_intraday_first_setup_of_day": np.nan,
    }
    for column, default in defaults.items():
        features[column] = default

    if frame.empty:
        return features

    _, in_session, session_day, session_bar_index = _session_components(
        frame.index,
        session_start_hour_utc=cfg.session_start_hour_utc,
        session_start_minute_utc=cfg.session_start_minute_utc,
        session_end_hour_utc=cfg.session_end_hour_utc,
        session_end_minute_utc=cfg.session_end_minute_utc,
    )
    step = frame.index.to_series().diff().dropna().median()
    bar_minutes = max(1, int(round(step.total_seconds() / 60.0))) if pd.notna(step) else 5
    opening_range_bars = max(1, int(math.ceil(cfg.opening_range_minutes / bar_minutes)))

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    closes = frame["close"].to_numpy()

    unique_session_days = pd.unique(session_day[in_session])
    for current_day in unique_session_days:
        session_positions = np.flatnonzero(in_session & (session_day == current_day))
        if len(session_positions) <= opening_range_bars:
            continue

        opening_positions = session_positions[:opening_range_bars]
        tradable_positions = session_positions[opening_range_bars:]
        opening_range_high = float(np.max(highs[opening_positions]))
        opening_range_low = float(np.min(lows[opening_positions]))
        opening_range_mid = (opening_range_high + opening_range_low) / 2.0
        opening_range_width = opening_range_high - opening_range_low

        features.iloc[tradable_positions, features.columns.get_loc("opening_range_high")] = opening_range_high
        features.iloc[tradable_positions, features.columns.get_loc("opening_range_low")] = opening_range_low
        features.iloc[tradable_positions, features.columns.get_loc("opening_range_mid")] = opening_range_mid
        features.iloc[tradable_positions, features.columns.get_loc("opening_range_width")] = opening_range_width
        features.iloc[tradable_positions, features.columns.get_loc("opening_range_ready")] = 1.0

        width_atr = opening_range_width / atr.iloc[tradable_positions].replace(0.0, np.nan)
        features.iloc[tradable_positions, features.columns.get_loc("opening_range_width_atr")] = width_atr.to_numpy()

        breakout_count = 0
        setup_count = 0
        active_long_deadline: int | None = None
        active_short_deadline: int | None = None
        active_long_breakout_count: int | None = None
        active_short_breakout_count: int | None = None
        active_bias_side = 0.0
        active_breakout_relative_index: int | None = None
        long_acceptance_bars = 0
        short_acceptance_bars = 0
        long_level_lost = False
        short_level_lost = False
        long_pullback_used = False
        short_pullback_used = False

        for relative_index, position in enumerate(tradable_positions):
            previous_close = closes[position - 1] if position > 0 else np.nan
            current_atr = float(atr.iloc[position]) if pd.notna(atr.iloc[position]) else float("nan")
            breakout_buffer = (
                current_atr * cfg.opening_range_breakout_buffer_atr
                if math.isfinite(current_atr) and current_atr > 0.0
                else 0.0
            )
            long_breakout_level = opening_range_high + breakout_buffer
            short_breakout_level = opening_range_low - breakout_buffer
            long_breakout = bool(
                closes[position] > long_breakout_level
                and previous_close <= long_breakout_level
                and not (active_bias_side == 1.0 and long_level_lost)
            )
            short_breakout = bool(
                closes[position] < short_breakout_level
                and previous_close >= short_breakout_level
                and not (active_bias_side == -1.0 and short_level_lost)
            )

            if active_long_deadline is not None and relative_index > active_long_deadline:
                active_long_deadline = None
                active_long_breakout_count = None
            if active_short_deadline is not None and relative_index > active_short_deadline:
                active_short_deadline = None
                active_short_breakout_count = None

            if long_breakout:
                breakout_count += 1
                features.iat[position, features.columns.get_loc("opening_range_long_breakout_entry")] = 1.0
                features.iat[position, features.columns.get_loc("opening_range_breakout_count_today")] = breakout_count
                features.iat[position, features.columns.get_loc("opening_range_first_breakout_of_day")] = (
                    1.0 if breakout_count == 1 else 0.0
                )
                if cfg.retest_max_bars > 0:
                    active_long_deadline = relative_index + cfg.retest_max_bars
                    active_long_breakout_count = breakout_count
                else:
                    active_long_deadline = None
                    active_long_breakout_count = None
                active_short_deadline = None
                active_short_breakout_count = None
                active_bias_side = 1.0
                active_breakout_relative_index = relative_index
                long_acceptance_bars = 1
                short_acceptance_bars = 0
                long_level_lost = False
                short_level_lost = False
                long_pullback_used = False
                short_pullback_used = False
                continue

            if short_breakout:
                breakout_count += 1
                features.iat[position, features.columns.get_loc("opening_range_short_breakout_entry")] = 1.0
                features.iat[position, features.columns.get_loc("opening_range_breakout_count_today")] = breakout_count
                features.iat[position, features.columns.get_loc("opening_range_first_breakout_of_day")] = (
                    1.0 if breakout_count == 1 else 0.0
                )
                if cfg.retest_max_bars > 0:
                    active_short_deadline = relative_index + cfg.retest_max_bars
                    active_short_breakout_count = breakout_count
                else:
                    active_short_deadline = None
                    active_short_breakout_count = None
                active_long_deadline = None
                active_long_breakout_count = None
                active_bias_side = -1.0
                active_breakout_relative_index = relative_index
                long_acceptance_bars = 0
                short_acceptance_bars = 1
                long_level_lost = False
                short_level_lost = False
                long_pullback_used = False
                short_pullback_used = False
                continue

            if active_bias_side == 1.0 and active_breakout_relative_index is not None:
                bars_since_breakout = relative_index - active_breakout_relative_index
                features.iat[position, features.columns.get_loc("opening_range_bias_side")] = 1.0
                features.iat[position, features.columns.get_loc("opening_range_breakout_age_bars")] = float(
                    bars_since_breakout
                )

                if closes[position] >= opening_range_high:
                    long_acceptance_bars += 1
                else:
                    long_acceptance_bars = 0
                    long_level_lost = True
                features.iat[position, features.columns.get_loc("opening_range_acceptance_bars")] = float(
                    long_acceptance_bars
                )

                if math.isfinite(current_atr) and current_atr > 0.0:
                    pullback_depth_atr = max(0.0, (opening_range_high - lows[position]) / current_atr)
                    reclaim_distance_atr = (
                        max(0.0, (closes[position] - opening_range_high) / current_atr)
                        if closes[position] >= opening_range_high
                        else 0.0
                    )
                    features.iat[position, features.columns.get_loc("opening_range_pullback_depth_atr")] = (
                        pullback_depth_atr
                    )
                    features.iat[position, features.columns.get_loc("opening_range_reclaim_distance_atr")] = (
                        reclaim_distance_atr
                    )
                else:
                    pullback_depth_atr = float("nan")

                continuation_entry = (
                    bars_since_breakout >= 1
                    and previous_close >= opening_range_high
                    and closes[position] > highs[position - 1]
                    and lows[position] >= opening_range_high - breakout_buffer
                )
                if continuation_entry:
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_long_continuation_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

                pullback_entry = (
                    not long_pullback_used
                    and bars_since_breakout >= 1
                    and (cfg.retest_max_bars <= 0 or bars_since_breakout <= cfg.retest_max_bars)
                    and lows[position] <= opening_range_high + breakout_buffer
                    and closes[position] >= opening_range_high
                    and closes[position] >= previous_close
                )
                if pullback_entry:
                    long_pullback_used = True
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_long_pullback_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

                reclaim_entry = (
                    long_level_lost
                    and previous_close <= opening_range_high
                    and closes[position] > long_breakout_level
                )
                if reclaim_entry:
                    long_level_lost = False
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_long_reclaim_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

            if active_bias_side == -1.0 and active_breakout_relative_index is not None:
                bars_since_breakout = relative_index - active_breakout_relative_index
                features.iat[position, features.columns.get_loc("opening_range_bias_side")] = -1.0
                features.iat[position, features.columns.get_loc("opening_range_breakout_age_bars")] = float(
                    bars_since_breakout
                )

                if closes[position] <= opening_range_low:
                    short_acceptance_bars += 1
                else:
                    short_acceptance_bars = 0
                    short_level_lost = True
                features.iat[position, features.columns.get_loc("opening_range_acceptance_bars")] = float(
                    short_acceptance_bars
                )

                if math.isfinite(current_atr) and current_atr > 0.0:
                    pullback_depth_atr = max(0.0, (highs[position] - opening_range_low) / current_atr)
                    reclaim_distance_atr = (
                        max(0.0, (opening_range_low - closes[position]) / current_atr)
                        if closes[position] <= opening_range_low
                        else 0.0
                    )
                    features.iat[position, features.columns.get_loc("opening_range_pullback_depth_atr")] = (
                        pullback_depth_atr
                    )
                    features.iat[position, features.columns.get_loc("opening_range_reclaim_distance_atr")] = (
                        reclaim_distance_atr
                    )
                else:
                    pullback_depth_atr = float("nan")

                continuation_entry = (
                    bars_since_breakout >= 1
                    and previous_close <= opening_range_low
                    and closes[position] < lows[position - 1]
                    and highs[position] <= opening_range_low + breakout_buffer
                )
                if continuation_entry:
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_short_continuation_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

                pullback_entry = (
                    not short_pullback_used
                    and bars_since_breakout >= 1
                    and (cfg.retest_max_bars <= 0 or bars_since_breakout <= cfg.retest_max_bars)
                    and highs[position] >= opening_range_low - breakout_buffer
                    and closes[position] <= opening_range_low
                    and closes[position] <= previous_close
                )
                if pullback_entry:
                    short_pullback_used = True
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_short_pullback_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

                reclaim_entry = (
                    short_level_lost
                    and previous_close >= opening_range_low
                    and closes[position] < short_breakout_level
                )
                if reclaim_entry:
                    short_level_lost = False
                    setup_count += 1
                    features.iat[position, features.columns.get_loc("opening_range_short_reclaim_entry")] = 1.0
                    features.iat[position, features.columns.get_loc("opening_range_intraday_setup_count_today")] = (
                        float(setup_count)
                    )
                    features.iat[position, features.columns.get_loc("opening_range_intraday_first_setup_of_day")] = (
                        1.0 if setup_count == 1 else 0.0
                    )

            long_retest = (
                active_long_deadline is not None
                and active_long_breakout_count is not None
                and lows[position] <= opening_range_high
                and closes[position] >= opening_range_high
            )
            if long_retest:
                features.iat[position, features.columns.get_loc("opening_range_long_retest_entry")] = 1.0
                features.iat[
                    position,
                    features.columns.get_loc("opening_range_breakout_count_today"),
                ] = active_long_breakout_count
                features.iat[position, features.columns.get_loc("opening_range_first_breakout_of_day")] = (
                    1.0 if active_long_breakout_count == 1 else 0.0
                )
                active_long_deadline = None
                active_long_breakout_count = None
                continue

            short_retest = (
                active_short_deadline is not None
                and active_short_breakout_count is not None
                and highs[position] >= opening_range_low
                and closes[position] <= opening_range_low
            )
            if short_retest:
                features.iat[position, features.columns.get_loc("opening_range_short_retest_entry")] = 1.0
                features.iat[
                    position,
                    features.columns.get_loc("opening_range_breakout_count_today"),
                ] = active_short_breakout_count
                features.iat[position, features.columns.get_loc("opening_range_first_breakout_of_day")] = (
                    1.0 if active_short_breakout_count == 1 else 0.0
                )
                active_short_deadline = None
                active_short_breakout_count = None

    return features


def _intraday_contextual_features(
    frame: pd.DataFrame,
    base_features: pd.DataFrame,
    atr: pd.Series,
    *,
    cfg: DeterministicFeatureConfig,
) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    defaults = {
        "context_bias_side": 0.0,
        "context_long_pullback_entry": 0.0,
        "context_short_pullback_entry": 0.0,
        "context_long_reclaim_entry": 0.0,
        "context_short_reclaim_entry": 0.0,
        "context_long_session_trend_entry": 0.0,
        "context_short_session_trend_entry": 0.0,
        "context_support_level": np.nan,
        "context_trigger_level": np.nan,
        "context_support_distance_atr": np.nan,
        "context_trigger_distance_atr": np.nan,
        "context_session_high_so_far": np.nan,
        "context_session_low_so_far": np.nan,
        "context_session_range_width_atr": np.nan,
        "context_setup_count_today": np.nan,
        "context_first_setup_of_day": np.nan,
    }
    for column, default in defaults.items():
        features[column] = default

    if frame.empty:
        return features

    _, in_session, session_day, _ = _session_components(
        frame.index,
        session_start_hour_utc=cfg.session_start_hour_utc,
        session_start_minute_utc=cfg.session_start_minute_utc,
        session_end_hour_utc=cfg.session_end_hour_utc,
        session_end_minute_utc=cfg.session_end_minute_utc,
    )

    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    opens = frame["open"].to_numpy()
    closes = frame["close"].to_numpy()
    intraday_vwap = base_features["intraday_vwap"].to_numpy()
    ema_20 = base_features["ema_20"].to_numpy()
    ema_50 = base_features["ema_50"].to_numpy()
    opening_range_mid = base_features["opening_range_mid"].to_numpy()
    opening_range_ready = base_features["opening_range_ready"].fillna(0.0).to_numpy()

    unique_session_days = pd.unique(session_day[in_session])
    for current_day in unique_session_days:
        session_positions = np.flatnonzero(in_session & (session_day == current_day))
        tradable_positions = [
            position for position in session_positions if opening_range_ready[position] > 0.0
        ]
        if not tradable_positions:
            continue

        setup_count = 0
        structure_lookback = max(1, int(cfg.session_trend_structure_lookback_bars))
        session_high_so_far = float("-inf")
        session_low_so_far = float("inf")

        for relative_index, position in enumerate(tradable_positions):
            previous_close = closes[position - 1] if position > 0 else np.nan
            previous_vwap = intraday_vwap[position - 1] if position > 0 else np.nan
            previous_ema20 = ema_20[position - 1] if position > 0 else np.nan
            previous_or_mid = opening_range_mid[position - 1] if position > 0 else np.nan
            current_atr = float(atr.iloc[position]) if pd.notna(atr.iloc[position]) else float("nan")
            trigger_buffer = (
                current_atr * cfg.opening_range_breakout_buffer_atr
                if math.isfinite(current_atr) and current_atr > 0.0
                else 0.0
            )

            if not (
                math.isfinite(intraday_vwap[position])
                and math.isfinite(ema_20[position])
                and math.isfinite(ema_50[position])
                and math.isfinite(opening_range_mid[position])
            ):
                continue

            def _long_bias() -> bool:
                conditions: list[bool] = []
                if cfg.use_opening_range_mid_filter:
                    conditions.append(closes[position] > opening_range_mid[position])
                if cfg.use_intraday_vwap_filter:
                    conditions.append(closes[position] > intraday_vwap[position])
                if cfg.use_intraday_ema50_alignment:
                    conditions.append(ema_20[position] >= ema_50[position])
                if not conditions:
                    return closes[position] > ema_20[position]
                return all(conditions)

            def _short_bias() -> bool:
                conditions: list[bool] = []
                if cfg.use_opening_range_mid_filter:
                    conditions.append(closes[position] < opening_range_mid[position])
                if cfg.use_intraday_vwap_filter:
                    conditions.append(closes[position] < intraday_vwap[position])
                if cfg.use_intraday_ema50_alignment:
                    conditions.append(ema_20[position] <= ema_50[position])
                if not conditions:
                    return closes[position] < ema_20[position]
                return all(conditions)

            def _support_anchor(*, is_long: bool) -> float:
                components = [ema_20[position]]
                if cfg.use_opening_range_mid_filter:
                    components.append(opening_range_mid[position])
                if cfg.use_intraday_vwap_filter:
                    components.append(intraday_vwap[position])
                return max(components) if is_long else min(components)

            def _previous_structure_anchor(*, is_long: bool) -> float:
                components = [previous_ema20]
                if cfg.use_opening_range_mid_filter and math.isfinite(previous_or_mid):
                    components.append(previous_or_mid)
                if cfg.use_intraday_vwap_filter and math.isfinite(previous_vwap):
                    components.append(previous_vwap)
                return (max(components) if is_long else min(components)) if components else np.nan

            def _reclaim_anchor(*, is_long: bool) -> float:
                components: list[float] = []
                if cfg.use_opening_range_mid_filter:
                    components.append(opening_range_mid[position])
                if cfg.use_intraday_vwap_filter:
                    components.append(intraday_vwap[position])
                if not components:
                    components.append(ema_20[position])
                return max(components) if is_long else min(components)

            def _previous_reclaim_anchor(*, is_long: bool) -> float:
                components: list[float] = []
                if cfg.use_opening_range_mid_filter and math.isfinite(previous_or_mid):
                    components.append(previous_or_mid)
                if cfg.use_intraday_vwap_filter and math.isfinite(previous_vwap):
                    components.append(previous_vwap)
                if not components and math.isfinite(previous_ema20):
                    components.append(previous_ema20)
                if not components:
                    return np.nan
                return max(components) if is_long else min(components)

            session_high_so_far = max(session_high_so_far, highs[position])
            session_low_so_far = min(session_low_so_far, lows[position])
            features.iat[position, features.columns.get_loc("context_session_high_so_far")] = session_high_so_far
            features.iat[position, features.columns.get_loc("context_session_low_so_far")] = session_low_so_far
            if math.isfinite(current_atr) and current_atr > 0.0:
                features.iat[
                    position,
                    features.columns.get_loc("context_session_range_width_atr"),
                ] = (session_high_so_far - session_low_so_far) / current_atr

            long_bias = _long_bias()
            short_bias = _short_bias()
            bias_side = 1.0 if long_bias else (-1.0 if short_bias else 0.0)
            features.iat[position, features.columns.get_loc("context_bias_side")] = bias_side

            long_support = _support_anchor(is_long=True)
            short_resistance = _support_anchor(is_long=False)
            if math.isfinite(current_atr) and current_atr > 0.0:
                if long_bias:
                    features.iat[
                        position,
                        features.columns.get_loc("context_support_level"),
                    ] = long_support
                    features.iat[
                        position,
                        features.columns.get_loc("context_support_distance_atr"),
                    ] = max(0.0, (closes[position] - long_support) / current_atr)
                if short_bias:
                    features.iat[
                        position,
                        features.columns.get_loc("context_support_level"),
                    ] = short_resistance
                    features.iat[
                        position,
                        features.columns.get_loc("context_support_distance_atr"),
                    ] = max(0.0, (short_resistance - closes[position]) / current_atr)

            if relative_index == 0:
                continue

            previous_long_structure_anchor = _previous_structure_anchor(is_long=True)
            previous_short_structure_anchor = _previous_structure_anchor(is_long=False)
            long_pullback_entry = bool(
                long_bias
                and math.isfinite(previous_long_structure_anchor)
                and previous_close > previous_long_structure_anchor
                and lows[position] <= long_support + trigger_buffer
                and closes[position] > long_support
                and closes[position] > opens[position]
            )
            if long_pullback_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_long_pullback_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = long_support
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (closes[position] - long_support) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

            short_pullback_entry = bool(
                short_bias
                and math.isfinite(previous_short_structure_anchor)
                and previous_close < previous_short_structure_anchor
                and highs[position] >= short_resistance - trigger_buffer
                and closes[position] < short_resistance
                and closes[position] < opens[position]
            )
            if short_pullback_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_short_pullback_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = short_resistance
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (short_resistance - closes[position]) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

            long_reclaim_anchor = _reclaim_anchor(is_long=True)
            short_reclaim_anchor = _reclaim_anchor(is_long=False)
            previous_long_reclaim_anchor = _previous_reclaim_anchor(is_long=True)
            previous_short_reclaim_anchor = _previous_reclaim_anchor(is_long=False)

            long_reclaim_entry = bool(
                long_bias
                and math.isfinite(previous_long_reclaim_anchor)
                and previous_close <= previous_long_reclaim_anchor
                and lows[position] <= long_reclaim_anchor + trigger_buffer
                and closes[position] > long_reclaim_anchor
                and closes[position] > ema_20[position]
                and closes[position] > opens[position]
            )
            if long_reclaim_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_long_reclaim_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = long_reclaim_anchor
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (closes[position] - long_reclaim_anchor) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

            short_reclaim_entry = bool(
                short_bias
                and math.isfinite(previous_short_reclaim_anchor)
                and previous_close >= previous_short_reclaim_anchor
                and highs[position] >= short_reclaim_anchor - trigger_buffer
                and closes[position] < short_reclaim_anchor
                and closes[position] < ema_20[position]
                and closes[position] < opens[position]
            )
            if short_reclaim_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_short_reclaim_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = short_reclaim_anchor
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (short_reclaim_anchor - closes[position]) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

            if relative_index < structure_lookback:
                continue

            recent_positions = tradable_positions[max(0, relative_index - structure_lookback) : relative_index]
            recent_high = float(np.max(highs[recent_positions]))
            recent_low = float(np.min(lows[recent_positions]))
            compression_width_atr = (
                (recent_high - recent_low) / current_atr
                if math.isfinite(current_atr) and current_atr > 0.0
                else float("nan")
            )
            recent_closes = closes[recent_positions]
            recent_or_mid = opening_range_mid[recent_positions]
            recent_vwap = intraday_vwap[recent_positions]
            compression_ok = (
                not math.isfinite(compression_width_atr)
                or cfg.maximum_context_compression_width_atr <= 0.0
                or compression_width_atr <= cfg.maximum_context_compression_width_atr
            )
            or_mid_structure_ok = (
                not cfg.require_context_or_mid_structure or np.all(recent_closes > recent_or_mid)
            )
            short_or_mid_structure_ok = (
                not cfg.require_context_or_mid_structure or np.all(recent_closes < recent_or_mid)
            )
            vwap_structure_ok = (
                not cfg.require_context_vwap_structure or np.all(recent_closes > recent_vwap)
            )
            short_vwap_structure_ok = (
                not cfg.require_context_vwap_structure or np.all(recent_closes < recent_vwap)
            )

            long_session_trend_entry = bool(
                long_bias
                and closes[position] > recent_high
                and closes[position] > opens[position]
                and or_mid_structure_ok
                and vwap_structure_ok
                and compression_ok
            )
            if long_session_trend_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_long_session_trend_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = recent_high
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (closes[position] - recent_high) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

            short_session_trend_entry = bool(
                short_bias
                and closes[position] < recent_low
                and closes[position] < opens[position]
                and short_or_mid_structure_ok
                and short_vwap_structure_ok
                and compression_ok
            )
            if short_session_trend_entry:
                setup_count += 1
                features.iat[position, features.columns.get_loc("context_short_session_trend_entry")] = 1.0
                features.iat[position, features.columns.get_loc("context_trigger_level")] = recent_low
                if math.isfinite(current_atr) and current_atr > 0.0:
                    features.iat[
                        position,
                        features.columns.get_loc("context_trigger_distance_atr"),
                    ] = max(0.0, (recent_low - closes[position]) / current_atr)
                features.iat[position, features.columns.get_loc("context_setup_count_today")] = float(setup_count)
                features.iat[position, features.columns.get_loc("context_first_setup_of_day")] = (
                    1.0 if setup_count == 1 else 0.0
                )

    return features


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
