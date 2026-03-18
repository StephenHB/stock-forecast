"""
Horizon-Aware Features for Multi-Horizon Stock Price Forecasting

Provides two feature sets calibrated to forecast horizon length:

  add_medium_term_features  — for 6–15 day horizons
    Adds technical indicators (RSI, MACD, Bollinger, ATR, Stochastic, CCI),
    multi-day momentum, extended lags, and OBV-based volume dynamics that
    become predictive over a 1–3 week look-ahead window.

  add_long_term_features    — for 16–30 day horizons
    Builds on the medium-term set and adds slow-moving regime features:
    50/200-day MA crossover, 52-week high/low distance, long-horizon
    momentum (20/30/60d), cyclic calendar encoding, and volatility-regime ratio.

Both functions operate on a daily OHLCV DataFrame and are designed to be
called by feature_factory.create_medium_features / create_long_features
*after* trend/seasonality features have been added.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period // 2).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────────────
# Medium-term feature set  (6–15 day horizon)
# ─────────────────────────────────────────────────────────────────────────────

def add_medium_term_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add medium-term features for 6–15 day forecast horizons.

    Assumes the input already contains basic lag/MA features from
    create_daily_features.  Adds:
    - Extended lags: 10, 15 days
    - Longer rolling stats: MA(50), std(50)
    - Multi-day momentum: 5d, 10d, 15d % returns
    - RSI(14) and RSI(21)
    - MACD(12,26,9): line, signal, histogram
    - Bollinger Band position and width % (20-day)
    - ATR(14) normalised by price
    - Stochastic %K(14)
    - CCI(20)
    - OBV normalised by 20-day moving average
    - Extended volume ratio (10-day MA)
    """
    f = data.copy()
    close = f["Close"].astype(float)

    # ── Extended lags ────────────────────────────────────────────────────────
    for lag in [10, 15]:
        f[f"close_lag_{lag}"] = close.shift(lag)

    # ── Longer rolling stats ─────────────────────────────────────────────────
    f["close_ma_50"] = close.rolling(50, min_periods=25).mean()
    f["close_std_50"] = close.rolling(50, min_periods=25).std()

    # ── Multi-day momentum ───────────────────────────────────────────────────
    for d in [5, 10, 15]:
        f[f"momentum_{d}d"] = close.pct_change(d)

    # ── RSI ──────────────────────────────────────────────────────────────────
    f["rsi_14"] = _rsi(close, 14)
    f["rsi_21"] = _rsi(close, 21)

    # ── MACD(12,26,9) ────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    f["macd_line"] = macd_line
    f["macd_signal"] = macd_signal
    f["macd_hist"] = macd_line - macd_signal

    # ── Bollinger Band position (20-day) ──────────────────────────────────────
    bb_ma = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    f["bb_position"] = (close - bb_lower) / band_width   # 0 = lower band, 1 = upper
    f["bb_width_pct"] = band_width / bb_ma.replace(0, np.nan)

    # ── ATR(14) normalised by price ───────────────────────────────────────────
    if "High" in f.columns and "Low" in f.columns:
        high = f["High"].astype(float)
        low = f["Low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr14 = tr.rolling(14, min_periods=7).mean()
        f["atr14_pct"] = atr14 / close.replace(0, np.nan)

        # ── Stochastic %K(14) ─────────────────────────────────────────────────
        ll14 = low.rolling(14, min_periods=7).min()
        hh14 = high.rolling(14, min_periods=7).max()
        f["stoch_k"] = 100 * (close - ll14) / (hh14 - ll14).replace(0, np.nan)

        # ── CCI(20) ───────────────────────────────────────────────────────────
        tp = (high + low + close) / 3
        tp_ma = tp.rolling(20, min_periods=10).mean()
        tp_mad = tp.rolling(20, min_periods=10).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        f["cci_20"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    # ── Volume features ───────────────────────────────────────────────────────
    if "Volume" in f.columns:
        vol = f["Volume"].astype(float)
        vol_ma10 = vol.rolling(10, min_periods=5).mean()
        f["volume_ratio_10"] = vol / vol_ma10.replace(0, np.nan)

        # OBV normalised so scale doesn't explode over long histories
        price_change = close.diff()
        obv_raw = pd.Series(
            np.where(price_change > 0, vol, np.where(price_change < 0, -vol, 0.0)),
            index=f.index,
        ).cumsum()
        obv_ma = obv_raw.rolling(20, min_periods=5).mean()
        f["obv_normalised"] = obv_raw / obv_ma.replace(0, np.nan)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# Long-term feature set  (16–30 day horizon)
# ─────────────────────────────────────────────────────────────────────────────

def add_long_term_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add long-term features for 16–30 day forecast horizons.

    Calls add_medium_term_features first, then appends:
    - Extended lags: 20, 30 days
    - Very-long rolling stats: MA(100), std(100)
    - Long-horizon momentum: 20d, 30d, 60d % returns
    - 50/200-day MA ratio (golden/death-cross proximity) and regime flag
    - Close / MA(200) distance (how far above/below the long-term trend)
    - 52-week (252-day) high/low distance
    - Cyclic calendar encoding: month_sin/cos, quarter_sin/cos
    - Volatility-regime ratio: 30-day vol / 60-day vol
    """
    f = add_medium_term_features(data)
    close = f["Close"].astype(float)

    # ── Extended lags ────────────────────────────────────────────────────────
    for lag in [20, 30]:
        f[f"close_lag_{lag}"] = close.shift(lag)

    # ── Very-long rolling stats ───────────────────────────────────────────────
    f["close_ma_100"] = close.rolling(100, min_periods=50).mean()
    f["close_std_100"] = close.rolling(100, min_periods=50).std()

    # ── Long-horizon momentum ────────────────────────────────────────────────
    for d in [20, 30, 60]:
        f[f"momentum_{d}d"] = close.pct_change(d)

    # ── 50/200-day MA regime ──────────────────────────────────────────────────
    ma50 = close.rolling(50, min_periods=25).mean()
    ma200 = close.rolling(200, min_periods=100).mean()
    f["ma50_200_ratio"] = (ma50 / ma200.replace(0, np.nan)) - 1.0
    f["close_ma200_dist"] = (close / ma200.replace(0, np.nan)) - 1.0
    f["above_ma200"] = (close > ma200).astype(float)

    # ── 52-week high/low distance ─────────────────────────────────────────────
    high_252 = close.rolling(252, min_periods=60).max()
    low_252 = close.rolling(252, min_periods=60).min()
    f["dist_52w_high"] = (close - high_252) / high_252.replace(0, np.nan)
    f["dist_52w_low"] = (close - low_252) / low_252.replace(0, np.nan)

    # ── Cyclic calendar encoding ──────────────────────────────────────────────
    month = f.index.month
    f["month_sin"] = np.sin(2 * np.pi * month / 12)
    f["month_cos"] = np.cos(2 * np.pi * month / 12)
    quarter = f.index.quarter
    f["quarter_sin"] = np.sin(2 * np.pi * quarter / 4)
    f["quarter_cos"] = np.cos(2 * np.pi * quarter / 4)

    # ── Volatility-regime ratio ───────────────────────────────────────────────
    returns = close.pct_change()
    vol_30 = returns.rolling(30, min_periods=10).std()
    vol_60 = returns.rolling(60, min_periods=20).std()
    f["vol_30d"] = vol_30
    f["vol_60d"] = vol_60
    f["vol_regime_ratio"] = vol_30 / vol_60.replace(0, np.nan)

    return f
