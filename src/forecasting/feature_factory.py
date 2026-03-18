"""
Feature factory for horizon-aware daily forecasting.

Three feature tiers keyed on forecast_days:
  Short   (1–5d)  : create_daily_features  — microstructure volatility, short lags,
                     intraday dynamics (open-close diff, gaps, candlestick),
                     FOMC proximity, and broad-market context (SPY/QQQ/VIX/yield)
  Medium  (6–15d) : create_medium_features — technical indicators, multi-day momentum,
                     FOMC proximity, and broad-market context
  Long   (16–30d) : create_long_features   — regime, 52-week levels, cyclic calendar,
                     FOMC proximity, and broad-market context

All three tiers accept an optional `market_data` dict (from
market_features.download_market_reference_data) and a `symbol` string so that
reference-ticker features are skipped when the target *is* the reference ticker.

All three tiers use daily OHLCV data; no weekly aggregation is required for
horizons up to 30 days.  Weekly helpers are retained for backward compatibility.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.feature_engineering import DailyVolatilityFeatures
from src.feature_engineering.horizon_features import (
    add_medium_term_features,
    add_long_term_features,
)
from src.feature_engineering.intraday_features import add_intraday_features
from src.feature_engineering.macro_features import add_fomc_features
from src.feature_engineering.market_features import add_market_features
from src.forecasting.trend_seasonality import (
    add_trend_seasonality_features,
    get_trend_seasonality_column_names,
)


def create_daily_features(
    data: pd.DataFrame,
    market_data: Optional[dict] = None,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Create features for daily (short-horizon 1–5 day) forecasting.

    Includes:
    - Trend/seasonality (Prophet when data sufficient, else MA fallback)
    - Lag, rolling stats, volatility features
    - Intraday dynamics: open-close diff, overnight gap, candlestick shadows,
      volume spike ratios  ← new
    - FOMC proximity: days_to_fomc, fomc_week_ahead/after  ← new
    - Market context: SPY/QQQ return, VIX level/change, 10Y yield  ← new
    """
    f = data.copy()

    # Trend and seasonality (Prophet or MA fallback)
    f, _ = add_trend_seasonality_features(f, target_col="Close", is_weekly=False)
    trend_col, seas_col = get_trend_seasonality_column_names(
        "trend_prophet" in f.columns
    )
    f = f.rename(columns={trend_col: "trend", seas_col: "seasonality"})

    # Lags
    for lag in [1, 2, 3, 5]:
        f[f"close_lag_{lag}"] = f["Close"].shift(lag)

    # Rolling stats
    for w in [5, 10, 20]:
        f[f"close_ma_{w}"] = f["Close"].rolling(window=w).mean()
        f[f"close_std_{w}"] = f["Close"].rolling(window=w).std()

    # Volume
    if "Volume" in f.columns:
        f["volume_ma_5"] = f["Volume"].rolling(window=5).mean()
        f["volume_ratio"] = f["Volume"] / f["volume_ma_5"]

    # Time
    f["day_of_week"] = f.index.dayofweek
    f["month"] = f.index.month

    # Volatility features (research-based)
    daily_vol = DailyVolatilityFeatures(
        return_lags=[1, 2, 3],
        volatility_windows=[5, 10],
        adr_window=20,
    )
    f = daily_vol.fit_transform(f)

    # ── New enhancements ───────────────────────────────────────────────────────
    # 1. Intraday price dynamics (open-close diff, gap, shadows, volume spikes)
    f = add_intraday_features(f)

    # 2. FOMC meeting proximity
    f = add_fomc_features(f)

    # 3. Broad-market context (SPY, QQQ, VIX, 10Y yield)
    f = add_market_features(f, market_data, symbol=symbol)

    return f


def create_medium_features(
    data: pd.DataFrame,
    market_data: Optional[dict] = None,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Create features for medium-horizon (6–15 day) daily forecasting.

    Adds trend/seasonality and the horizon_features.add_medium_term_features set
    (extended lags, RSI, MACD, Bollinger, ATR, Stochastic, CCI, OBV), plus
    FOMC proximity and broad-market context.
    """
    f = data.copy()

    # Trend and seasonality
    f, _ = add_trend_seasonality_features(f, target_col="Close", is_weekly=False)
    trend_col, seas_col = get_trend_seasonality_column_names(
        "trend_prophet" in f.columns
    )
    f = f.rename(columns={trend_col: "trend", seas_col: "seasonality"})

    # Short-term base: lags 1-5, rolling 5/10/20, volume, time
    for lag in [1, 2, 3, 5]:
        f[f"close_lag_{lag}"] = f["Close"].shift(lag)
    for w in [5, 10, 20]:
        f[f"close_ma_{w}"] = f["Close"].rolling(w).mean()
        f[f"close_std_{w}"] = f["Close"].rolling(w).std()
    if "Volume" in f.columns:
        f["volume_ma_5"] = f["Volume"].rolling(5).mean()
        f["volume_ratio"] = f["Volume"] / f["volume_ma_5"]
    f["day_of_week"] = f.index.dayofweek
    f["month"] = f.index.month
    f["quarter"] = f.index.quarter

    # Medium-term technical indicators
    f = add_medium_term_features(f)

    # ── New enhancements ───────────────────────────────────────────────────────
    f = add_fomc_features(f)
    f = add_market_features(f, market_data, symbol=symbol)

    return f


def create_long_features(
    data: pd.DataFrame,
    market_data: Optional[dict] = None,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Create features for long-horizon (16–30 day) daily forecasting.

    Adds trend/seasonality and the full horizon_features.add_long_term_features
    set (50/200-day MA regime, 52-week high/low distance, long-horizon momentum,
    cyclic calendar, volatility-regime ratio), plus FOMC proximity and
    broad-market context.
    """
    f = data.copy()

    # Trend and seasonality
    f, _ = add_trend_seasonality_features(f, target_col="Close", is_weekly=False)
    trend_col, seas_col = get_trend_seasonality_column_names(
        "trend_prophet" in f.columns
    )
    f = f.rename(columns={trend_col: "trend", seas_col: "seasonality"})

    # Short-term base
    for lag in [1, 2, 3, 5]:
        f[f"close_lag_{lag}"] = f["Close"].shift(lag)
    for w in [5, 10, 20]:
        f[f"close_ma_{w}"] = f["Close"].rolling(w).mean()
        f[f"close_std_{w}"] = f["Close"].rolling(w).std()
    if "Volume" in f.columns:
        f["volume_ma_5"] = f["Volume"].rolling(5).mean()
        f["volume_ratio"] = f["Volume"] / f["volume_ma_5"]
    f["day_of_week"] = f.index.dayofweek
    f["month"] = f.index.month
    f["quarter"] = f.index.quarter

    # Long-term additions (includes medium-term internally)
    f = add_long_term_features(f)

    # ── New enhancements ───────────────────────────────────────────────────────
    f = add_fomc_features(f)
    f = add_market_features(f, market_data, symbol=symbol)

    return f


def create_weekly_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create features for weekly forecasting with trend/seasonality."""
    f = data.copy()
    # Trend and seasonality (Prophet or MA fallback)
    f, _ = add_trend_seasonality_features(f, target_col="Close", is_weekly=True)
    trend_col, seas_col = get_trend_seasonality_column_names(
        "trend_prophet" in f.columns
    )
    f = f.rename(columns={trend_col: "trend", seas_col: "seasonality"})
    for lag in [1, 2, 4]:
        f[f"close_lag_{lag}"] = f["Close"].shift(lag)
    for w in [4, 8]:
        f[f"close_ma_{w}"] = f["Close"].rolling(window=w).mean()
        f[f"close_std_{w}"] = f["Close"].rolling(window=w).std()
    f["price_change_1w"] = f["Close"].pct_change(1)
    f["price_change_4w"] = f["Close"].pct_change(4)
    if "Volume" in f.columns:
        f["volume_ma_4"] = f["Volume"].rolling(window=4).mean()
        f["volume_ratio"] = f["Volume"] / f["volume_ma_4"]
    f["week_of_year"] = f.index.isocalendar().week
    f["month"] = f.index.month
    f["quarter"] = f.index.quarter
    return f


def create_daily_targets(data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Create target = Close price N business days ahead."""
    t = data.copy()
    t[f"target_{horizon_days}d"] = t["Close"].shift(-horizon_days)
    return t


def create_weekly_targets(data: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    """Create target = Close price N weeks ahead."""
    t = data.copy()
    t[f"target_{horizon_weeks}w"] = t["Close"].shift(-horizon_weeks)
    t[f"target_{horizon_weeks}w_pct"] = (
        (t["Close"].shift(-horizon_weeks) - t["Close"]) / t["Close"] * 100
    )
    return t


def get_feature_columns(
    data: pd.DataFrame, target_col: str
) -> List[str]:
    """Get numeric feature columns excluding target."""
    return [
        c for c in data.columns
        if c != target_col
        and not c.startswith("target_")
        and pd.api.types.is_numeric_dtype(data[c])
    ]
