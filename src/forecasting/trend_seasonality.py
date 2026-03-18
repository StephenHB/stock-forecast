"""
Trend and Seasonality Features for LGBM 2-Stage Forecasting

Provides Prophet-based or moving-average-based trend and seasonality features
for use as LGBM inputs. Uses Prophet when data is sufficient; falls back to
simple moving average methods when data is too small.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Prophet minimum rows: need ~2 cycles for weekly seasonality (2*52=104) or 2 weeks for daily
MIN_PROPHET_ROWS = 60


def _ensure_datetime_index(data: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """Ensure data has DatetimeIndex."""
    if isinstance(data.index, pd.DatetimeIndex):
        return data
    if date_column in data.columns:
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        return df.set_index(date_column)
    raise ValueError("Data must have DatetimeIndex or Date column")


def _add_ma_trend_seasonality(
    data: pd.DataFrame,
    target_col: str,
    trend_window: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add trend and seasonality using moving average (causal, no leakage).

    Trend = rolling mean. Seasonality = residual (close - trend) or simple
    week-of-year / day-of-week component.

    Returns:
        (data with trend_ma, seasonality_ma columns, state dict for forecasting)
    """
    df = data.copy()
    close = df[target_col]

    # Causal trend: rolling mean uses only past and current
    trend = close.rolling(window=trend_window, min_periods=1).mean()
    df["trend_ma"] = trend

    # Seasonality: residual from trend (captures cyclical component)
    # Using min_periods=1 to avoid NaN at start
    seasonality = close - trend
    df["seasonality_ma"] = seasonality.fillna(0)

    state = {
        "method": "ma",
        "trend_window": trend_window,
        "last_trend": trend.iloc[-1] if len(trend.dropna()) > 0 else np.nan,
        "last_seasonality": seasonality.iloc[-1] if len(seasonality.dropna()) > 0 else 0.0,
    }
    return df, state


def _add_prophet_trend_seasonality(
    data: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, Any]:
    """
    Add trend and seasonality using Prophet (when data is sufficient).

    Returns:
        (data with trend_prophet, seasonality_prophet columns, fitted Prophet model)
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed. Use: pip install prophet")
        return _add_ma_trend_seasonality(data, target_col)

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have DatetimeIndex for Prophet")

    # Prophet expects 'ds' and 'y' columns
    prophet_df = pd.DataFrame({
        "ds": df.index,
        "y": df[target_col].values,
    }).dropna()

    if len(prophet_df) < 2:
        return _add_ma_trend_seasonality(data, target_col)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,  # For weekly data, daily doesn't apply
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    model.fit(prophet_df)

    # In-sample prediction (trend + seasonality components)
    forecast = model.predict(prophet_df)
    df["trend_prophet"] = forecast["trend"].values
    # additive_terms = weekly + yearly (and daily if enabled)
    seasonality = forecast.get("additive_terms", pd.Series(0.0, index=forecast.index))
    df["seasonality_prophet"] = seasonality.values

    return df, model


def add_trend_seasonality_features(
    data: pd.DataFrame,
    target_col: str = "Close",
    min_prophet_rows: int = MIN_PROPHET_ROWS,
    date_column: str = "Date",
    is_weekly: bool = False,
) -> Tuple[pd.DataFrame, Any]:
    """
    Add trend and seasonality features for LGBM.

    Uses Prophet when len(data) >= min_prophet_rows; otherwise uses
    moving average (causal, no leakage).

    Args:
        data: Time series with Date index or column
        target_col: Target variable column (e.g. Close)
        min_prophet_rows: Minimum rows to use Prophet
        date_column: Date column name if not index
        is_weekly: True for weekly data (affects MA window)

    Returns:
        (data with trend_* and seasonality_* columns, fitted model or state dict)
    """
    df = _ensure_datetime_index(data.copy(), date_column)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in data")

    n = len(df.dropna(subset=[target_col]))
    trend_window = 8 if is_weekly else 20  # weeks vs days

    if n >= min_prophet_rows:
        try:
            df, model = _add_prophet_trend_seasonality(df, target_col)
            logger.info("Using Prophet for trend/seasonality")
            return df, model
        except Exception as e:
            logger.warning(f"Prophet failed ({e}), falling back to MA")
            return _add_ma_trend_seasonality(df, target_col, trend_window)
    else:
        logger.info(f"Data too small for Prophet (n={n}), using MA trend/seasonality")
        return _add_ma_trend_seasonality(df, target_col, trend_window)


def get_forecast_trend_seasonality(
    model_or_state: Any,
    future_dates: pd.DatetimeIndex,
    method: str = "prophet",
) -> pd.DataFrame:
    """
    Get trend and seasonality for future dates (for LGBM prediction phase).

    For Prophet: uses model.predict() for future dates.
    For MA fallback: uses last known trend, seasonality=0 (or simple extrapolation).

    Args:
        model_or_state: Fitted Prophet model or MA state dict
        future_dates: Dates to forecast
        method: "prophet" or "ma"

    Returns:
        DataFrame with columns trend, seasonality for each future date
    """
    if method == "prophet" and hasattr(model_or_state, "predict"):
        try:
            future_df = pd.DataFrame({"ds": future_dates})
            forecast = model_or_state.predict(future_df)
            trend = forecast["trend"].values
            seasonality = forecast.get("additive_terms", pd.Series(0.0, index=forecast.index)).values
            return pd.DataFrame({
                "trend": trend,
                "seasonality": seasonality,
                "date": future_dates,
            }, index=future_dates)
        except Exception as e:
            logger.warning(f"Prophet forecast failed: {e}")
            # Fallback: repeat last values
            pass

    # MA fallback: use last known values (no future seasonality info)
    state = model_or_state if isinstance(model_or_state, dict) else {}
    last_trend = state.get("last_trend", np.nan)
    last_seasonality = state.get("last_seasonality", 0.0)
    n_future = len(future_dates)
    return pd.DataFrame({
        "trend": np.full(n_future, last_trend),
        "seasonality": np.full(n_future, last_seasonality),
        "date": future_dates,
    }, index=future_dates)


def get_trend_seasonality_column_names(used_prophet: bool) -> Tuple[str, str]:
    """Return (trend_col, seasonality_col) based on method used."""
    if used_prophet:
        return "trend_prophet", "seasonality_prophet"
    return "trend_ma", "seasonality_ma"
