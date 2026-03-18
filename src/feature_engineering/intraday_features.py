"""
Intraday Price Dynamics Features

Captures open-to-close price action, overnight gaps, candlestick structure,
and volume dynamics that are particularly predictive for 1–5 day horizons.
"""

import numpy as np
import pandas as pd


def add_intraday_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add intraday price dynamics and volume features for short-horizon forecasts.

    Requires OHLCV columns.  Adds:

    Price action:
      - open_close_diff:      (Close - Open) / Open   — intraday directional move
      - body_size_pct:        |Close - Open| / Close  — candlestick body magnitude
      - gap_open:             (Open - prev_Close) / prev_Close  — overnight gap
      - upper_shadow_pct:     (High - max(Open,Close)) / Close  — upper wick
      - lower_shadow_pct:     (min(Open,Close) - Low) / Close   — lower wick
      - open_close_diff_lag1/2, gap_open_lag1/2 — lagged versions

    Volume dynamics:
      - volume_change_1d:     daily volume % change
      - volume_ma20_ratio:    Volume / 20-day avg volume (turnover spike detector)
      - volume_ma20_ratio_lag1
    """
    f = data.copy()

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(f.columns):
        return f

    open_ = f["Open"].astype(float)
    high = f["High"].astype(float)
    low = f["Low"].astype(float)
    close = f["Close"].astype(float)
    prev_close = close.shift(1)

    # ── Intraday price action ──────────────────────────────────────────────────
    f["open_close_diff"] = (close - open_) / open_.replace(0, np.nan)
    f["body_size_pct"] = (close - open_).abs() / close.replace(0, np.nan)
    f["gap_open"] = (open_ - prev_close) / prev_close.replace(0, np.nan)

    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bot = pd.concat([open_, close], axis=1).min(axis=1)
    f["upper_shadow_pct"] = (high - body_top) / close.replace(0, np.nan)
    f["lower_shadow_pct"] = (body_bot - low) / close.replace(0, np.nan)

    # ── Lagged intraday features ───────────────────────────────────────────────
    for lag in [1, 2]:
        f[f"open_close_diff_lag{lag}"] = f["open_close_diff"].shift(lag)
        f[f"gap_open_lag{lag}"] = f["gap_open"].shift(lag)

    # ── Volume dynamics ────────────────────────────────────────────────────────
    if "Volume" in f.columns:
        vol = f["Volume"].astype(float)
        f["volume_change_1d"] = vol.pct_change()
        vol_ma20 = vol.rolling(20, min_periods=5).mean()
        f["volume_ma20_ratio"] = vol / vol_ma20.replace(0, np.nan)
        f["volume_ma20_ratio_lag1"] = f["volume_ma20_ratio"].shift(1)

    return f
