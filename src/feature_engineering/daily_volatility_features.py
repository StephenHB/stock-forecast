"""
Daily Volatility Features for Short-Horizon Forecasting

Creates features that capture daily price changes and volatility for
forecast horizons <= 5 days. Based on research: decomposition of trend
vs fluctuation, OHLC-based volatility estimators, and rolling statistics
of daily returns.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from .base import BaseFeatureTransformer


class DailyVolatilityFeatures(BaseFeatureTransformer):
    """
    Daily volatility and return features for short-horizon forecasting.

    Designed for forecast horizon <= 5 days. Captures:
    - Daily returns (1d, 2d, 3d, 5d)
    - Rolling volatility (std of returns)
    - High-low range (intraday volatility proxy)
    - Parkinson volatility (log(High/Low))
    - Average daily range (ADR)
    """

    def __init__(
        self,
        return_lags: Optional[List[int]] = None,
        volatility_windows: Optional[List[int]] = None,
        adr_window: int = 20,
        feature_prefix: str = "daily_vol",
    ):
        """
        Initialize daily volatility feature transformer.

        Args:
            return_lags: Lags for daily return features (default: [1, 2, 3, 5])
            volatility_windows: Windows for rolling volatility (default: [5, 10, 20])
            adr_window: Window for average daily range (default: 20)
            feature_prefix: Prefix for feature names
        """
        super().__init__(feature_prefix)
        self.return_lags = return_lags or [1, 2, 3, 5]
        self.volatility_windows = volatility_windows or [5, 10, 20]
        self.adr_window = adr_window

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create daily volatility features from OHLCV data."""
        result = X.copy()

        if "Close" not in result.columns:
            raise ValueError("Data must contain 'Close' column")
        close = result["Close"]

        # Daily returns
        returns = close.pct_change()
        for lag in self.return_lags:
            name = self._create_feature_name(f"return_{lag}d")
            result[name] = returns.shift(lag - 1)

        # Rolling volatility (std of returns)
        for w in self.volatility_windows:
            name = self._create_feature_name(f"vol_{w}d")
            result[name] = returns.rolling(window=w, min_periods=2).std()

        # High-low range (intraday volatility proxy)
        if "High" in result.columns and "Low" in result.columns:
            result[self._create_feature_name("hl_range_pct")] = (
                result["High"] - result["Low"]
            ) / result["Close"]
            # Parkinson volatility: sqrt(1/(4*ln2) * (ln(H/L))^2)
            hl_ratio = np.log(result["High"] / result["Low"].replace(0, np.nan))
            park = hl_ratio ** 2 / (4 * np.log(2))
            result[self._create_feature_name("parkinson_vol")] = np.sqrt(park)
            # Average daily range
            daily_range = result["High"] - result["Low"]
            result[self._create_feature_name("adr")] = daily_range.rolling(
                window=self.adr_window, min_periods=1
            ).mean()

        # Volume-price dynamics (if Volume exists)
        if "Volume" in result.columns:
            result[self._create_feature_name("vol_return")] = (
                returns * np.log1p(result["Volume"])
            )

        new_cols = [c for c in result.columns if c not in X.columns]
        self.feature_names_ = new_cols
        return result
