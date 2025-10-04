"""
Rolling Window Features for Time Series

This module implements rolling window feature engineering for time series data,
creating features based on moving statistics over specified windows.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class RollingFeatures(BaseFeatureTransformer):
    """
    Rolling Window Features Transformer for Time Series Data.
    
    Creates features based on rolling window statistics over specified periods.
    This captures local trends, volatility, and other temporal patterns.
    
    Parameters:
    -----------
    windows : list of int, default=[5, 10, 20]
        List of window sizes for rolling calculations
        
    columns : list of str, optional
        Columns to create rolling features for. If None, applies to all numeric columns
        
    statistics : list of str, default=['mean', 'std', 'min', 'max']
        List of statistics to compute for each window
        
    center : bool, default=False
        Whether to center the rolling window
        
    min_periods : int, default=1
        Minimum number of observations in window required to have a value
        
    feature_prefix : str, default="rolling"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        windows: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
        center: bool = False,
        min_periods: int = 1,
        feature_prefix: str = "rolling"
    ):
        super().__init__(feature_prefix)
        self.windows = windows or [5, 10, 20]
        self.columns = columns
        self.statistics = statistics or ['mean', 'std', 'min', 'max']
        self.center = center
        self.min_periods = min_periods
        
        # Validate parameters
        if not isinstance(self.windows, list) or len(self.windows) == 0:
            raise ValueError("windows must be a non-empty list")
        if any(window <= 0 for window in self.windows):
            raise ValueError("All windows must be positive integers")
        
        valid_stats = ['mean', 'std', 'var', 'min', 'max', 'median', 'skew', 'kurt', 'sum', 'count']
        invalid_stats = [stat for stat in self.statistics if stat not in valid_stats]
        if invalid_stats:
            raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options: {valid_stats}")
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that specified columns exist in the input data."""
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
    
    def _get_target_columns(self, X: pd.DataFrame) -> List[str]:
        """Get the columns to create rolling features for."""
        if self.columns is not None:
            return self.columns
        else:
            # Use all numeric columns
            return X.select_dtypes(include=[np.number]).columns.tolist()
    
    def _compute_rolling_statistic(self, series: pd.Series, window: int, statistic: str) -> pd.Series:
        """
        Compute a specific rolling statistic for a series.
        
        Args:
            series: Input time series
            window: Window size
            statistic: Statistic to compute
            
        Returns:
            Series with rolling statistic values
        """
        rolling = series.rolling(
            window=window, 
            center=self.center, 
            min_periods=self.min_periods
        )
        
        if statistic == 'mean':
            return rolling.mean()
        elif statistic == 'std':
            return rolling.std()
        elif statistic == 'var':
            return rolling.var()
        elif statistic == 'min':
            return rolling.min()
        elif statistic == 'max':
            return rolling.max()
        elif statistic == 'median':
            return rolling.median()
        elif statistic == 'skew':
            return rolling.skew()
        elif statistic == 'kurt':
            return rolling.kurt()
        elif statistic == 'sum':
            return rolling.sum()
        elif statistic == 'count':
            return rolling.count()
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating rolling window features.
        
        Args:
            X: Input DataFrame with time series data
            
        Returns:
            DataFrame with rolling window features
        """
        target_columns = self._get_target_columns(X)
        result = X.copy()
        
        for col in target_columns:
            logger.info("Creating rolling features for column: %s", col)
            
            for window in self.windows:
                for statistic in self.statistics:
                    feature_name = self._create_feature_name(f"{col}_{statistic}_{window}")
                    rolling_values = self._compute_rolling_statistic(X[col], window, statistic)
                    result[feature_name] = rolling_values
        
        # Store feature names
        new_columns = [col for col in result.columns if col not in X.columns]
        self.feature_names_ = new_columns
        
        return result
