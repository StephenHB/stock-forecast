"""
Difference Features for Time Series

This module implements difference-based feature engineering for time series data,
creating features based on various difference transformations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class DifferenceFeatures(BaseFeatureTransformer):
    """
    Difference Features Transformer for Time Series Data.
    
    Creates features based on various difference transformations including
    first differences, percentage changes, and seasonal differences.
    
    Parameters:
    -----------
    differences : list of int, default=[1, 2, 3, 5, 10]
        List of difference periods to compute
        
    columns : list of str, optional
        Columns to create difference features for. If None, applies to all numeric columns
        
    include_pct_change : bool, default=True
        Whether to include percentage change features
        
    include_log_diff : bool, default=False
        Whether to include log difference features
        
    include_seasonal_diff : bool, default=False
        Whether to include seasonal difference features
        
    seasonal_period : int, default=252
        Period for seasonal differences (e.g., 252 for yearly in daily data)
        
    feature_prefix : str, default="diff"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        differences: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
        include_pct_change: bool = True,
        include_log_diff: bool = False,
        include_seasonal_diff: bool = False,
        seasonal_period: int = 252,
        feature_prefix: str = "diff"
    ):
        super().__init__(feature_prefix)
        self.differences = differences or [1, 2, 3, 5, 10]
        self.columns = columns
        self.include_pct_change = include_pct_change
        self.include_log_diff = include_log_diff
        self.include_seasonal_diff = include_seasonal_diff
        self.seasonal_period = seasonal_period
        
        # Validate parameters
        if not isinstance(self.differences, list) or len(self.differences) == 0:
            raise ValueError("differences must be a non-empty list")
        if any(diff <= 0 for diff in self.differences):
            raise ValueError("All differences must be positive integers")
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be positive")
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that specified columns exist in the input data."""
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
    
    def _get_target_columns(self, X: pd.DataFrame) -> List[str]:
        """Get the columns to create difference features for."""
        if self.columns is not None:
            return self.columns
        else:
            # Use all numeric columns
            return X.select_dtypes(include=[np.number]).columns.tolist()
    
    def _compute_difference(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Compute difference for a series.
        
        Args:
            series: Input time series
            periods: Number of periods to difference
            
        Returns:
            Series with difference values
        """
        return series.diff(periods=periods)
    
    def _compute_pct_change(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Compute percentage change for a series.
        
        Args:
            series: Input time series
            periods: Number of periods for percentage change
            
        Returns:
            Series with percentage change values
        """
        return series.pct_change(periods=periods)
    
    def _compute_log_diff(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Compute log difference for a series.
        
        Args:
            series: Input time series
            periods: Number of periods for log difference
            
        Returns:
            Series with log difference values
        """
        # Ensure positive values for log
        positive_series = series.where(series > 0, np.nan)
        log_series = np.log(positive_series)
        return log_series.diff(periods=periods)
    
    def _compute_seasonal_diff(self, series: pd.Series, periods: int) -> pd.Series:
        """
        Compute seasonal difference for a series.
        
        Args:
            series: Input time series
            periods: Seasonal period
            
        Returns:
            Series with seasonal difference values
        """
        return series.diff(periods=periods)
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating difference features.
        
        Args:
            X: Input DataFrame with time series data
            
        Returns:
            DataFrame with difference features
        """
        target_columns = self._get_target_columns(X)
        result = X.copy()
        
        for col in target_columns:
            logger.info("Creating difference features for column: %s", col)
            
            # Regular differences
            for diff_period in self.differences:
                # First difference
                diff_name = self._create_feature_name(f"{col}_diff_{diff_period}")
                result[diff_name] = self._compute_difference(X[col], diff_period)
                
                # Percentage change
                if self.include_pct_change:
                    pct_name = self._create_feature_name(f"{col}_pct_{diff_period}")
                    result[pct_name] = self._compute_pct_change(X[col], diff_period)
                
                # Log difference
                if self.include_log_diff:
                    log_name = self._create_feature_name(f"{col}_log_diff_{diff_period}")
                    result[log_name] = self._compute_log_diff(X[col], diff_period)
            
            # Seasonal differences
            if self.include_seasonal_diff:
                seasonal_name = self._create_feature_name(f"{col}_seasonal_diff_{self.seasonal_period}")
                result[seasonal_name] = self._compute_seasonal_diff(X[col], self.seasonal_period)
        
        # Store feature names
        new_columns = [col for col in result.columns if col not in X.columns]
        self.feature_names_ = new_columns
        
        return result
