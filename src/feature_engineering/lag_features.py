"""
Lag Features for Time Series

This module implements lag feature engineering for time series data,
creating features based on previous observations.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class LagFeatures(BaseFeatureTransformer):
    """
    Lag Features Transformer for Time Series Data.
    
    Creates lagged features by shifting the time series data by specified periods.
    This is useful for capturing temporal dependencies in time series forecasting.
    
    Parameters:
    -----------
    lags : list of int, default=[1, 2, 3, 5, 10]
        List of lag periods to create features for
        
    columns : list of str, optional
        Columns to create lag features for. If None, applies to all numeric columns
        
    fill_method : str, default='ffill'
        Method to fill NaN values: 'ffill', 'bfill', 'zero', or 'drop'
        
    feature_prefix : str, default="lag"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        lags: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
        fill_method: str = 'ffill',
        feature_prefix: str = "lag"
    ):
        super().__init__(feature_prefix)
        self.lags = lags or [1, 2, 3, 5, 10]
        self.columns = columns
        self.fill_method = fill_method
        
        # Validate parameters
        if not isinstance(self.lags, list) or len(self.lags) == 0:
            raise ValueError("lags must be a non-empty list")
        if any(lag <= 0 for lag in self.lags):
            raise ValueError("All lags must be positive integers")
        if fill_method not in ['ffill', 'bfill', 'zero', 'drop']:
            raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'zero', 'drop'")
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that specified columns exist in the input data."""
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
    
    def _get_target_columns(self, X: pd.DataFrame) -> List[str]:
        """Get the columns to create lag features for."""
        if self.columns is not None:
            return self.columns
        else:
            # Use all numeric columns
            return X.select_dtypes(include=[np.number]).columns.tolist()
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating lag features.
        
        Args:
            X: Input DataFrame with time series data
            
        Returns:
            DataFrame with lag features
        """
        target_columns = self._get_target_columns(X)
        result = X.copy()
        
        for col in target_columns:
            logger.info("Creating lag features for column: %s", col)
            
            for lag in self.lags:
                lag_col_name = self._create_feature_name(f"{col}_{lag}")
                lag_values = X[col].shift(lag)
                
                # Handle NaN values based on fill_method
                if self.fill_method == 'ffill':
                    lag_values = lag_values.ffill()
                elif self.fill_method == 'bfill':
                    lag_values = lag_values.bfill()
                elif self.fill_method == 'zero':
                    lag_values = lag_values.fillna(0)
                elif self.fill_method == 'drop':
                    # NaN values will remain as NaN
                    pass
                
                result[lag_col_name] = lag_values
        
        # Store feature names
        new_columns = [col for col in result.columns if col not in X.columns]
        self.feature_names_ = new_columns
        
        return result
