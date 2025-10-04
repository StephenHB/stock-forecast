"""
Time-based Features for Time Series

This module implements time-based feature engineering for time series data,
extracting temporal patterns and seasonality information.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class TimeFeatures(BaseFeatureTransformer):
    """
    Time-based Features Transformer for Time Series Data.
    
    Extracts various time-based features from datetime index including
    cyclical encoding, seasonality indicators, and temporal patterns.
    
    Parameters:
    -----------
    features : list of str, default=['year', 'month', 'day', 'dayofweek', 'hour']
        List of time features to extract
        
    cyclical_encoding : bool, default=True
        Whether to apply cyclical encoding to periodic features
        
    include_quarter : bool, default=True
        Whether to include quarter features
        
    include_week_of_year : bool, default=True
        Whether to include week of year features
        
    include_day_of_year : bool, default=True
        Whether to include day of year features
        
    feature_prefix : str, default="time"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        features: Optional[List[str]] = None,
        cyclical_encoding: bool = True,
        include_quarter: bool = True,
        include_week_of_year: bool = True,
        include_day_of_year: bool = True,
        feature_prefix: str = "time"
    ):
        super().__init__(feature_prefix)
        self.features = features or ['year', 'month', 'day', 'dayofweek', 'hour']
        self.cyclical_encoding = cyclical_encoding
        self.include_quarter = include_quarter
        self.include_week_of_year = include_week_of_year
        self.include_day_of_year = include_day_of_year
        
        # Valid time features
        self.valid_features = {
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear',
            'hour', 'minute', 'second', 'quarter', 'is_month_start', 'is_month_end',
            'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end',
            'is_leap_year', 'days_in_month'
        }
        
        # Features that benefit from cyclical encoding
        self.cyclical_features = {'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 
                                'hour', 'minute', 'second', 'quarter'}
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data and check for datetime index."""
        super()._validate_input(X)
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex")
        
        # Validate feature names
        invalid_features = set(self.features) - self.valid_features
        if invalid_features:
            raise ValueError(f"Invalid time features: {invalid_features}. "
                           f"Valid options: {self.valid_features}")
    
    def _extract_time_feature(self, dt_index: pd.DatetimeIndex, feature: str) -> pd.Series:
        """
        Extract a specific time feature from datetime index.
        
        Args:
            dt_index: Datetime index
            feature: Name of the time feature
            
        Returns:
            Series with time feature values
        """
        if feature == 'year':
            return pd.Series(dt_index.year, index=dt_index)
        elif feature == 'month':
            return pd.Series(dt_index.month, index=dt_index)
        elif feature == 'day':
            return pd.Series(dt_index.day, index=dt_index)
        elif feature == 'dayofweek':
            return pd.Series(dt_index.dayofweek, index=dt_index)
        elif feature == 'dayofyear':
            return pd.Series(dt_index.dayofyear, index=dt_index)
        elif feature == 'weekofyear':
            return pd.Series(dt_index.isocalendar().week, index=dt_index)
        elif feature == 'hour':
            return pd.Series(dt_index.hour, index=dt_index)
        elif feature == 'minute':
            return pd.Series(dt_index.minute, index=dt_index)
        elif feature == 'second':
            return pd.Series(dt_index.second, index=dt_index)
        elif feature == 'quarter':
            return pd.Series(dt_index.quarter, index=dt_index)
        elif feature == 'is_month_start':
            return pd.Series(dt_index.is_month_start, index=dt_index).astype(int)
        elif feature == 'is_month_end':
            return pd.Series(dt_index.is_month_end, index=dt_index).astype(int)
        elif feature == 'is_quarter_start':
            return pd.Series(dt_index.is_quarter_start, index=dt_index).astype(int)
        elif feature == 'is_quarter_end':
            return pd.Series(dt_index.is_quarter_end, index=dt_index).astype(int)
        elif feature == 'is_year_start':
            return pd.Series(dt_index.is_year_start, index=dt_index).astype(int)
        elif feature == 'is_year_end':
            return pd.Series(dt_index.is_year_end, index=dt_index).astype(int)
        elif feature == 'is_leap_year':
            return pd.Series(dt_index.is_leap_year, index=dt_index).astype(int)
        elif feature == 'days_in_month':
            return pd.Series(dt_index.days_in_month, index=dt_index)
        else:
            raise ValueError(f"Unknown time feature: {feature}")
    
    def _apply_cyclical_encoding(self, series: pd.Series, feature: str) -> Dict[str, pd.Series]:
        """
        Apply cyclical encoding to a time feature.
        
        Args:
            series: Time feature series
            feature: Name of the time feature
            
        Returns:
            Dictionary with sin and cos encoded features
        """
        # Define maximum values for cyclical encoding
        max_values = {
            'month': 12,
            'day': 31,
            'dayofweek': 7,
            'dayofyear': 366,  # Account for leap years
            'weekofyear': 53,
            'hour': 24,
            'minute': 60,
            'second': 60,
            'quarter': 4
        }
        
        max_val = max_values.get(feature, series.max())
        
        # Apply cyclical encoding
        sin_encoded = np.sin(2 * np.pi * series / max_val)
        cos_encoded = np.cos(2 * np.pi * series / max_val)
        
        return {
            f'{feature}_sin': pd.Series(sin_encoded, index=series.index),
            f'{feature}_cos': pd.Series(cos_encoded, index=series.index)
        }
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by extracting time-based features.
        
        Args:
            X: Input DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with time-based features
        """
        result = X.copy()
        
        # Add additional features if requested
        all_features = list(self.features)
        if self.include_quarter and 'quarter' not in all_features:
            all_features.append('quarter')
        if self.include_week_of_year and 'weekofyear' not in all_features:
            all_features.append('weekofyear')
        if self.include_day_of_year and 'dayofyear' not in all_features:
            all_features.append('dayofyear')
        
        for feature in all_features:
            logger.info("Extracting time feature: %s", feature)
            
            # Extract the time feature
            time_series = self._extract_time_feature(X.index, feature)
            
            if self.cyclical_encoding and feature in self.cyclical_features:
                # Apply cyclical encoding
                encoded_features = self._apply_cyclical_encoding(time_series, feature)
                for encoded_name, encoded_values in encoded_features.items():
                    feature_name = self._create_feature_name(encoded_name)
                    result[feature_name] = encoded_values
            else:
                # Use raw feature
                feature_name = self._create_feature_name(feature)
                result[feature_name] = time_series
        
        # Store feature names
        new_columns = [col for col in result.columns if col not in X.columns]
        self.feature_names_ = new_columns
        
        return result
