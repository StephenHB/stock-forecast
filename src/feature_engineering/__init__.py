"""
Feature Engineering Module for Stock Forecasting

This module provides various feature engineering techniques for time series data,
specifically designed for stock price forecasting.

Available Feature Engineering Methods:
- FourierTransformer: Extracts frequency domain features using FFT
- LagFeatures: Creates lagged features for time series
- RollingFeatures: Computes rolling window statistics
- TechnicalIndicators: Implements technical analysis indicators
- TimeFeatures: Extracts time-based features (seasonality, trends)
- DifferenceFeatures: Computes various difference transformations

Usage:
    from src.feature_engineering import FourierTransformer
    from src.feature_engineering import create_feature_pipeline
    
    # Single transformer
    fourier = FourierTransformer(n_components=10)
    features = fourier.fit_transform(data)
    
    # Feature pipeline
    pipeline = create_feature_pipeline([
        ('fourier', FourierTransformer(n_components=10)),
        ('lags', LagFeatures(lags=[1, 5, 10])),
        ('rolling', RollingFeatures(windows=[5, 10, 20]))
    ])
    features = pipeline.fit_transform(data)
"""

from .base import BaseFeatureTransformer, FeaturePipeline
from .fourier_transformer import FourierTransformer
from .lag_features import LagFeatures
from .rolling_features import RollingFeatures
from .technical_indicators import TechnicalIndicators
from .time_features import TimeFeatures
from .difference_features import DifferenceFeatures
from .daily_volatility_features import DailyVolatilityFeatures
from .horizon_features import add_medium_term_features, add_long_term_features
from .intraday_features import add_intraday_features
from .macro_features import add_fomc_features, FOMC_DATES
from .market_features import add_market_features, download_market_reference_data

__all__ = [
    'BaseFeatureTransformer',
    'FeaturePipeline',
    'FourierTransformer',
    'LagFeatures',
    'RollingFeatures',
    'TechnicalIndicators',
    'TimeFeatures',
    'DifferenceFeatures',
    'DailyVolatilityFeatures',
    'add_medium_term_features',
    'add_long_term_features',
    'add_intraday_features',
    'add_fomc_features',
    'FOMC_DATES',
    'add_market_features',
    'download_market_reference_data',
]

def create_feature_pipeline(transformers):
    """
    Create a feature engineering pipeline from a list of transformers.
    
    Args:
        transformers: List of (name, transformer) tuples
        
    Returns:
        FeaturePipeline: Configured feature pipeline
        
    Example:
        pipeline = create_feature_pipeline([
            ('fourier', FourierTransformer(n_components=10)),
            ('lags', LagFeatures(lags=[1, 5, 10])),
            ('rolling', RollingFeatures(windows=[5, 10, 20]))
        ])
    """
    return FeaturePipeline(transformers)
