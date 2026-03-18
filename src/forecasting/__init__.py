"""
Stock Forecasting Module

This module provides comprehensive stock forecasting capabilities using LightGBM
with weekly data aggregation, dynamic feature engineering, and time series backtesting.

Available Components:
- WeeklyAggregator: Aggregates daily stock data to weekly frequency
- DynamicFeatureEngineer: Creates features for multi-step ahead forecasting
- LightGBMForecaster: LightGBM-based forecasting model
- TimeSeriesBacktester: Backtesting framework with MAPE validation
- ForecastingPipeline: Complete end-to-end forecasting pipeline

Usage:
    from src.forecasting import ForecastingPipeline
    
    # Create and run forecasting pipeline
    pipeline = ForecastingPipeline(
        target_column='Close',
        forecast_horizon=4,  # 4 weeks ahead
        backtest_windows=12  # 12 months of backtesting
    )
    
    results = pipeline.fit_predict(stock_data)
"""

from src.forecasting.weekly_aggregator import WeeklyAggregator
from src.forecasting.dynamic_feature_engineer import DynamicFeatureEngineer
from src.forecasting.lgbm_forecaster import LightGBMForecaster
from src.forecasting.time_series_backtester import TimeSeriesBacktester
from src.forecasting.standalone_backtester import StandaloneBacktester
from src.forecasting.forecasting_pipeline import ForecastingPipeline
from src.forecasting.trading_simulator import (
    run_simulation,
    run_multi_stock_simulation,
    SimulationResult,
)
from src.forecasting.trend_seasonality import (
    add_trend_seasonality_features,
    get_forecast_trend_seasonality,
)

__all__ = [
    'WeeklyAggregator',
    'DynamicFeatureEngineer', 
    'LightGBMForecaster',
    'TimeSeriesBacktester',
    'StandaloneBacktester',
    'ForecastingPipeline',
    'run_simulation',
    'run_multi_stock_simulation',
    'SimulationResult',
    'add_trend_seasonality_features',
    'get_forecast_trend_seasonality',
]
