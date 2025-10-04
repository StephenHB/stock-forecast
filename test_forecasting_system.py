"""
Test script for the forecasting system

This script tests the basic functionality of the forecasting pipeline
without requiring full data downloads.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample stock data for testing."""
    
    # Create date range (2 years of daily data)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create sample stock data with some realistic patterns
    np.random.seed(42)
    n_days = len(dates)
    
    # Generate price data with trend and volatility
    base_price = 100
    trend = np.linspace(0, 20, n_days)  # Upward trend
    noise = np.random.normal(0, 2, n_days)  # Random noise
    prices = base_price + trend + noise
    
    # Create OHLC data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.normal(0, 0.5, n_days),
        'High': prices + np.abs(np.random.normal(0, 1, n_days)),
        'Low': prices - np.abs(np.random.normal(0, 1, n_days)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_days),
        'Adj Close': prices
    })
    
    # Ensure High >= Low and High >= Close >= Low
    data['High'] = np.maximum(data['High'], data['Close'])
    data['Low'] = np.minimum(data['Low'], data['Close'])
    data['High'] = np.maximum(data['High'], data['Low'])
    
    data.set_index('Date', inplace=True)
    
    return data

def test_weekly_aggregator():
    """Test the weekly aggregator."""
    logger.info("Testing Weekly Aggregator...")
    
    from src.forecasting.weekly_aggregator import WeeklyAggregator
    
    # Create sample data
    daily_data = create_sample_data()
    logger.info(f"Created sample data: {daily_data.shape}")
    
    # Test aggregator
    aggregator = WeeklyAggregator()
    weekly_data = aggregator.aggregate(daily_data)
    
    logger.info(f"Weekly data shape: {weekly_data.shape}")
    logger.info(f"Weekly data columns: {list(weekly_data.columns)}")
    logger.info(f"Weekly data date range: {weekly_data.index.min()} to {weekly_data.index.max()}")
    
    return weekly_data

def test_dynamic_feature_engineer():
    """Test the dynamic feature engineer."""
    logger.info("Testing Dynamic Feature Engineer...")
    
    from src.forecasting.dynamic_feature_engineer import DynamicFeatureEngineer
    
    # Get weekly data
    weekly_data = test_weekly_aggregator()
    
    # Test feature engineer
    feature_engineer = DynamicFeatureEngineer(
        forecast_horizon=4,
        target_column='Close',
        feature_engineering_config={
            'lags': {'lags': [1, 2], 'columns': ['Close']},
            'rolling': {'windows': [4], 'columns': ['Close'], 'statistics': ['mean']}
        }
    )
    
    forecasting_data = feature_engineer.create_forecasting_dataset(weekly_data)
    
    logger.info(f"Forecasting data shape: {forecasting_data.shape}")
    logger.info(f"Forecasting data columns: {list(forecasting_data.columns)}")
    
    return forecasting_data

def test_lgbm_forecaster():
    """Test the LightGBM forecaster."""
    logger.info("Testing LightGBM Forecaster...")
    
    from src.forecasting.lgbm_forecaster import LightGBMForecaster
    
    # Get forecasting data
    forecasting_data = test_dynamic_feature_engineer()
    
    # Prepare training data
    from src.forecasting.dynamic_feature_engineer import DynamicFeatureEngineer
    feature_engineer = DynamicFeatureEngineer(forecast_horizon=1)
    X, y = feature_engineer.prepare_training_data(forecasting_data, target_horizon=1)
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Test forecaster
    forecaster = LightGBMForecaster(
        forecast_horizon=1,
        hyperparameter_tuning=False  # Skip for speed
    )
    
    forecaster.fit(X, y)
    
    # Make predictions
    predictions = forecaster.predict(X.tail(10))
    logger.info(f"Sample predictions: {predictions[:5]}")
    
    # Get feature importance
    importance = forecaster.get_feature_importance()
    logger.info(f"Top 5 features: {importance.head()['feature'].tolist()}")
    
    return forecaster

def test_backtester():
    """Test the time series backtester."""
    logger.info("Testing Time Series Backtester...")
    
    from src.forecasting.time_series_backtester import TimeSeriesBacktester
    from src.forecasting.lgbm_forecaster import LightGBMForecaster
    
    # Get forecasting data
    forecasting_data = test_dynamic_feature_engineer()
    
    # Prepare data for backtesting
    backtest_data = forecasting_data.dropna()
    
    # Test backtester
    backtester = TimeSeriesBacktester(
        initial_train_size=20,  # Smaller for testing
        test_size=4,
        step_size=4,
        min_train_size=10
    )
    
    results = backtester.backtest(
        data=backtest_data,
        model_class=LightGBMForecaster,
        model_params={
            'forecast_horizon': 1,
            'target_column': 'Close',
            'hyperparameter_tuning': False
        },
        target_column='Close',
        hyperparameter_tuning=False
    )
    
    logger.info(f"Backtest completed: {results['total_iterations']} iterations")
    if results.get('performance_summary'):
        perf = results['performance_summary']
        logger.info(f"MAPE: {perf['primary_metric_mean']:.2f}%")
    
    return results

def test_complete_pipeline():
    """Test the complete forecasting pipeline."""
    logger.info("Testing Complete Forecasting Pipeline...")
    
    from src.forecasting.forecasting_pipeline import ForecastingPipeline
    
    # Create sample data
    daily_data = create_sample_data()
    
    # Test pipeline
    pipeline = ForecastingPipeline(
        target_column='Close',
        forecast_horizon=2,  # 2 weeks ahead
        backtest_windows=6,  # 6 months
        hyperparameter_tuning=False,  # Skip for speed
        feature_engineering_config={
            'lags': {'lags': [1, 2], 'columns': ['Close']},
            'rolling': {'windows': [4], 'columns': ['Close'], 'statistics': ['mean']}
        }
    )
    
    results = pipeline.fit_predict(
        daily_data=daily_data,
        run_backtesting=True
    )
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Predictions: {results['predictions']}")
    
    if results.get('performance_summary'):
        perf = results['performance_summary']
        logger.info(f"Performance: {perf['primary_metric_mean']:.2f}% MAPE")
    
    return results

def main():
    """Run all tests."""
    logger.info("Starting Forecasting System Tests")
    logger.info("=" * 50)
    
    try:
        # Test individual components
        test_weekly_aggregator()
        logger.info("✓ Weekly Aggregator test passed")
        
        test_dynamic_feature_engineer()
        logger.info("✓ Dynamic Feature Engineer test passed")
        
        test_lgbm_forecaster()
        logger.info("✓ LightGBM Forecaster test passed")
        
        test_backtester()
        logger.info("✓ Time Series Backtester test passed")
        
        # Test complete pipeline
        test_complete_pipeline()
        logger.info("✓ Complete Pipeline test passed")
        
        logger.info("=" * 50)
        logger.info("All tests passed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
