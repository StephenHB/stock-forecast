"""
Quick Backtesting Example

A simplified version that runs faster for testing purposes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.forecasting.standalone_backtester import StandaloneBacktester
from src.data_preprocess.stock_data_loader import StockDataLoader
from src.forecasting.weekly_aggregator import WeeklyAggregator
from src.forecasting.dynamic_feature_engineer import DynamicFeatureEngineer
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run quick backtesting example."""
    
    print("🚀 Quick Backtesting Example")
    print("=" * 50)
    
    # Configuration
    STOCK_SYMBOL = 'AAPL'
    TARGET_COLUMN = 'Close'
    
    # Step 1: Load and prepare data
    print(f"\n📥 Step 1: Loading data for {STOCK_SYMBOL}...")
    data_loader = StockDataLoader()
    stock_data = data_loader.load_saved_data([STOCK_SYMBOL])
    daily_data = stock_data[STOCK_SYMBOL].copy()
    
    # Convert to datetime index
    if 'Date' in daily_data.columns:
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        daily_data.set_index('Date', inplace=True)
    
    print(f"✅ Loaded {len(daily_data)} days of data")
    
    # Step 2: Weekly aggregation
    print(f"\n📊 Step 2: Converting to weekly data...")
    weekly_aggregator = WeeklyAggregator(
        price_columns=['Open', 'High', 'Low', 'Close'],
        volume_columns=['Volume']
    )
    weekly_data = weekly_aggregator.aggregate(daily_data)
    print(f"✅ Aggregated to {len(weekly_data)} weeks of data")
    
    # Step 3: Simple feature engineering (just lags)
    print(f"\n🔧 Step 3: Creating simple features...")
    feature_config = {
        'lags': {
            'lags': [1, 2, 4], 
            'columns': ['Close']
        }
    }
    
    feature_engineer = DynamicFeatureEngineer(
        forecast_horizon=1,  # 1 quarter ahead
        target_column=TARGET_COLUMN,
        feature_engineering_config=feature_config
    )
    
    # Create forecasting dataset
    forecasting_data = feature_engineer.create_forecasting_dataset(weekly_data)
    print(f"✅ Created {len(forecasting_data)} rows with {len(forecasting_data.columns)} features")
    
    # Step 4: Quick backtesting (fewer windows)
    print(f"\n🎯 Step 4: Running quick backtesting...")
    
    backtester = StandaloneBacktester(
        initial_train_size=20,  # Larger initial training size
        test_size=1,  # 1 quarter ahead
        step_size=4,  # Move forward 4 quarters each iteration (faster)
        min_train_size=10,  # Smaller minimum training size
        target_column=TARGET_COLUMN,
        forecast_horizon=1
    )
    
    # Run backtesting
    results = backtester.backtest(forecasting_data)
    
    # Step 5: Display results
    print(f"\n📈 Step 5: Backtesting Results")
    print("=" * 50)
    
    overall_metrics = results['overall_metrics']
    print(f"🎯 Overall Performance:")
    print(f"   • MAPE: {overall_metrics['mape']:.4f} ({overall_metrics['mape']*100:.2f}%)")
    print(f"   • RMSE: {overall_metrics['rmse']:.4f}")
    print(f"   • R²: {overall_metrics['r2']:.4f}")
    print(f"   • Total Windows: {results['total_windows']}")
    
    # Display performance summary (first 10 windows)
    print(f"\n📊 Performance by Window (First 10):")
    summary_df = backtester.get_performance_summary()
    print(summary_df.head(10).to_string(index=False))
    
    print(f"\n✅ Quick backtesting completed successfully!")
    print(f"🎉 The model achieved {overall_metrics['mape']*100:.2f}% MAPE on {results['total_windows']} test windows")

if __name__ == "__main__":
    main()
