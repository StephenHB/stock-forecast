"""
Example: Standalone Backtesting

This script demonstrates how to use the standalone backtester
without the complete forecasting pipeline.
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
    """Run standalone backtesting example."""
    
    print("🚀 Standalone Backtesting Example")
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
    print(f"📅 Date range: {daily_data.index.min()} to {daily_data.index.max()}")
    
    # Step 2: Weekly aggregation
    print(f"\n📊 Step 2: Converting to weekly data...")
    weekly_aggregator = WeeklyAggregator(
        price_columns=['Open', 'High', 'Low', 'Close'],
        volume_columns=['Volume']
    )
    weekly_data = weekly_aggregator.aggregate(daily_data)
    print(f"✅ Aggregated to {len(weekly_data)} weeks of data")
    
    # Step 3: Feature engineering
    print(f"\n🔧 Step 3: Creating features...")
    feature_config = {
        'lags': {
            'lags': [1, 2, 4, 8], 
            'columns': ['Close']
        },
        'rolling': {
            'windows': [4, 8, 12], 
            'columns': ['Close'], 
            'statistics': ['mean', 'std']
        },
        'technical': {
            'indicators': ['sma', 'ema', 'rsi', 'macd']
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
    
    # Step 4: Standalone backtesting
    print(f"\n🎯 Step 4: Running standalone backtesting...")
    
    backtester = StandaloneBacktester(
        initial_train_size=13,  # 1 year of quarterly data
        test_size=1,  # 1 quarter ahead
        step_size=1,  # Move forward 1 quarter each iteration
        min_train_size=6,  # Minimum 1.5 years of training data
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
    
    # Display performance summary
    print(f"\n📊 Performance by Window:")
    summary_df = backtester.get_performance_summary()
    print(summary_df.to_string(index=False))
    
    # Step 6: Plot results (optional)
    try:
        print(f"\n📊 Generating performance plots...")
        backtester.plot_results()
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    # Step 7: Save results (optional)
    try:
        results_file = f"backtesting_results_{STOCK_SYMBOL}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        backtester.save_results(results_file)
        print(f"💾 Results saved to {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    print(f"\n✅ Standalone backtesting completed successfully!")
    print(f"🎉 The model achieved {overall_metrics['mape']*100:.2f}% MAPE on {results['total_windows']} test windows")

if __name__ == "__main__":
    main()
