"""
Example: Complete Stock Forecasting Pipeline

This example demonstrates how to use the complete forecasting pipeline
for stock price prediction with LightGBM, weekly aggregation, and backtesting.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import our forecasting modules
from src.forecasting import ForecastingPipeline
from src.data_preprocess.stock_data_loader import StockDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the complete forecasting pipeline example."""
    
    logger.info("Starting Stock Forecasting Pipeline Example")
    
    # Step 1: Load stock data
    logger.info("Step 1: Loading stock data...")
    data_loader = StockDataLoader()
    
    # Load data for a specific stock (e.g., AAPL)
    try:
        stock_data = data_loader.load_saved_data(['AAPL'])
        if 'AAPL' not in stock_data or stock_data['AAPL'].empty:
            logger.error("No data found for AAPL. Please download data first.")
            return
        
        daily_data = stock_data['AAPL']
        logger.info(f"Loaded {len(daily_data)} days of data for AAPL")
        logger.info(f"Date range: {daily_data.index.min()} to {daily_data.index.max()}")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Step 2: Configure and run forecasting pipeline
    logger.info("Step 2: Setting up forecasting pipeline...")
    
    # Configure feature engineering
    feature_config = {
        'fourier': {'n_components': 3, 'columns': ['Close']},
        'lags': {'lags': [1, 2, 4], 'columns': ['Close']},
        'rolling': {'windows': [4, 8], 'columns': ['Close'], 'statistics': ['mean', 'std']},
        'technical': {'indicators': ['sma', 'ema', 'rsi'], 'price_column': 'Close'},
        'time': {'features': ['month', 'dayofweek'], 'cyclical_encoding': True},
        'difference': {'differences': [1, 4], 'columns': ['Close'], 'include_pct_change': True}
    }
    
    # Create forecasting pipeline
    pipeline = ForecastingPipeline(
        target_column='Close',
        forecast_horizon=4,  # 4 weeks ahead
        backtest_windows=12,  # 12 months of backtesting
        hyperparameter_tuning=True,
        feature_engineering_config=feature_config,
        lgbm_params={
            'cv_folds': 3,  # Reduce for faster execution
            'random_state': 42
        },
        backtest_params={
            'initial_train_size': 26,  # 6 months
            'test_size': 4,
            'step_size': 4,
            'min_train_size': 13  # 3 months minimum
        }
    )
    
    # Step 3: Run the complete pipeline
    logger.info("Step 3: Running forecasting pipeline...")
    
    try:
        results = pipeline.fit_predict(
            daily_data=daily_data,
            end_date=None,  # Use all available data
            run_backtesting=True
        )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return
    
    # Step 4: Display results
    logger.info("Step 4: Displaying results...")
    
    print("\n" + "="*60)
    print("FORECASTING RESULTS")
    print("="*60)
    
    # Display predictions
    if results['predictions']:
        print("\nMulti-Step Ahead Predictions:")
        print("-" * 40)
        for horizon, prediction in results['predictions'].items():
            print(f"{horizon}-week ahead: ${prediction:.2f}")
    
    # Display performance metrics
    if results.get('performance_summary'):
        perf = results['performance_summary']
        print(f"\nBacktesting Performance:")
        print("-" * 40)
        print(f"Primary Metric ({perf['primary_metric'].upper()}): {perf['primary_metric_mean']:.2f} ± {perf['primary_metric_std']:.2f}")
        print(f"Best Iteration: {perf['best_iteration']}")
        print(f"Total Backtest Windows: {perf['total_iterations']}")
    
    # Display data information
    if results.get('data_info'):
        data_info = results['data_info']
        print(f"\nData Information:")
        print("-" * 40)
        print(f"Weekly Data Shape: {data_info['weekly_data_shape']}")
        print(f"Forecasting Data Shape: {data_info['forecasting_data_shape']}")
        if data_info['date_range']['start']:
            print(f"Date Range: {data_info['date_range']['start'].date()} to {data_info['date_range']['end'].date()}")
    
    # Display feature importance
    if results.get('model_info', {}).get('feature_importance') is not None:
        importance_df = results['model_info']['feature_importance']
        print(f"\nTop 10 Most Important Features:")
        print("-" * 40)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Step 5: Plot results
    logger.info("Step 5: Generating plots...")
    
    try:
        pipeline.plot_results()
        logger.info("Plots generated successfully")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    # Step 6: Save results
    logger.info("Step 6: Saving results...")
    
    try:
        # Save pipeline
        pipeline.save_pipeline('forecasting_results.pkl')
        
        # Save predictions to CSV
        if results['predictions']:
            pred_df = pd.DataFrame([
                {'horizon_weeks': k, 'predicted_price': v} 
                for k, v in results['predictions'].items()
            ])
            pred_df.to_csv('predictions.csv', index=False)
            logger.info("Predictions saved to predictions.csv")
        
        logger.info("Results saved successfully")
        
    except Exception as e:
        logger.warning(f"Could not save results: {e}")
    
    print("\n" + "="*60)
    print("FORECASTING PIPELINE COMPLETED")
    print("="*60)


def run_quick_example():
    """Run a quick example with minimal configuration."""
    
    logger.info("Running Quick Forecasting Example")
    
    # Load data
    data_loader = StockDataLoader()
    stock_data = data_loader.load_saved_data(['AAPL'])
    
    if 'AAPL' not in stock_data or stock_data['AAPL'].empty:
        logger.error("No data found for AAPL")
        return
    
    daily_data = stock_data['AAPL']
    
    # Quick pipeline with minimal features
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
    
    # Run pipeline
    results = pipeline.fit_predict(daily_data, run_backtesting=True)
    
    # Display results
    print("\nQuick Example Results:")
    print("-" * 30)
    for horizon, prediction in results['predictions'].items():
        print(f"{horizon}-week ahead: ${prediction:.2f}")
    
    if results.get('performance_summary'):
        perf = results['performance_summary']
        print(f"MAPE: {perf['primary_metric_mean']:.2f}%")


if __name__ == "__main__":
    # Run the main example
    main()
    
    # Uncomment to run quick example
    # run_quick_example()
