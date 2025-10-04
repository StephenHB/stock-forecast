"""
Example usage of the StockDataLoader and data preprocessing utilities.

This script demonstrates how to download, clean, and preprocess stock data
using the implemented modules.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocess import StockDataLoader, validate_stock_data, clean_stock_data, calculate_technical_indicators, create_features
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate stock data loading and preprocessing."""
    
    # Initialize the data loader
    logger.info("Initializing StockDataLoader...")
    loader = StockDataLoader()
    
    # Example 1: Download data for a few stocks
    logger.info("Example 1: Downloading data for a few stocks...")
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        stock_data = loader.download_stock_data(
            stock_symbols=sample_stocks,
            start_date='2023-01-01',
            end_date='2023-12-31',
            save_data=True
        )
        
        logger.info(f"Downloaded data for {len(stock_data)} stocks")
        
        # Display summary for each stock
        for symbol, data in stock_data.items():
            logger.info(f"{symbol}: {data.shape[0]} days of data")
            logger.info(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
            logger.info(f"  Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return
    
    # Example 2: Validate and clean data
    logger.info("\nExample 2: Validating and cleaning data...")
    
    for symbol, data in stock_data.items():
        logger.info(f"\nProcessing {symbol}...")
        
        # Validate data
        validation_results = validate_stock_data(data)
        logger.info(f"Validation results: {validation_results}")
        
        # Clean data
        cleaned_data = clean_stock_data(data, remove_outliers=True)
        logger.info(f"Cleaned data shape: {cleaned_data.shape}")
        
        # Calculate technical indicators
        indicators_data = calculate_technical_indicators(cleaned_data)
        logger.info(f"Data with indicators shape: {indicators_data.shape}")
        
        # Create features for machine learning
        features_data = create_features(indicators_data, target_column='Close', lookback_days=5)
        logger.info(f"Data with features shape: {features_data.shape}")
        
        # Save processed data
        loader._save_stock_data(symbol, features_data, 'processed')
        logger.info(f"Saved processed data for {symbol}")
    
    # Example 3: Load saved data
    logger.info("\nExample 3: Loading saved data...")
    
    try:
        # Load specific stock data
        aapl_data = loader.load_saved_data(symbol='AAPL', subdir='raw')
        logger.info(f"Loaded AAPL raw data: {aapl_data.shape}")
        
        # Load all processed data
        all_processed_data = loader.load_saved_data(subdir='processed')
        logger.info(f"Loaded {len(all_processed_data)} processed datasets")
        
    except Exception as e:
        logger.error(f"Error loading saved data: {e}")
    
    # Example 4: Generate data summary
    logger.info("\nExample 4: Generating data summary...")
    
    try:
        summary = loader.get_data_summary(stock_data)
        logger.info("Data Summary:")
        print(summary.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
    
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
