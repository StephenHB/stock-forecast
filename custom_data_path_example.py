"""
Example showing how to use a custom data directory path.

This script demonstrates how to specify a different data storage location
when initializing the StockDataLoader.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocess import StockDataLoader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Example of using custom data directory paths."""
    
    # Example 1: Use default path (/Users/stephenzhang/Downloads/stock_data)
    logger.info("Example 1: Using default data path")
    loader_default = StockDataLoader()
    logger.info(f"Default data directory: {loader_default.data_dir}")
    
    # Example 2: Use custom data path
    logger.info("\nExample 2: Using custom data path")
    custom_path = "/Users/stephenzhang/Downloads/custom_stock_data"
    loader_custom = StockDataLoader(data_dir=custom_path)
    logger.info(f"Custom data directory: {loader_custom.data_dir}")
    
    # Example 3: Download data to custom location
    logger.info("\nExample 3: Downloading data to custom location")
    sample_stocks = ['AAPL', 'TSLA']
    
    try:
        # Download data to custom location
        stock_data = loader_custom.download_stock_data(
            stock_symbols=sample_stocks,
            start_date='2024-01-01',
            end_date='2024-12-31',
            save_data=True
        )
        
        logger.info(f"Downloaded data for {len(stock_data)} stocks to {custom_path}")
        
        # Check what was saved
        raw_dir = loader_custom.data_dir / 'raw'
        if raw_dir.exists():
            files = list(raw_dir.glob('*.csv'))
            logger.info(f"Files saved in custom location: {[f.name for f in files]}")
        
    except Exception as e:
        logger.error(f"Error downloading to custom location: {e}")
    
    # Example 4: Compare data freshness between locations
    logger.info("\nExample 4: Comparing data freshness between locations")
    
    try:
        # Check freshness in default location
        freshness_default = loader_default.check_data_freshness(['AAPL'])
        logger.info("Default location freshness:")
        print(freshness_default.to_string(index=False))
        
        # Check freshness in custom location
        freshness_custom = loader_custom.check_data_freshness(['AAPL'])
        logger.info("Custom location freshness:")
        print(freshness_custom.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error checking freshness: {e}")


if __name__ == "__main__":
    main()
