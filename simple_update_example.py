"""
Simple example demonstrating incremental data updates.

This script shows the basic usage of the incremental update functionality.
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
    """Simple example of incremental data updates."""
    
    # Initialize the data loader
    loader = StockDataLoader()
    
    # Example 1: Check data freshness for a few stocks
    logger.info("Checking data freshness...")
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    freshness = loader.check_data_freshness(sample_stocks)
    print("Data Freshness Report:")
    print(freshness.to_string(index=False))
    
    # Example 2: Update only stocks that need updates
    logger.info("\nUpdating stale data...")
    
    # Get stocks that need updates
    needs_update = freshness[freshness['Needs_Update'] == True]['Symbol'].tolist()
    
    if needs_update:
        logger.info(f"Updating {len(needs_update)} stocks: {needs_update}")
        
        # Perform incremental update
        updated_data = loader.update_stock_data(
            stock_symbols=needs_update,
            days_back=3,  # Safety buffer
            save_data=True
        )
        
        logger.info(f"Successfully updated {len(updated_data)} stocks")
        
        # Show summary
        for symbol, data in updated_data.items():
            logger.info(f"{symbol}: {len(data)} total records, "
                       f"latest date: {data['Date'].max()}")
    else:
        logger.info("All stocks are up to date!")
    
    # Example 3: Get last update dates
    logger.info("\nLast update dates:")
    for symbol in sample_stocks:
        last_date = loader.get_last_update_date(symbol)
        if last_date:
            logger.info(f"{symbol}: {last_date.date()}")
        else:
            logger.info(f"{symbol}: No data")


if __name__ == "__main__":
    main()
