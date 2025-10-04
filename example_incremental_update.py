"""
Example script demonstrating incremental data updates.

This script shows how to use the new incremental update functionality
to keep stock data current without re-downloading all historical data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocess import StockDataLoader
import pandas as pd
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate incremental data updates."""
    
    # Initialize the data loader
    logger.info("Initializing StockDataLoader...")
    loader = StockDataLoader()
    
    # Example 1: Check data freshness
    logger.info("Example 1: Checking data freshness...")
    sample_stocks = ['AAPL', 'MSFT', 'GOOGL']
    
    try:
        freshness_report = loader.check_data_freshness(sample_stocks)
        logger.info("Data Freshness Report:")
        print(freshness_report.to_string(index=False))
        
        # Show which stocks need updates
        needs_update = freshness_report[freshness_report['Needs_Update'] == True]
        if not needs_update.empty:
            logger.info(f"Stocks needing updates: {needs_update['Symbol'].tolist()}")
        else:
            logger.info("All stocks are up to date!")
            
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
    
    # Example 2: Get last update dates
    logger.info("\nExample 2: Getting last update dates...")
    
    for symbol in sample_stocks:
        try:
            last_date = loader.get_last_update_date(symbol)
            if last_date:
                logger.info(f"{symbol}: Last updated on {last_date.date()}")
            else:
                logger.info(f"{symbol}: No data found")
        except Exception as e:
            logger.error(f"Error getting last update date for {symbol}: {e}")
    
    # Example 3: Incremental update
    logger.info("\nExample 3: Performing incremental update...")
    
    try:
        # Update only stocks that need updates
        updated_data = loader.update_stock_data(
            stock_symbols=sample_stocks,
            days_back=3,  # Safety buffer of 3 days
            save_data=True
        )
        
        logger.info(f"Updated data for {len(updated_data)} stocks")
        
        # Show summary of updated data
        for symbol, data in updated_data.items():
            logger.info(f"{symbol}: {data.shape[0]} total records")
            logger.info(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
            
    except Exception as e:
        logger.error(f"Error during incremental update: {e}")
    
    # Example 4: Compare before and after
    logger.info("\nExample 4: Comparing data before and after update...")
    
    try:
        # Check freshness again after update
        post_update_freshness = loader.check_data_freshness(sample_stocks)
        logger.info("Post-update freshness report:")
        print(post_update_freshness.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error checking post-update freshness: {e}")
    
    # Example 5: Simulate daily update workflow
    logger.info("\nExample 5: Simulating daily update workflow...")
    
    def daily_update_workflow():
        """Simulate a daily update workflow."""
        logger.info("Starting daily update workflow...")
        
        # Check which stocks need updates
        freshness = loader.check_data_freshness()
        stale_stocks = freshness[freshness['Needs_Update'] == True]['Symbol'].tolist()
        
        if stale_stocks:
            logger.info(f"Updating {len(stale_stocks)} stale stocks...")
            
            # Update only stale stocks
            updated = loader.update_stock_data(
                stock_symbols=stale_stocks,
                days_back=2,  # Conservative safety buffer
                save_data=True
            )
            
            logger.info(f"Daily update completed: {len(updated)} stocks updated")
            
            # Generate summary report
            summary = loader.get_data_summary(updated)
            logger.info("Updated stocks summary:")
            print(summary.to_string(index=False))
            
        else:
            logger.info("All stocks are current - no updates needed")
    
    # Run the daily workflow
    daily_update_workflow()
    
    logger.info("\nIncremental update examples completed successfully!")


def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    
    logger.info("\n=== Advanced Usage Examples ===")
    
    loader = StockDataLoader()
    
    # Example: Update only specific stocks with custom parameters
    logger.info("Example: Custom update parameters...")
    
    custom_stocks = ['AAPL', 'TSLA']  # High-volume stocks that change frequently
    
    try:
        updated_data = loader.update_stock_data(
            stock_symbols=custom_stocks,
            days_back=5,  # Larger safety buffer for volatile stocks
            save_data=True
        )
        
        # Process updated data
        for symbol, data in updated_data.items():
            if not data.empty:
                # Calculate some basic statistics
                latest_price = data['Close'].iloc[-1]
                price_change = data['Close'].pct_change().iloc[-1] * 100
                volume_avg = data['Volume'].tail(5).mean()
                
                logger.info(f"{symbol} - Latest: ${latest_price:.2f}, "
                          f"Change: {price_change:+.2f}%, "
                          f"Avg Volume (5d): {volume_avg:,.0f}")
                
    except Exception as e:
        logger.error(f"Error in advanced usage example: {e}")


if __name__ == "__main__":
    main()
    demonstrate_advanced_usage()
