"""
Stock data loader module for downloading and managing stock market data.

This module provides functionality to download stock data from Yahoo Finance
with customizable configuration and data validation.
"""

import os
import yaml
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime, date
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataLoader:
    """
    A class to handle downloading and managing stock market data from Yahoo Finance.
    
    This class provides methods to download stock data based on configuration files,
    validate the data, and save it in various formats.
    """
    
    def __init__(self, config_path: Optional[str] = None, data_dir: Optional[str] = None):
        """
        Initialize the StockDataLoader.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses default config.
            data_dir: Directory to save downloaded data. If None, uses '/Users/stephenzhang/Downloads/stock_data'.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.data_dir = Path(data_dir) if data_dir else Path('/Users/stephenzhang/Downloads/stock_data')
        self.config = self._load_config()
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organized data storage
        (self.data_dir / 'raw').mkdir(exist_ok=True)
        (self.data_dir / 'processed').mkdir(exist_ok=True)
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'stocks_config.yaml')
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get_stock_list(self, custom_stocks: Optional[List[str]] = None) -> List[str]:
        """
        Get the list of stocks to download.
        
        Args:
            custom_stocks: Custom list of stock symbols. If None, uses default from config.
            
        Returns:
            List of stock symbols.
        """
        if custom_stocks:
            logger.info(f"Using custom stock list with {len(custom_stocks)} stocks")
            return custom_stocks
        
        stocks = self.config.get('default_stocks', [])
        logger.info(f"Using default stock list with {len(stocks)} stocks")
        return stocks
    
    def download_stock_data(
        self, 
        stock_symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        interval: Optional[str] = None,
        save_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download stock data for specified symbols.
        
        Args:
            stock_symbols: List of stock symbols to download. If None, uses config default.
            start_date: Start date for data download. If None, uses config default.
            end_date: End date for data download. If None, uses config default.
            interval: Data interval (1d, 5d, 1wk, 1mo, 3mo). If None, uses config default.
            save_data: Whether to save downloaded data to files.
            
        Returns:
            Dictionary with stock symbols as keys and DataFrames as values.
        """
        # Get parameters from config if not provided
        stock_symbols = stock_symbols or self.get_stock_list()
        download_settings = self.config.get('download_settings', {})
        
        start_date = start_date or download_settings.get('start_date', '2020-01-01')
        end_date = end_date or download_settings.get('end_date')
        interval = interval or download_settings.get('interval', '1d')
        
        logger.info(f"Downloading data for {len(stock_symbols)} stocks")
        logger.info(f"Date range: {start_date} to {end_date or 'current'}")
        logger.info(f"Interval: {interval}")
        
        stock_data = {}
        failed_downloads = []
        
        # Download data with progress bar
        for symbol in tqdm(stock_symbols, desc="Downloading stock data"):
            try:
                data = self._download_single_stock(symbol, start_date, end_date, interval)
                if data is not None and not data.empty:
                    stock_data[symbol] = data
                    
                    # Individual files are no longer saved - only combined data
                        
                else:
                    failed_downloads.append(symbol)
                    logger.warning(f"No data downloaded for {symbol}")
                    
            except Exception as e:
                failed_downloads.append(symbol)
                logger.error(f"Failed to download data for {symbol}: {e}")
        
        # Log summary
        logger.info(f"Successfully downloaded data for {len(stock_data)} stocks")
        if failed_downloads:
            logger.warning(f"Failed to download data for {len(failed_downloads)} stocks: {failed_downloads}")
        
        # Save combined data if requested
        if save_data and stock_data:
            self._save_combined_data(stock_data)
        
        return stock_data
    
    def _download_single_stock(
        self, 
        symbol: str, 
        start_date: Union[str, date], 
        end_date: Optional[Union[str, date]], 
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single stock symbol.
        
        Args:
            symbol: Stock symbol to download.
            start_date: Start date for data download.
            end_date: End date for data download.
            interval: Data interval.
            
        Returns:
            DataFrame with stock data or None if download failed.
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=self.config.get('download_settings', {}).get('auto_adjust', True),
                prepost=self.config.get('download_settings', {}).get('prepost', False)
            )
            
            if data.empty:
                return None
            
            # Add stock symbol as a column
            data['Symbol'] = symbol
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns to standard format
            data.columns = data.columns.str.replace(' ', '_')
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            return None
    
    def _save_stock_data(self, symbol: str, data: pd.DataFrame, subdir: str) -> None:
        """
        Save individual stock data to file.
        
        Args:
            symbol: Stock symbol.
            data: Stock data DataFrame.
            subdir: Subdirectory to save data ('raw' or 'processed').
        """
        storage_settings = self.config.get('storage_settings', {})
        data_format = storage_settings.get('data_format', 'csv')
        compression = storage_settings.get('compression')
        
        file_path = self.data_dir / subdir / f"{symbol}.{data_format}"
        
        try:
            if data_format == 'csv':
                data.to_csv(file_path, index=False, compression=compression)
            elif data_format == 'parquet':
                data.to_parquet(file_path, compression=compression)
            elif data_format == 'pickle':
                data.to_pickle(file_path, compression=compression)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
            
            logger.debug(f"Saved {symbol} data to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
    
    def _save_combined_data(self, stock_data: Dict[str, pd.DataFrame]) -> None:
        """
        Save combined stock data to a single file.
        
        Args:
            stock_data: Dictionary with stock symbols as keys and DataFrames as values.
        """
        storage_settings = self.config.get('storage_settings', {})
        data_format = storage_settings.get('data_format', 'csv')
        compression = storage_settings.get('compression')
        
        # Combine all data
        combined_data = pd.concat(stock_data.values(), ignore_index=True)
        
        # Sort by Symbol and Date
        combined_data = combined_data.sort_values(['Symbol', 'Date']).reset_index(drop=True)
        
        file_path = self.data_dir / 'raw' / f"combined_stock_data.{data_format}"
        
        try:
            if data_format == 'csv':
                combined_data.to_csv(file_path, index=False, compression=compression)
            elif data_format == 'parquet':
                combined_data.to_parquet(file_path, compression=compression)
            elif data_format == 'pickle':
                combined_data.to_pickle(file_path, compression=compression)
            
            logger.info(f"Saved combined data to {file_path}")
            logger.info(f"Combined data shape: {combined_data.shape}")
            
        except Exception as e:
            logger.error(f"Error saving combined data: {e}")
    
    def load_saved_data(self, symbol: Optional[Union[str, List[str]]] = None, subdir: str = 'raw') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load previously saved stock data from the combined data file.
        
        Args:
            symbol: Specific stock symbol(s) to load. Can be a string, list of strings, or None.
                   If None, loads all available data.
            subdir: Subdirectory to load from ('raw' or 'processed').
            
        Returns:
            DataFrame for specific symbol or dictionary of DataFrames for all symbols.
        """
        storage_settings = self.config.get('storage_settings', {})
        data_format = storage_settings.get('data_format', 'csv')
        
        data_dir = self.data_dir / subdir
        
        # Always load from combined data file
        combined_file_path = data_dir / f"combined_stock_data.{data_format}"
        
        if not combined_file_path.exists():
            raise FileNotFoundError(f"Combined data file not found: {combined_file_path}")
        
        try:
            # Load combined data
            if data_format == 'csv':
                combined_data = pd.read_csv(combined_file_path)
            elif data_format == 'parquet':
                combined_data = pd.read_parquet(combined_file_path)
            elif data_format == 'pickle':
                combined_data = pd.read_pickle(combined_file_path)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")
            
            if symbol:
                # Handle both single symbol and list of symbols
                if isinstance(symbol, str):
                    # Single symbol
                    symbol_data = combined_data[combined_data['Symbol'] == symbol].copy()
                    if symbol_data.empty:
                        raise FileNotFoundError(f"No data found for symbol: {symbol}")
                    return symbol_data
                elif isinstance(symbol, list):
                    # List of symbols - return dictionary
                    stock_data = {}
                    for sym in symbol:
                        sym_data = combined_data[combined_data['Symbol'] == sym].copy()
                        if not sym_data.empty:
                            stock_data[sym] = sym_data
                        else:
                            logger.warning(f"No data found for symbol: {sym}")
                    if not stock_data:
                        raise FileNotFoundError(f"No data found for any of the symbols: {symbol}")
                    return stock_data
                else:
                    raise ValueError(f"Symbol must be a string, list of strings, or None, got: {type(symbol)}")
            else:
                # Return dictionary of DataFrames for all symbols
                stock_data = {}
                for sym in combined_data['Symbol'].unique():
                    stock_data[sym] = combined_data[combined_data['Symbol'] == sym].copy()
                return stock_data
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_summary(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for downloaded stock data.
        
        Args:
            stock_data: Dictionary with stock symbols as keys and DataFrames as values.
            
        Returns:
            DataFrame with summary statistics for each stock.
        """
        summary_data = []
        
        for symbol, data in stock_data.items():
            if data.empty:
                continue
            
            summary = {
                'Symbol': symbol,
                'Start_Date': data['Date'].min(),
                'End_Date': data['Date'].max(),
                'Total_Days': len(data),
                'Missing_Days': data['Date'].isna().sum(),
                'Avg_Volume': data['Volume'].mean() if 'Volume' in data.columns else None,
                'Avg_Close': data['Close'].mean() if 'Close' in data.columns else None,
                'Price_Range': data['Close'].max() - data['Close'].min() if 'Close' in data.columns else None
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def update_stock_data(
        self, 
        stock_symbols: Optional[List[str]] = None,
        days_back: int = 7,
        save_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Update existing stock data by appending only the latest data.
        
        This function checks the last date in existing data and downloads only
        the new data since that date, then appends it to the existing dataset.
        
        Args:
            stock_symbols: List of stock symbols to update. If None, uses config default.
            days_back: Number of days to look back for updates (safety buffer).
            save_data: Whether to save updated data to files.
            
        Returns:
            Dictionary with updated stock data.
        """
        stock_symbols = stock_symbols or self.get_stock_list()
        updated_data = {}
        
        logger.info(f"Updating data for {len(stock_symbols)} stocks")
        
        for symbol in tqdm(stock_symbols, desc="Updating stock data"):
            try:
                # Load existing data
                existing_data = self._load_existing_data(symbol)
                
                if existing_data is not None and not existing_data.empty:
                    # Get the last date from existing data (convert to timezone-naive)
                    last_date = pd.to_datetime(existing_data['Date']).max()
                    if last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                    
                    start_date = last_date + pd.Timedelta(days=1)
                    
                    # Add safety buffer to ensure we don't miss any data
                    start_date = start_date - pd.Timedelta(days=days_back)
                    
                    logger.info(f"Updating {symbol} from {start_date.date()}")
                    
                    # Download new data
                    new_data = self._download_single_stock(
                        symbol, 
                        start_date=start_date, 
                        end_date=None,  # Current date
                        interval=self.config.get('download_settings', {}).get('interval', '1d')
                    )
                    
                    if new_data is not None and not new_data.empty:
                        # Convert new_data Date column to timezone-naive for comparison
                        new_data = new_data.copy()
                        new_data['Date'] = pd.to_datetime(new_data['Date'])
                        if new_data['Date'].dt.tz is not None:
                            new_data['Date'] = new_data['Date'].dt.tz_localize(None)
                        
                        # Remove any overlapping data (in case of safety buffer)
                        new_data = new_data[new_data['Date'] > last_date]
                        
                        if not new_data.empty:
                            # Combine existing and new data
                            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                            combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                            
                            # Remove duplicates based on Date
                            combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                            
                            updated_data[symbol] = combined_data
                            
                            # Individual files are no longer saved - only combined data
                            
                            logger.info(f"Updated {symbol}: {len(new_data)} new records, total: {len(combined_data)}")
                        else:
                            logger.info(f"No new data available for {symbol}")
                            updated_data[symbol] = existing_data
                    else:
                        logger.warning(f"No new data downloaded for {symbol}")
                        updated_data[symbol] = existing_data
                        
                else:
                    # No existing data, download full dataset
                    logger.info(f"No existing data found for {symbol}, downloading full dataset")
                    full_data = self._download_single_stock(
                        symbol,
                        start_date=self.config.get('download_settings', {}).get('start_date', '2020-01-01'),
                        end_date=None,
                        interval=self.config.get('download_settings', {}).get('interval', '1d')
                    )
                    
                    if full_data is not None and not full_data.empty:
                        updated_data[symbol] = full_data
                        
                        # Individual files are no longer saved - only combined data
                        
                        logger.info(f"Downloaded full dataset for {symbol}: {len(full_data)} records")
                    else:
                        logger.error(f"Failed to download data for {symbol}")
                        
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")
        
        # Save combined updated data (preserve all existing data)
        if save_data and updated_data:
            # Load all existing data to preserve non-updated stocks
            try:
                existing_all_data = self.load_saved_data(subdir='raw')
                # Update with new data
                existing_all_data.update(updated_data)
                self._save_combined_data(existing_all_data)
            except FileNotFoundError:
                # No existing combined data, save only updated data
                self._save_combined_data(updated_data)
        
        logger.info(f"Data update completed for {len(updated_data)} stocks")
        return updated_data
    
    def _load_existing_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load existing data for a specific stock symbol.
        
        Args:
            symbol: Stock symbol to load data for.
            
        Returns:
            DataFrame with existing data or None if not found.
        """
        try:
            data = self.load_saved_data(symbol=symbol, subdir='raw')
            if data is not None and not data.empty:
                # Ensure Date column is properly formatted as datetime
                data = data.copy()
                data['Date'] = pd.to_datetime(data['Date'], utc=True)
                data['Date'] = data['Date'].dt.tz_localize(None)
            return data
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error loading existing data for {symbol}: {e}")
            return None
    
    def get_last_update_date(self, symbol: str) -> Optional[pd.Timestamp]:
        """
        Get the last update date for a specific stock.
        
        Args:
            symbol: Stock symbol to check.
            
        Returns:
            Last update date or None if no data exists.
        """
        try:
            existing_data = self._load_existing_data(symbol)
            if existing_data is not None and not existing_data.empty:
                last_date = pd.to_datetime(existing_data['Date']).max()
                # Convert to timezone-naive for consistency
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                return last_date
            return None
        except Exception as e:
            logger.error(f"Error getting last update date for {symbol}: {e}")
            return None
    
    def check_data_freshness(self, stock_symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Check the freshness of existing data for given stocks.
        
        Args:
            stock_symbols: List of stock symbols to check. If None, uses config default.
            
        Returns:
            DataFrame with freshness information for each stock.
        """
        stock_symbols = stock_symbols or self.get_stock_list()
        freshness_data = []
        
        for symbol in stock_symbols:
            last_date = self.get_last_update_date(symbol)
            current_date = pd.Timestamp.now().normalize()
            
            if last_date is not None:
                # Ensure both dates are timezone-naive for comparison
                if last_date.tz is not None:
                    last_date = last_date.tz_localize(None)
                if current_date.tz is not None:
                    current_date = current_date.tz_localize(None)
                
                days_old = (current_date - last_date).days
                is_weekend = current_date.weekday() >= 5
                
                # Adjust for weekends (markets are closed)
                if is_weekend and days_old <= 2:
                    days_old = 0
                elif is_weekend and days_old <= 3:
                    days_old = days_old - 2
                
                freshness_status = "Fresh" if days_old <= 1 else "Stale" if days_old <= 7 else "Very Stale"
                
                freshness_info = {
                    'Symbol': symbol,
                    'Last_Update': last_date,
                    'Days_Old': days_old,
                    'Status': freshness_status,
                    'Needs_Update': days_old > 1
                }
            else:
                freshness_info = {
                    'Symbol': symbol,
                    'Last_Update': None,
                    'Days_Old': None,
                    'Status': 'No Data',
                    'Needs_Update': True
                }
            
            freshness_data.append(freshness_info)
        
        return pd.DataFrame(freshness_data)
