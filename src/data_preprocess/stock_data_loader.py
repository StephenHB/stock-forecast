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
            data_dir: Directory to save downloaded data. If None, uses './data'.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.data_dir = Path(data_dir) if data_dir else Path('./data')
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
                    
                    # Save individual stock data if requested
                    if save_data:
                        self._save_stock_data(symbol, data, 'raw')
                        
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
    
    def load_saved_data(self, symbol: Optional[str] = None, subdir: str = 'raw') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load previously saved stock data.
        
        Args:
            symbol: Specific stock symbol to load. If None, loads all available data.
            subdir: Subdirectory to load from ('raw' or 'processed').
            
        Returns:
            DataFrame for specific symbol or dictionary of DataFrames for all symbols.
        """
        storage_settings = self.config.get('storage_settings', {})
        data_format = storage_settings.get('data_format', 'csv')
        
        data_dir = self.data_dir / subdir
        
        if symbol:
            # Load specific stock data
            file_path = data_dir / f"{symbol}.{data_format}"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            try:
                if data_format == 'csv':
                    return pd.read_csv(file_path)
                elif data_format == 'parquet':
                    return pd.read_parquet(file_path)
                elif data_format == 'pickle':
                    return pd.read_pickle(file_path)
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                raise
        
        else:
            # Load all available stock data
            stock_data = {}
            pattern = f"*.{data_format}"
            
            for file_path in data_dir.glob(pattern):
                if file_path.name.startswith('combined_'):
                    continue  # Skip combined files
                
                symbol = file_path.stem
                
                try:
                    if data_format == 'csv':
                        data = pd.read_csv(file_path)
                    elif data_format == 'parquet':
                        data = pd.read_parquet(file_path)
                    elif data_format == 'pickle':
                        data = pd.read_pickle(file_path)
                    
                    stock_data[symbol] = data
                    
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
            
            logger.info(f"Loaded data for {len(stock_data)} stocks")
            return stock_data
    
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
