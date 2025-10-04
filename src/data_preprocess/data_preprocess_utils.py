"""
Data preprocessing utilities for stock market data.

This module provides utility functions for data validation, cleaning,
feature engineering, and technical indicator calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_stock_data(data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Validate stock data for completeness and quality.
    
    Args:
        data: Stock data DataFrame to validate.
        required_columns: List of required columns. If None, uses default columns.
        
    Returns:
        Dictionary with validation results.
    """
    if required_columns is None:
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    validation_results = {
        'has_required_columns': all(col in data.columns for col in required_columns),
        'has_data': not data.empty,
        'has_date_column': 'Date' in data.columns,
        'date_is_datetime': False,
        'no_duplicate_dates': True,
        'no_missing_prices': True,
        'positive_prices': True,
        'valid_high_low': True,
        'valid_volume': True
    }
    
    if not validation_results['has_data']:
        logger.warning("Data is empty")
        return validation_results
    
    # Check date column
    if validation_results['has_date_column']:
        try:
            pd.to_datetime(data['Date'])
            validation_results['date_is_datetime'] = True
        except:
            logger.warning("Date column cannot be converted to datetime")
    
    # Check for duplicate dates
    if validation_results['has_date_column']:
        validation_results['no_duplicate_dates'] = not data['Date'].duplicated().any()
    
    # Check for missing price data
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in data.columns:
            if data[col].isna().any():
                validation_results['no_missing_prices'] = False
                break
    
    # Check for positive prices
    for col in price_columns:
        if col in data.columns:
            if (data[col] <= 0).any():
                validation_results['positive_prices'] = False
                break
    
    # Check High >= Low
    if 'High' in data.columns and 'Low' in data.columns:
        validation_results['valid_high_low'] = (data['High'] >= data['Low']).all()
    
    # Check volume
    if 'Volume' in data.columns:
        validation_results['valid_volume'] = (data['Volume'] >= 0).all()
    
    return validation_results


def clean_stock_data(data: pd.DataFrame, remove_outliers: bool = True, outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Clean stock data by handling missing values and outliers.
    
    Args:
        data: Stock data DataFrame to clean.
        remove_outliers: Whether to remove outliers based on price changes.
        outlier_threshold: Standard deviation threshold for outlier detection.
        
    Returns:
        Cleaned DataFrame.
    """
    cleaned_data = data.copy()
    
    # Convert Date to datetime if not already
    if 'Date' in cleaned_data.columns:
        cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])
    
    # Sort by Date
    cleaned_data = cleaned_data.sort_values('Date').reset_index(drop=True)
    
    # Handle missing values in price columns
    price_columns = ['Open', 'High', 'Low', 'Close']
    for col in price_columns:
        if col in cleaned_data.columns:
            # Forward fill missing prices
            cleaned_data[col] = cleaned_data[col].ffill()
            # If still missing, backward fill
            cleaned_data[col] = cleaned_data[col].bfill()
    
    # Handle missing volume
    if 'Volume' in cleaned_data.columns:
        cleaned_data['Volume'] = cleaned_data['Volume'].fillna(0)
    
    # Remove outliers if requested
    if remove_outliers and 'Close' in cleaned_data.columns:
        # Calculate daily returns
        cleaned_data['Daily_Return'] = cleaned_data['Close'].pct_change()
        
        # Identify outliers based on returns
        mean_return = cleaned_data['Daily_Return'].mean()
        std_return = cleaned_data['Daily_Return'].std()
        
        outlier_mask = np.abs(cleaned_data['Daily_Return'] - mean_return) > (outlier_threshold * std_return)
        
        if outlier_mask.any():
            logger.info(f"Removing {outlier_mask.sum()} outlier observations")
            cleaned_data = cleaned_data[~outlier_mask].reset_index(drop=True)
        
        # Drop the temporary return column
        cleaned_data = cleaned_data.drop('Daily_Return', axis=1)
    
    # Ensure High >= Low
    if 'High' in cleaned_data.columns and 'Low' in cleaned_data.columns:
        invalid_hl = cleaned_data['High'] < cleaned_data['Low']
        if invalid_hl.any():
            logger.warning(f"Fixing {invalid_hl.sum()} invalid High-Low pairs")
            cleaned_data.loc[invalid_hl, 'High'] = cleaned_data.loc[invalid_hl, 'Low']
    
    # Ensure Open and Close are within High-Low range
    if all(col in cleaned_data.columns for col in ['Open', 'High', 'Low', 'Close']):
        # Fix Open prices outside High-Low range
        open_high = cleaned_data['Open'] > cleaned_data['High']
        open_low = cleaned_data['Open'] < cleaned_data['Low']
        cleaned_data.loc[open_high, 'Open'] = cleaned_data.loc[open_high, 'High']
        cleaned_data.loc[open_low, 'Open'] = cleaned_data.loc[open_low, 'Low']
        
        # Fix Close prices outside High-Low range
        close_high = cleaned_data['Close'] > cleaned_data['High']
        close_low = cleaned_data['Close'] < cleaned_data['Low']
        cleaned_data.loc[close_high, 'Close'] = cleaned_data.loc[close_high, 'High']
        cleaned_data.loc[close_low, 'Close'] = cleaned_data.loc[close_low, 'Low']
    
    logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
    return cleaned_data


def calculate_technical_indicators(data: pd.DataFrame, periods: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        data: Stock data DataFrame with OHLCV columns.
        periods: Dictionary with indicator names and periods. If None, uses defaults.
        
    Returns:
        DataFrame with additional technical indicator columns.
    """
    if periods is None:
        periods = {
            'sma_short': 20,
            'sma_long': 50,
            'ema_short': 12,
            'ema_long': 26,
            'rsi': 14,
            'bb': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
    
    indicators_data = data.copy()
    
    if 'Close' not in indicators_data.columns:
        logger.error("Close price column not found")
        return indicators_data
    
    # Simple Moving Averages
    if 'sma_short' in periods:
        indicators_data[f'SMA_{periods["sma_short"]}'] = indicators_data['Close'].rolling(
            window=periods['sma_short']
        ).mean()
    
    if 'sma_long' in periods:
        indicators_data[f'SMA_{periods["sma_long"]}'] = indicators_data['Close'].rolling(
            window=periods['sma_long']
        ).mean()
    
    # Exponential Moving Averages
    if 'ema_short' in periods:
        indicators_data[f'EMA_{periods["ema_short"]}'] = indicators_data['Close'].ewm(
            span=periods['ema_short']
        ).mean()
    
    if 'ema_long' in periods:
        indicators_data[f'EMA_{periods["ema_long"]}'] = indicators_data['Close'].ewm(
            span=periods['ema_long']
        ).mean()
    
    # RSI (Relative Strength Index)
    if 'rsi' in periods:
        indicators_data['RSI'] = _calculate_rsi(indicators_data['Close'], periods['rsi'])
    
    # Bollinger Bands
    if 'bb' in periods and 'High' in indicators_data.columns and 'Low' in indicators_data.columns:
        bb_period = periods['bb']
        sma = indicators_data['Close'].rolling(window=bb_period).mean()
        std = indicators_data['Close'].rolling(window=bb_period).std()
        
        indicators_data['BB_Upper'] = sma + (2 * std)
        indicators_data['BB_Lower'] = sma - (2 * std)
        indicators_data['BB_Middle'] = sma
        indicators_data['BB_Width'] = indicators_data['BB_Upper'] - indicators_data['BB_Lower']
        indicators_data['BB_Position'] = (indicators_data['Close'] - indicators_data['BB_Lower']) / indicators_data['BB_Width']
    
    # MACD (Moving Average Convergence Divergence)
    if all(key in periods for key in ['macd_fast', 'macd_slow', 'macd_signal']):
        ema_fast = indicators_data['Close'].ewm(span=periods['macd_fast']).mean()
        ema_slow = indicators_data['Close'].ewm(span=periods['macd_slow']).mean()
        
        indicators_data['MACD'] = ema_fast - ema_slow
        indicators_data['MACD_Signal'] = indicators_data['MACD'].ewm(span=periods['macd_signal']).mean()
        indicators_data['MACD_Histogram'] = indicators_data['MACD'] - indicators_data['MACD_Signal']
    
    # Price-based indicators
    if all(col in indicators_data.columns for col in ['High', 'Low', 'Close']):
        # True Range
        indicators_data['TR'] = _calculate_true_range(
            indicators_data['High'], 
            indicators_data['Low'], 
            indicators_data['Close']
        )
        
        # Average True Range (ATR)
        indicators_data['ATR'] = indicators_data['TR'].rolling(window=14).mean()
    
    # Volume indicators
    if 'Volume' in indicators_data.columns:
        # Volume Moving Average
        indicators_data['Volume_MA'] = indicators_data['Volume'].rolling(window=20).mean()
        
        # Volume Rate of Change
        indicators_data['Volume_ROC'] = indicators_data['Volume'].pct_change(periods=10)
    
    # Price change indicators
    indicators_data['Daily_Return'] = indicators_data['Close'].pct_change()
    indicators_data['Log_Return'] = np.log(indicators_data['Close'] / indicators_data['Close'].shift(1))
    indicators_data['Price_Change'] = indicators_data['Close'] - indicators_data['Close'].shift(1)
    indicators_data['Price_Change_Pct'] = indicators_data['Price_Change'] / indicators_data['Close'].shift(1) * 100
    
    logger.info(f"Technical indicators calculated. New shape: {indicators_data.shape}")
    return indicators_data


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def create_features(data: pd.DataFrame, target_column: str = 'Close', lookback_days: int = 5) -> pd.DataFrame:
    """
    Create additional features for machine learning models.
    
    Args:
        data: Stock data DataFrame with technical indicators.
        target_column: Column to use as target variable.
        lookback_days: Number of days to look back for feature creation.
        
    Returns:
        DataFrame with additional features.
    """
    features_data = data.copy()
    
    if target_column not in features_data.columns:
        logger.error(f"Target column '{target_column}' not found")
        return features_data
    
    # Lag features
    for i in range(1, lookback_days + 1):
        features_data[f'{target_column}_lag_{i}'] = features_data[target_column].shift(i)
    
    # Rolling statistics
    for window in [5, 10, 20]:
        features_data[f'{target_column}_mean_{window}'] = features_data[target_column].rolling(window=window).mean()
        features_data[f'{target_column}_std_{window}'] = features_data[target_column].rolling(window=window).std()
        features_data[f'{target_column}_min_{window}'] = features_data[target_column].rolling(window=window).min()
        features_data[f'{target_column}_max_{window}'] = features_data[target_column].rolling(window=window).max()
    
    # Price position within rolling window
    for window in [5, 10, 20]:
        rolling_min = features_data[target_column].rolling(window=window).min()
        rolling_max = features_data[target_column].rolling(window=window).max()
        features_data[f'{target_column}_position_{window}'] = (
            (features_data[target_column] - rolling_min) / (rolling_max - rolling_min)
        )
    
    # Volatility features
    features_data['Volatility_5'] = features_data['Daily_Return'].rolling(window=5).std()
    features_data['Volatility_10'] = features_data['Daily_Return'].rolling(window=10).std()
    features_data['Volatility_20'] = features_data['Daily_Return'].rolling(window=20).std()
    
    # Momentum features
    for period in [5, 10, 20]:
        features_data[f'Momentum_{period}'] = features_data[target_column] / features_data[target_column].shift(period) - 1
    
    # Time-based features
    if 'Date' in features_data.columns:
        features_data['Year'] = pd.to_datetime(features_data['Date']).dt.year
        features_data['Month'] = pd.to_datetime(features_data['Date']).dt.month
        features_data['Day'] = pd.to_datetime(features_data['Date']).dt.day
        features_data['DayOfWeek'] = pd.to_datetime(features_data['Date']).dt.dayofweek
        features_data['DayOfYear'] = pd.to_datetime(features_data['Date']).dt.dayofyear
        features_data['Quarter'] = pd.to_datetime(features_data['Date']).dt.quarter
    
    # Target variable for next day (for supervised learning)
    features_data['Target_Next_Day'] = features_data[target_column].shift(-1)
    features_data['Target_Next_Day_Return'] = features_data['Daily_Return'].shift(-1)
    
    # Remove rows with NaN values created by feature engineering
    initial_shape = features_data.shape
    features_data = features_data.dropna().reset_index(drop=True)
    final_shape = features_data.shape
    
    logger.info(f"Feature creation completed. Shape: {initial_shape} -> {final_shape}")
    return features_data
