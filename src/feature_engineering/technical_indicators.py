"""
Technical Indicators for Stock Price Analysis

This module implements various technical analysis indicators commonly used
in stock price analysis and trading.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class TechnicalIndicators(BaseFeatureTransformer):
    """
    Technical Indicators Transformer for Stock Price Data.
    
    Implements various technical analysis indicators including moving averages,
    momentum indicators, volatility indicators, and trend indicators.
    
    Parameters:
    -----------
    indicators : list of str, default=['sma', 'ema', 'rsi', 'macd', 'bollinger']
        List of technical indicators to compute
        
    price_column : str, default='Close'
        Column name for price data
        
    volume_column : str, optional
        Column name for volume data (required for some indicators)
        
    windows : dict, optional
        Dictionary mapping indicator names to their window parameters
        
    feature_prefix : str, default="technical"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        price_column: str = 'Close',
        volume_column: Optional[str] = None,
        windows: Optional[Dict[str, Any]] = None,
        feature_prefix: str = "technical"
    ):
        super().__init__(feature_prefix)
        self.indicators = indicators or ['sma', 'ema', 'rsi', 'macd', 'bollinger']
        self.price_column = price_column
        self.volume_column = volume_column
        self.windows = windows or {}
        
        # Default window parameters
        self.default_windows = {
            'sma': [5, 10, 20, 50],
            'ema': [5, 10, 20, 50],
            'rsi': 14,
            'macd': (12, 26, 9),
            'bollinger': (20, 2),
            'stochastic': (14, 3, 3),
            'williams_r': 14,
            'cci': 20,
            'atr': 14,
            'obv': None
        }
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that required columns exist in the input data."""
        if self.price_column not in X.columns:
            raise ValueError(f"Price column '{self.price_column}' not found in data")
        
        volume_indicators = ['obv', 'ad']
        if any(ind in self.indicators for ind in volume_indicators):
            if self.volume_column is None or self.volume_column not in X.columns:
                raise ValueError(f"Volume column required for indicators: {volume_indicators}")
    
    def _sma(self, series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=window).mean()
    
    def _ema(self, series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=window).mean()
    
    def _rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._ema(series, fast)
        ema_slow = self._ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _bollinger_bands(self, series: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = self._sma(series, window)
        std = series.rolling(window=window).std()
        
        return {
            'bb_upper': sma + (std * std_dev),
            'bb_middle': sma,
            'bb_lower': sma - (std * std_dev),
            'bb_width': (sma + (std * std_dev)) - (sma - (std * std_dev)),
            'bb_position': (series - (sma - (std * std_dev))) / ((sma + (std * std_dev)) - (sma - (std * std_dev)))
        }
    
    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3, smooth: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        k_percent_smooth = k_percent.rolling(window=smooth).mean()
        
        return {
            'stoch_k': k_percent_smooth,
            'stoch_d': d_percent
        }
    
    def _williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    def _cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        return true_range.rolling(window=window).mean()
    
    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        return pd.Series(obv, index=close.index).cumsum()
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by computing technical indicators.
        
        Args:
            X: Input DataFrame with stock price data
            
        Returns:
            DataFrame with technical indicator features
        """
        result = X.copy()
        
        # Get window parameters
        windows = {**self.default_windows, **self.windows}
        
        for indicator in self.indicators:
            logger.info("Computing technical indicator: %s", indicator)
            
            if indicator == 'sma':
                window_list = windows.get('sma', self.default_windows['sma'])
                for window in window_list:
                    feature_name = self._create_feature_name(f"sma_{window}")
                    result[feature_name] = self._sma(X[self.price_column], window)
            
            elif indicator == 'ema':
                window_list = windows.get('ema', self.default_windows['ema'])
                for window in window_list:
                    feature_name = self._create_feature_name(f"ema_{window}")
                    result[feature_name] = self._ema(X[self.price_column], window)
            
            elif indicator == 'rsi':
                window = windows.get('rsi', self.default_windows['rsi'])
                feature_name = self._create_feature_name("rsi")
                result[feature_name] = self._rsi(X[self.price_column], window)
            
            elif indicator == 'macd':
                fast, slow, signal = windows.get('macd', self.default_windows['macd'])
                macd_features = self._macd(X[self.price_column], fast, slow, signal)
                for name, values in macd_features.items():
                    feature_name = self._create_feature_name(name)
                    result[feature_name] = values
            
            elif indicator == 'bollinger':
                window, std_dev = windows.get('bollinger', self.default_windows['bollinger'])
                bb_features = self._bollinger_bands(X[self.price_column], window, std_dev)
                for name, values in bb_features.items():
                    feature_name = self._create_feature_name(name)
                    result[feature_name] = values
            
            elif indicator == 'stochastic':
                if 'High' in X.columns and 'Low' in X.columns:
                    k_window, d_window, smooth = windows.get('stochastic', self.default_windows['stochastic'])
                    stoch_features = self._stochastic(X['High'], X['Low'], X[self.price_column], 
                                                    k_window, d_window, smooth)
                    for name, values in stoch_features.items():
                        feature_name = self._create_feature_name(name)
                        result[feature_name] = values
            
            elif indicator == 'williams_r':
                if 'High' in X.columns and 'Low' in X.columns:
                    window = windows.get('williams_r', self.default_windows['williams_r'])
                    feature_name = self._create_feature_name("williams_r")
                    result[feature_name] = self._williams_r(X['High'], X['Low'], X[self.price_column], window)
            
            elif indicator == 'cci':
                if 'High' in X.columns and 'Low' in X.columns:
                    window = windows.get('cci', self.default_windows['cci'])
                    feature_name = self._create_feature_name("cci")
                    result[feature_name] = self._cci(X['High'], X['Low'], X[self.price_column], window)
            
            elif indicator == 'atr':
                if 'High' in X.columns and 'Low' in X.columns:
                    window = windows.get('atr', self.default_windows['atr'])
                    feature_name = self._create_feature_name("atr")
                    result[feature_name] = self._atr(X['High'], X['Low'], X[self.price_column], window)
            
            elif indicator == 'obv':
                if self.volume_column and self.volume_column in X.columns:
                    feature_name = self._create_feature_name("obv")
                    result[feature_name] = self._obv(X[self.price_column], X[self.volume_column])
        
        # Store feature names
        new_columns = [col for col in result.columns if col not in X.columns]
        self.feature_names_ = new_columns
        
        return result
