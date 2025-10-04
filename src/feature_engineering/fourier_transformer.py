"""
Fourier Transform Feature Engineering

This module implements Fourier transform-based feature engineering for time series data,
inspired by the tabpfn-time-series implementation. It extracts frequency domain features
that can capture seasonality and cyclical patterns in stock price data.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import logging

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class FourierTransformer(BaseFeatureTransformer):
    """
    Fourier Transform Feature Engineering for Time Series Data.
    
    This transformer extracts frequency domain features from time series data using
    Fast Fourier Transform (FFT). It can capture seasonality, cyclical patterns,
    and dominant frequencies in the data.
    
    Features extracted:
    - Real and imaginary parts of FFT coefficients
    - Magnitude and phase of dominant frequencies
    - Power spectral density features
    - Dominant frequency components
    
    Parameters:
    -----------
    n_components : int, default=10
        Number of Fourier components to extract
        
    columns : list of str, optional
        Columns to apply Fourier transform to. If None, applies to all numeric columns
        
    include_phase : bool, default=True
        Whether to include phase information in features
        
    include_magnitude : bool, default=True
        Whether to include magnitude information in features
        
    include_power : bool, default=True
        Whether to include power spectral density features
        
    dominant_freqs : int, default=5
        Number of dominant frequencies to extract
        
    feature_prefix : str, default="fourier"
        Prefix for generated feature names
    """
    
    def __init__(
        self,
        n_components: int = 10,
        columns: Optional[List[str]] = None,
        include_phase: bool = True,
        include_magnitude: bool = True,
        include_power: bool = True,
        dominant_freqs: int = 5,
        feature_prefix: str = "fourier"
    ):
        super().__init__(feature_prefix)
        self.n_components = n_components
        self.columns = columns
        self.include_phase = include_phase
        self.include_magnitude = include_magnitude
        self.include_power = include_power
        self.dominant_freqs = dominant_freqs
        
        # Validate parameters
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if dominant_freqs <= 0:
            raise ValueError("dominant_freqs must be positive")
        if dominant_freqs > n_components:
            raise ValueError("dominant_freqs cannot be greater than n_components")
    
    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Validate that specified columns exist in the input data."""
        if self.columns is not None:
            missing_cols = set(self.columns) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in data: {missing_cols}")
    
    def _get_target_columns(self, X: pd.DataFrame) -> List[str]:
        """Get the columns to apply Fourier transform to."""
        if self.columns is not None:
            return self.columns
        else:
            # Use all numeric columns
            return X.select_dtypes(include=[np.number]).columns.tolist()
    
    def _compute_fft_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Compute FFT-based features for a single time series.
        
        Args:
            series: Input time series data
            
        Returns:
            DataFrame with FFT features
        """
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < self.n_components:
            logger.warning(f"Series length ({len(clean_series)}) is less than n_components ({self.n_components})")
            n_components = len(clean_series)
        else:
            n_components = self.n_components
        
        # Compute FFT
        fft_values = fft(clean_series.values)
        freqs = fftfreq(len(clean_series))
        
        # Extract features
        features = {}
        
        # Real and imaginary parts of first n_components
        for i in range(n_components):
            features[f'real_{i}'] = fft_values[i].real
            features[f'imag_{i}'] = fft_values[i].imag
        
        # Magnitude and phase of dominant frequencies
        if self.include_magnitude or self.include_phase:
            # Get dominant frequencies (excluding DC component)
            magnitudes = np.abs(fft_values[1:len(fft_values)//2])
            dominant_indices = np.argsort(magnitudes)[-self.dominant_freqs:][::-1]
            
            for i, idx in enumerate(dominant_indices):
                freq_idx = idx + 1  # +1 because we excluded DC component
                if self.include_magnitude:
                    features[f'magnitude_{i}'] = magnitudes[idx]
                if self.include_phase:
                    features[f'phase_{i}'] = np.angle(fft_values[freq_idx])
                features[f'freq_{i}'] = freqs[freq_idx]
        
        # Power spectral density features
        if self.include_power:
            freqs_psd, psd = periodogram(clean_series.values)
            
            # Total power
            features['total_power'] = np.sum(psd)
            
            # Peak power and frequency
            peak_idx = np.argmax(psd)
            features['peak_power'] = psd[peak_idx]
            features['peak_frequency'] = freqs_psd[peak_idx]
            
            # Power in different frequency bands
            n_freqs = len(psd)
            low_freq_power = np.sum(psd[:n_freqs//4])
            mid_freq_power = np.sum(psd[n_freqs//4:3*n_freqs//4])
            high_freq_power = np.sum(psd[3*n_freqs//4:])
            
            features['low_freq_power'] = low_freq_power
            features['mid_freq_power'] = mid_freq_power
            features['high_freq_power'] = high_freq_power
            
            # Power ratios
            total_power = features['total_power']
            if total_power > 0:
                features['low_freq_ratio'] = low_freq_power / total_power
                features['mid_freq_ratio'] = mid_freq_power / total_power
                features['high_freq_ratio'] = high_freq_power / total_power
        
        return pd.DataFrame([features])
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using Fourier transform.
        
        Args:
            X: Input DataFrame with time series data
            
        Returns:
            DataFrame with Fourier transform features
        """
        target_columns = self._get_target_columns(X)
        all_features = []
        
        for col in target_columns:
            logger.info("Computing Fourier features for column: %s", col)
            
            # Compute FFT features for this column
            col_features = self._compute_fft_features(X[col])
            
            # Add column prefix to feature names
            col_features.columns = [self._create_feature_name(f"{col}_{name}") 
                                  for name in col_features.columns]
            
            all_features.append(col_features)
        
        # Combine all features
        if all_features:
            result = pd.concat(all_features, axis=1)
        else:
            result = pd.DataFrame(index=X.index)
        
        # Store feature names
        self.feature_names_ = list(result.columns)
        
        return result
    
    def get_dominant_frequencies(self, X: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Get the dominant frequencies for a specific column.
        
        Args:
            X: Input DataFrame
            column: Column name to analyze
            
        Returns:
            DataFrame with dominant frequency information
        """
        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        series = X[column].dropna()
        fft_values = fft(series.values)
        freqs = fftfreq(len(series))
        
        # Get dominant frequencies (excluding DC component)
        magnitudes = np.abs(fft_values[1:len(fft_values)//2])
        dominant_indices = np.argsort(magnitudes)[-self.dominant_freqs:][::-1]
        
        results = []
        for i, idx in enumerate(dominant_indices):
            freq_idx = idx + 1
            results.append({
                'rank': i + 1,
                'frequency': freqs[freq_idx],
                'magnitude': magnitudes[idx],
                'phase': np.angle(fft_values[freq_idx])
            })
        
        return pd.DataFrame(results)
    
    def plot_frequency_spectrum(self, X: pd.DataFrame, column: str, ax=None):
        """
        Plot the frequency spectrum for a specific column.
        
        Args:
            X: Input DataFrame
            column: Column name to plot
            ax: Matplotlib axis (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting") from exc
        
        if column not in X.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        series = X[column].dropna()
        fft_values = fft(series.values)
        freqs = fftfreq(len(series))
        
        # Plot only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_values[:len(fft_values)//2])
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(positive_freqs, positive_fft)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Frequency Spectrum - {column}')
        ax.grid(True)
        
        return ax
