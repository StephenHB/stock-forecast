"""
Dynamic Feature Engineering for Multi-Step Ahead Forecasting

This module creates features for multi-step ahead forecasting by generating
rows that match with future time periods. For missing future data, it uses
the latest available data.
"""

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import logging

from src.feature_engineering import (
    FourierTransformer,
    LagFeatures,
    RollingFeatures,
    TechnicalIndicators,
    TimeFeatures,
    DifferenceFeatures
)

logger = logging.getLogger(__name__)


class DynamicFeatureEngineer:
    """
    Dynamic feature engineering for multi-step ahead forecasting.
    
    Creates features for future time periods by:
    1. Generating future date rows
    2. Applying feature engineering to historical data
    3. Using latest available data for missing future values
    4. Creating target variables for multi-step ahead prediction
    """
    
    def __init__(
        self,
        forecast_horizon: int = 4,  # 4 weeks ahead
        target_column: str = 'Close',
        feature_engineering_config: Optional[Dict[str, Any]] = None,
        fill_method: str = 'latest'  # 'latest', 'forward', 'interpolate'
    ):
        """
        Initialize the dynamic feature engineer.
        
        Args:
            forecast_horizon: Number of weeks to forecast ahead
            target_column: Name of the target column to predict
            feature_engineering_config: Configuration for feature engineering
            fill_method: Method to fill missing future data
        """
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.fill_method = fill_method
        
        # Default feature engineering configuration
        self.fe_config = feature_engineering_config or {
            'fourier': {'n_components': 5, 'columns': [target_column]},
            'lags': {'lags': [1, 2, 4, 8], 'columns': [target_column]},
            'rolling': {'windows': [4, 8, 12], 'columns': [target_column], 'statistics': ['mean', 'std']},
            'technical': {'indicators': ['sma', 'ema', 'rsi'], 'price_column': target_column},
            'time': {'features': ['month', 'dayofweek'], 'cyclical_encoding': True},
            'difference': {'differences': [1, 4], 'columns': [target_column], 'include_pct_change': True}
        }
        
        # Initialize feature transformers
        self._initialize_transformers()
    
    def _initialize_transformers(self):
        """Initialize feature engineering transformers."""
        self.transformers = {}
        
        if 'fourier' in self.fe_config:
            self.transformers['fourier'] = FourierTransformer(**self.fe_config['fourier'])
        
        if 'lags' in self.fe_config:
            self.transformers['lags'] = LagFeatures(**self.fe_config['lags'])
        
        if 'rolling' in self.fe_config:
            self.transformers['rolling'] = RollingFeatures(**self.fe_config['rolling'])
        
        if 'technical' in self.fe_config:
            self.transformers['technical'] = TechnicalIndicators(**self.fe_config['technical'])
        
        if 'time' in self.fe_config:
            self.transformers['time'] = TimeFeatures(**self.fe_config['time'])
        
        if 'difference' in self.fe_config:
            self.transformers['difference'] = DifferenceFeatures(**self.fe_config['difference'])
    
    def create_forecasting_dataset(
        self, 
        data: pd.DataFrame, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create dataset for multi-step ahead forecasting.
        
        Args:
            data: Historical weekly data
            end_date: End date for forecasting (if None, uses last available date)
            
        Returns:
            DataFrame with features and targets for forecasting
        """
        logger.info(f"Creating forecasting dataset with {self.forecast_horizon} weeks ahead...")
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'Date' in data.columns:
                data = data.copy()
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            else:
                raise ValueError("Data must have datetime index or Date column")
        
        # Determine end date
        if end_date is None:
            end_date = data.index.max()
        else:
            end_date = pd.to_datetime(end_date)
        
        # Create extended dataset with future dates
        extended_data = self._create_extended_dataset(data, end_date)
        
        # Apply feature engineering
        features_data = self._apply_feature_engineering(extended_data)
        
        # Create target variables for multi-step ahead forecasting
        forecasting_data = self._create_target_variables(features_data)
        
        logger.info(f"Forecasting dataset created: {len(forecasting_data)} rows, {len(forecasting_data.columns)} features")
        return forecasting_data
    
    def _create_extended_dataset(self, data: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Create extended dataset with future dates.
        
        Args:
            data: Historical data
            end_date: End date for historical data
            
        Returns:
            Extended dataset with future dates
        """
        # Get the last available date
        last_date = data.index.max()
        
        # Create future dates (weekly frequency)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=self.forecast_horizon,
            freq='W-FRI'
        )
        
        # Create future rows with NaN values
        future_data = pd.DataFrame(index=future_dates, columns=data.columns)
        
        # Combine historical and future data
        extended_data = pd.concat([data, future_data])
        
        # Remove duplicate columns if any
        extended_data = extended_data.loc[:, ~extended_data.columns.duplicated()]
        
        # Fill missing future values based on fill method
        extended_data = self._fill_future_values(extended_data, end_date)
        
        return extended_data
    
    def _fill_future_values(self, data: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Fill missing future values using specified method.
        
        Args:
            data: Extended dataset with future NaN values
            end_date: End date for historical data
            
        Returns:
            Dataset with filled future values
        """
        filled_data = data.copy()
        
        # Get historical data (up to end_date)
        historical_mask = filled_data.index <= end_date
        future_mask = filled_data.index > end_date
        
        for column in filled_data.columns:
            if self.fill_method == 'latest':
                # Use the last available value
                last_value = filled_data.loc[historical_mask, column].iloc[-1]
                filled_data.loc[future_mask, column] = last_value
            
            elif self.fill_method == 'forward':
                # Forward fill from last historical value
                filled_data[column] = filled_data[column].fillna(method='ffill')
            
            elif self.fill_method == 'interpolate':
                # Linear interpolation
                filled_data[column] = filled_data[column].interpolate(method='linear')
            
            else:
                # Default to latest
                last_value = filled_data.loc[historical_mask, column].iloc[-1]
                filled_data.loc[future_mask, column] = last_value
        
        return filled_data
    
    def _apply_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the extended dataset.
        
        Args:
            data: Extended dataset with historical and future data
            
        Returns:
            Dataset with engineered features
        """
        logger.info("Applying feature engineering...")
        
        # Start with original data
        features_data = data.copy()
        
        # Apply each transformer
        for name, transformer in self.transformers.items():
            logger.info(f"Applying {name} transformer...")
            try:
                # Fit on historical data only (non-NaN values)
                historical_data = features_data.dropna()
                if len(historical_data) > 0:
                    transformer.fit(historical_data)
                    # Transform entire dataset
                    new_features = transformer.transform(features_data)
                    # Add new features to dataset
                    features_data = pd.concat([features_data, new_features], axis=1)
            except Exception as e:
                logger.warning(f"Error applying {name} transformer: {e}")
                continue
        
        return features_data
    
    def _create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for multi-step ahead forecasting.
        
        Args:
            data: Dataset with features
            
        Returns:
            Dataset with target variables
        """
        logger.info("Creating target variables for multi-step ahead forecasting...")
        
        target_data = data.copy()
        
        # Create target variables for each forecast horizon
        for horizon in range(1, self.forecast_horizon + 1):
            target_col_name = f'target_{horizon}w'
            # Shift target column backward by horizon weeks
            # Ensure we're working with a Series, not DataFrame
            target_series = target_data[self.target_column]
            if isinstance(target_series, pd.DataFrame):
                target_series = target_series.iloc[:, 0]  # Take first column if DataFrame
            target_data[target_col_name] = target_series.shift(-horizon)
        
        # Create relative change targets (percentage change)
        for horizon in range(1, self.forecast_horizon + 1):
            target_col_name = f'target_{horizon}w_pct'
            # Ensure we're working with Series, not DataFrame
            current_price = target_data[self.target_column]
            if isinstance(current_price, pd.DataFrame):
                current_price = current_price.iloc[:, 0]
            
            future_price = target_data[f'target_{horizon}w']
            if isinstance(future_price, pd.DataFrame):
                future_price = future_price.iloc[:, 0]
            
            target_data[target_col_name] = (future_price - current_price) / current_price * 100
        
        # Create binary targets (up/down)
        for horizon in range(1, self.forecast_horizon + 1):
            target_col_name = f'target_{horizon}w_direction'
            # Ensure we're working with Series, not DataFrame
            current_price = target_data[self.target_column]
            if isinstance(current_price, pd.DataFrame):
                current_price = current_price.iloc[:, 0]
            
            future_price = target_data[f'target_{horizon}w']
            if isinstance(future_price, pd.DataFrame):
                future_price = future_price.iloc[:, 0]
            
            target_data[target_col_name] = (future_price > current_price).astype(int)
        
        return target_data
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained LightGBM model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importance_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importance_
            }).sort_values('importance', ascending=False)
        else:
            logger.warning("Model does not have feature_importance_ attribute")
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': 0})
        
        return importance_df
    
    def prepare_training_data(
        self, 
        forecasting_data: pd.DataFrame, 
        target_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data for a specific forecast horizon.
        
        Args:
            forecasting_data: Dataset with features and targets
            target_horizon: Which forecast horizon to use (1-4 weeks)
            
        Returns:
            Tuple of (features, target)
        """
        # Remove rows with NaN targets (future data)
        training_data = forecasting_data.dropna(subset=[f'target_{target_horizon}w'])
        
        # Separate features and target
        feature_columns = [col for col in training_data.columns 
                          if not col.startswith('target_') and col != self.target_column]
        
        X = training_data[feature_columns]
        y = training_data[f'target_{target_horizon}w']
        
        logger.info(f"Training data prepared: {len(X)} samples, {len(feature_columns)} features")
        return X, y
