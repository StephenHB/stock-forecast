"""
LightGBM-based Stock Forecasting Model

This module implements a LightGBM model for stock price forecasting with
hyperparameter tuning and multi-step ahead prediction capabilities.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

try:
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}") from e

logger = logging.getLogger(__name__)


class LightGBMForecaster:
    """
    LightGBM-based forecasting model for stock prices.
    
    Features:
    - Multi-step ahead forecasting
    - Hyperparameter tuning with time series cross-validation
    - Feature importance analysis
    - Multiple evaluation metrics
    """
    
    def __init__(
        self,
        forecast_horizon: int = 4,
        target_column: str = 'Close',
        hyperparameter_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Initialize the LightGBM forecaster.
        
        Args:
            forecast_horizon: Number of weeks to forecast ahead
            target_column: Name of the target column
            hyperparameter_grid: Grid for hyperparameter tuning
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            early_stopping_rounds: Early stopping rounds
            verbose: Whether to print training progress
        """
        self.forecast_horizon = forecast_horizon
        self.target_column = target_column
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        # Default hyperparameter grid
        self.hyperparameter_grid = hyperparameter_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Models for each forecast horizon
        self.models = {}
        self.best_params = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        hyperparameter_tuning: bool = True
    ) -> 'LightGBMForecaster':
        """
        Fit the LightGBM model with optional hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target variable
            validation_data: Optional validation data
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Fitted model
        """
        logger.info("Starting LightGBM model training...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        if hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            best_params = self._tune_hyperparameters(X_scaled, y)
        else:
            # Use default parameters
            best_params = {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'verbose': -1
            }
        
        self.best_params = best_params
        
        # Train final model
        logger.info("Training final model...")
        self.model = lgb.LGBMRegressor(**best_params)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = pd.DataFrame(X_val_scaled, columns=self.feature_names, index=X_val.index)
            
            self.model.fit(
                X_scaled, y,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.model.fit(X_scaled, y)
        
        logger.info("Model training completed")
        return self
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Best hyperparameters
        """
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Create base model
        base_model = lgb.LGBMRegressor(
            random_state=self.random_state,
            verbose=-1
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.hyperparameter_grid,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            verbose=1 if self.verbose else 0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_multi_step(
        self, 
        X: pd.DataFrame, 
        forecasting_data: pd.DataFrame
    ) -> Dict[int, np.ndarray]:
        """
        Make multi-step ahead predictions.
        
        Args:
            X: Feature matrix for prediction
            forecasting_data: Full forecasting dataset
            
        Returns:
            Dictionary with predictions for each horizon
        """
        logger.info(f"Making multi-step ahead predictions for {self.forecast_horizon} weeks...")
        
        predictions = {}
        
        for horizon in range(1, self.forecast_horizon + 1):
            # Get the last available data point for this horizon
            horizon_data = forecasting_data.dropna(subset=[f'target_{horizon}w'])
            
            if len(horizon_data) == 0:
                logger.warning(f"No data available for {horizon}-week horizon")
                continue
            
            # Use the last row for prediction
            last_row = horizon_data.iloc[-1:][self.feature_names]
            
            # Make prediction
            pred = self.predict(last_row)
            predictions[horizon] = pred[0]
            
            logger.info(f"{horizon}-week prediction: {pred[0]:.2f}")
        
        return predictions
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'mape': mean_absolute_percentage_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': np.mean(np.abs(y - predictions)),
            'r2': r2_score(y, predictions)
        }
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'permutation')
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model must be fitted before getting feature importance")
        
        if importance_type == 'gain':
            importance = self.model.feature_importances_
        elif importance_type == 'split':
            importance = self.model.booster_.feature_importance(importance_type='split')
        else:
            raise ValueError("importance_type must be 'gain' or 'split'")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model must be fitted before saving")
        
        import joblib
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'forecast_horizon': self.forecast_horizon,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        import joblib
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        self.forecast_horizon = model_data['forecast_horizon']
        self.target_column = model_data['target_column']
        
        logger.info(f"Model loaded from {filepath}")
