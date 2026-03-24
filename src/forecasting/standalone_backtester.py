"""
Standalone Time Series Backtesting Module

This module provides a simplified backtesting framework that:
1. Uses pre-optimized hyperparameters (no tuning required)
2. Works independently of the forecasting pipeline
3. Focuses on performance evaluation with full sample data
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

try:
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise ImportError(f"Required packages not installed: {e}") from e

logger = logging.getLogger(__name__)


class StandaloneBacktester:
    """
    Standalone backtesting framework for time series forecasting.
    
    Features:
    - Uses pre-optimized hyperparameters (no tuning)
    - Works with full sample data
    - Independent of forecasting pipeline
    - Focuses on performance evaluation
    """
    
    def __init__(
        self,
        initial_train_size: int = 13,  # 1 year of quarterly data
        test_size: int = 1,  # 1 quarter ahead
        step_size: int = 1,  # Move forward 1 quarter each iteration
        min_train_size: int = 6,  # Minimum 1.5 years of training data
        target_column: str = 'Close',
        forecast_horizon: int = 1,  # 1 quarter ahead
        lgb_num_threads: Optional[int] = None,
    ):
        """
        Initialize the standalone backtester.
        
        Args:
            initial_train_size: Initial training window size (quarters)
            test_size: Test window size (quarters)
            step_size: Step size for moving window (quarters)
            min_train_size: Minimum training window size (quarters)
            target_column: Name of the target column
            forecast_horizon: Number of quarters to forecast ahead
            lgb_num_threads: If set, passed to LightGBM as ``num_threads`` (e.g. 1 when
                fitting many models in parallel to avoid CPU oversubscription).
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon
        
        # Pre-optimized hyperparameters (based on typical stock forecasting performance)
        self.best_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        if lgb_num_threads is not None:
            self.best_params = {**self.best_params, 'num_threads': lgb_num_threads}

        self.scaler = StandardScaler()
        self.results = []
        
    def backtest(
        self, 
        data: pd.DataFrame, 
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run backtesting with pre-optimized hyperparameters.
        
        Args:
            data: Time series data with datetime index
            feature_columns: List of feature columns to use
            
        Returns:
            Backtesting results dictionary
        """
        logger.info("Starting standalone backtesting...")
        
        # Validate input data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
        
        # Sort data by date
        data = data.sort_index()
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [col for col in data.columns 
                             if col != self.target_column and pd.api.types.is_numeric_dtype(data[col])]
        
        logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
        
        # Generate backtest windows
        windows = self._generate_backtest_windows(len(data))
        logger.info(f"Generated {len(windows)} backtest windows")
        
        # Run backtesting
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Backtest iteration {i+1}/{len(windows)}")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[self.target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[self.target_column]
            
            # Ensure targets are Series
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.iloc[:, 0]
            if isinstance(y_test, pd.DataFrame):
                y_test = y_test.iloc[:, 0]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with pre-optimized parameters
            model = lgb.LGBMRegressor(**self.best_params)
            model.fit(
                X_train_scaled, 
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # Make predictions
            predictions = model.predict(X_test_scaled)
            
            # Store results
            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)
            all_dates.extend(test_data.index.tolist())
            
            # Calculate metrics for this window
            mape = mean_absolute_percentage_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.results.append({
                'window': i + 1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'mape': mape,
                'mse': mse,
                'r2': r2,
                'predictions': predictions.tolist(),
                'actuals': y_test.values.tolist()
            })
        
        # Calculate overall metrics
        overall_mape = mean_absolute_percentage_error(all_actuals, all_predictions)
        overall_mse = mean_squared_error(all_actuals, all_predictions)
        overall_r2 = r2_score(all_actuals, all_predictions)
        
        # Compile results
        backtest_results = {
            'overall_metrics': {
                'mape': overall_mape,
                'mse': overall_mse,
                'r2': overall_r2,
                'rmse': np.sqrt(overall_mse)
            },
            'window_results': self.results,
            'predictions': all_predictions,
            'actuals': all_actuals,
            'dates': all_dates,
            'hyperparameters': self.best_params,
            'feature_columns': feature_columns,
            'total_windows': len(windows)
        }
        
        logger.info(f"Backtesting completed. Overall MAPE: {overall_mape:.4f}")
        return backtest_results
    
    def _generate_backtest_windows(self, data_length: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate backtest windows for time series validation.
        
        Args:
            data_length: Total length of the dataset
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        # Start with initial training size
        train_start = 0
        train_end = self.initial_train_size
        
        while train_end + self.test_size <= data_length:
            # Define test window
            test_start = train_end
            test_end = min(test_start + self.test_size, data_length)
            
            # Ensure minimum training size
            if train_end - train_start >= self.min_train_size:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move window forward
            train_start += self.step_size
            train_end = min(train_start + self.initial_train_size, data_length)
            
            # Stop if we can't create more windows
            if train_end >= data_length:
                break
        
        return windows
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get a summary of backtesting performance.
        
        Returns:
            DataFrame with performance metrics by window
        """
        if not self.results:
            raise ValueError("No backtesting results available. Run backtest() first.")
        
        summary_data = []
        for result in self.results:
            summary_data.append({
                'Window': result['window'],
                'Train_Period': f"{result['train_start'].strftime('%Y-%m-%d')} to {result['train_end'].strftime('%Y-%m-%d')}",
                'Test_Period': f"{result['test_start'].strftime('%Y-%m-%d')} to {result['test_end'].strftime('%Y-%m-%d')}",
                'MAPE': result['mape'],
                'MSE': result['mse'],
                'R2': result['r2']
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot backtesting results.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if not self.results:
                raise ValueError("No backtesting results available. Run backtest() first.")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtesting Results Summary', fontsize=16)
            
            # Extract data for plotting
            windows = [r['window'] for r in self.results]
            mapes = [r['mape'] for r in self.results]
            mses = [r['mse'] for r in self.results]
            r2s = [r['r2'] for r in self.results]
            
            # Plot MAPE over time
            axes[0, 0].plot(windows, mapes, marker='o', linewidth=2, markersize=6)
            axes[0, 0].set_title('MAPE by Window')
            axes[0, 0].set_xlabel('Window')
            axes[0, 0].set_ylabel('MAPE')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot MSE over time
            axes[0, 1].plot(windows, mses, marker='o', linewidth=2, markersize=6, color='orange')
            axes[0, 1].set_title('MSE by Window')
            axes[0, 1].set_xlabel('Window')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot R2 over time
            axes[1, 0].plot(windows, r2s, marker='o', linewidth=2, markersize=6, color='green')
            axes[1, 0].set_title('R² by Window')
            axes[1, 0].set_xlabel('Window')
            axes[1, 0].set_ylabel('R²')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot predictions vs actuals (last window)
            last_result = self.results[-1]
            axes[1, 1].scatter(last_result['actuals'], last_result['predictions'], alpha=0.6)
            axes[1, 1].plot([min(last_result['actuals']), max(last_result['actuals'])], 
                           [min(last_result['actuals']), max(last_result['actuals'])], 
                           'r--', linewidth=2)
            axes[1, 1].set_title('Predictions vs Actuals (Last Window)')
            axes[1, 1].set_xlabel('Actual')
            axes[1, 1].set_ylabel('Predicted')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def save_results(self, filepath: str):
        """
        Save backtesting results to a file.
        
        Args:
            filepath: Path to save the results
        """
        if not self.results:
            raise ValueError("No backtesting results available. Run backtest() first.")
        
        # Create results dictionary
        results_dict = {
            'backtest_config': {
                'initial_train_size': self.initial_train_size,
                'test_size': self.test_size,
                'step_size': self.step_size,
                'min_train_size': self.min_train_size,
                'target_column': self.target_column,
                'forecast_horizon': self.forecast_horizon
            },
            'hyperparameters': self.best_params,
            'results': self.results
        }
        
        # Save as JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
