"""
Time Series Backtesting Framework

This module provides a comprehensive backtesting framework for time series
forecasting models with MAPE validation and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TimeSeriesBacktester:
    """
    Time series backtesting framework for forecasting models.
    
    Features:
    - Rolling window backtesting
    - Expanding window backtesting
    - Hyperparameter optimization with time series CV
    - MAPE and other accuracy metrics
    - Performance tracking and visualization
    """
    
    def __init__(
        self,
        initial_train_size: int = 52,  # 1 year of weekly data
        test_size: int = 4,  # 4 weeks ahead
        step_size: int = 4,  # Move forward 4 weeks each iteration
        min_train_size: int = 26,  # Minimum 6 months of training data
        metric: str = 'mape',  # Primary metric for optimization
        additional_metrics: Optional[List[str]] = None
    ):
        """
        Initialize the time series backtester.
        
        Args:
            initial_train_size: Initial training window size (weeks)
            test_size: Test window size (weeks)
            step_size: Step size for moving window (weeks)
            min_train_size: Minimum training window size (weeks)
            metric: Primary metric for optimization
            additional_metrics: Additional metrics to track
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.metric = metric
        self.additional_metrics = additional_metrics or ['rmse', 'mae', 'r2']
        
        # Results storage
        self.backtest_results = []
        self.hyperparameter_results = []
        self.best_hyperparameters = {}
        
    def backtest(
        self,
        data: pd.DataFrame,
        model_class: Any,
        model_params: Dict[str, Any],
        target_column: str = 'Close',
        feature_columns: Optional[List[str]] = None,
        hyperparameter_tuning: bool = True,
        hyperparameter_grid: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """
        Perform time series backtesting.
        
        Args:
            data: Time series data
            model_class: Model class to use
            model_params: Model parameters
            target_column: Target column name
            feature_columns: Feature columns (if None, uses all except target)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            hyperparameter_grid: Grid for hyperparameter tuning
            
        Returns:
            Backtesting results
        """
        logger.info("Starting time series backtesting...")
        
        # Prepare data
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        # Ensure data is sorted by date
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index")
        
        data = data.sort_index()
        
        # Generate backtest windows
        windows = self._generate_backtest_windows(len(data))
        
        logger.info(f"Generated {len(windows)} backtest windows")
        
        # Perform backtesting
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Backtest iteration {i+1}/{len(windows)}")
            
            # Get training and test data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Hyperparameter tuning if requested
            if hyperparameter_tuning and hyperparameter_grid:
                best_params = self._tune_hyperparameters(
                    X_train, y_train, model_class, model_params, hyperparameter_grid
                )
                model_params.update(best_params)
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, predictions)
            
            # Store results
            result = {
                'iteration': i + 1,
                'train_start': data.index[train_start],
                'train_end': data.index[train_end - 1],
                'test_start': data.index[test_start],
                'test_end': data.index[test_end - 1],
                'train_size': len(train_data),
                'test_size': len(test_data),
                'predictions': predictions,
                'actual': y_test.values,
                'metrics': metrics
            }
            
            self.backtest_results.append(result)
        
        # Aggregate results
        aggregated_results = self._aggregate_results()
        
        logger.info("Backtesting completed")
        return aggregated_results
    
    def _generate_backtest_windows(self, data_length: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate backtest windows.
        
        Args:
            data_length: Total length of data
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        # Start with initial training size
        train_end = self.initial_train_size
        
        while train_end + self.test_size <= data_length:
            train_start = max(0, train_end - self.initial_train_size)
            test_start = train_end
            test_end = min(data_length, test_start + self.test_size)
            
            # Ensure minimum training size
            if train_end - train_start >= self.min_train_size:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move window forward
            train_end += self.step_size
        
        return windows
    
    def _tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_class: Any,
        base_params: Dict[str, Any],
        hyperparameter_grid: Dict[str, List]
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using time series cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_class: Model class
            base_params: Base model parameters
            hyperparameter_grid: Grid for hyperparameter tuning
            
        Returns:
            Best hyperparameters
        """
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Use fewer splits for backtesting
        
        # Create base model
        base_model = model_class(**base_params)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=hyperparameter_grid,
            cv=tscv,
            scoring=f'neg_{self.metric}',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['mape'] = mape
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics['rmse'] = rmse
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        metrics['mae'] = mae
        
        # R² (Coefficient of Determination)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r2'] = r2
        
        # Directional Accuracy (for price prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
            metrics['directional_accuracy'] = directional_accuracy
        
        return metrics
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate backtesting results.
        
        Returns:
            Aggregated results
        """
        if not self.backtest_results:
            return {}
        
        # Extract metrics
        all_metrics = [result['metrics'] for result in self.backtest_results]
        
        # Calculate mean and std for each metric
        aggregated_metrics = {}
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics]
            aggregated_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Calculate overall performance
        primary_metric_values = [result['metrics'][self.metric] for result in self.backtest_results]
        best_iteration = np.argmin(primary_metric_values)  # Lower is better for MAPE
        
        aggregated_results = {
            'total_iterations': len(self.backtest_results),
            'aggregated_metrics': aggregated_metrics,
            'best_iteration': best_iteration + 1,
            'best_metrics': self.backtest_results[best_iteration]['metrics'],
            'primary_metric': self.metric,
            'primary_metric_mean': np.mean(primary_metric_values),
            'primary_metric_std': np.std(primary_metric_values),
            'detailed_results': self.backtest_results
        }
        
        return aggregated_results
    
    def plot_backtest_results(self, metric: str = 'mape', figsize: Tuple[int, int] = (12, 8)):
        """
        Plot backtesting results.
        
        Args:
            metric: Metric to plot
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if not self.backtest_results:
            logger.warning("No backtest results to plot")
            return
        
        # Extract metric values
        metric_values = [result['metrics'][metric] for result in self.backtest_results]
        dates = [result['test_start'] for result in self.backtest_results]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot metric over time
        ax1.plot(dates, metric_values, marker='o', linewidth=2, markersize=4)
        ax1.set_title(f'{metric.upper()} Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel(metric.upper())
        ax1.grid(True, alpha=0.3)
        
        # Plot histogram
        ax2.hist(metric_values, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title(f'{metric.upper()} Distribution')
        ax2.set_xlabel(metric.upper())
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(metric_values)
        std_val = np.std(metric_values)
        ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax2.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1 Std: {mean_val + std_val:.2f}')
        ax2.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1 Std: {mean_val - std_val:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary as DataFrame.
        
        Returns:
            DataFrame with performance summary
        """
        if not self.backtest_results:
            return pd.DataFrame()
        
        # Extract all metrics
        summary_data = []
        for result in self.backtest_results:
            row = {
                'iteration': result['iteration'],
                'test_start': result['test_start'],
                'test_end': result['test_end'],
                'train_size': result['train_size'],
                'test_size': result['test_size']
            }
            row.update(result['metrics'])
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, filepath: str):
        """
        Save backtesting results.
        
        Args:
            filepath: Path to save results
        """
        import joblib
        
        results_data = {
            'backtest_results': self.backtest_results,
            'hyperparameter_results': self.hyperparameter_results,
            'best_hyperparameters': self.best_hyperparameters,
            'config': {
                'initial_train_size': self.initial_train_size,
                'test_size': self.test_size,
                'step_size': self.step_size,
                'min_train_size': self.min_train_size,
                'metric': self.metric,
                'additional_metrics': self.additional_metrics
            }
        }
        
        joblib.dump(results_data, filepath)
        logger.info(f"Backtesting results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load backtesting results.
        
        Args:
            filepath: Path to load results from
        """
        import joblib
        
        results_data = joblib.load(filepath)
        
        self.backtest_results = results_data['backtest_results']
        self.hyperparameter_results = results_data['hyperparameter_results']
        self.best_hyperparameters = results_data['best_hyperparameters']
        
        config = results_data['config']
        self.initial_train_size = config['initial_train_size']
        self.test_size = config['test_size']
        self.step_size = config['step_size']
        self.min_train_size = config['min_train_size']
        self.metric = config['metric']
        self.additional_metrics = config['additional_metrics']
        
        logger.info(f"Backtesting results loaded from {filepath}")
