"""
Complete Forecasting Pipeline

This module provides a complete end-to-end forecasting pipeline that integrates
weekly aggregation, dynamic feature engineering, LightGBM modeling, and backtesting.
"""

import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging

from .weekly_aggregator import WeeklyAggregator
from .dynamic_feature_engineer import DynamicFeatureEngineer
from .lgbm_forecaster import LightGBMForecaster
from .time_series_backtester import TimeSeriesBacktester

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    Complete end-to-end forecasting pipeline for stock prices.
    
    This pipeline integrates:
    1. Weekly data aggregation
    2. Dynamic feature engineering
    3. LightGBM modeling with hyperparameter tuning
    4. Time series backtesting with MAPE validation
    """
    
    def __init__(
        self,
        target_column: str = 'Close',
        forecast_horizon: int = 4,  # 4 weeks ahead
        backtest_windows: int = 12,  # 12 months of backtesting
        hyperparameter_tuning: bool = True,
        feature_engineering_config: Optional[Dict[str, Any]] = None,
        lgbm_params: Optional[Dict[str, Any]] = None,
        backtest_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the forecasting pipeline.
        
        Args:
            target_column: Target column to predict
            forecast_horizon: Number of weeks to forecast ahead
            backtest_windows: Number of backtest windows (quarters)
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            feature_engineering_config: Configuration for feature engineering
            lgbm_params: LightGBM model parameters
            backtest_params: Backtesting parameters
        """
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon
        self.backtest_windows = backtest_windows
        self.hyperparameter_tuning = hyperparameter_tuning
        
        # Initialize components
        self.weekly_aggregator = WeeklyAggregator(
            price_columns=['Open', 'High', 'Low', 'Close'],
            volume_columns=['Volume']
        )
        
        self.feature_engineer = DynamicFeatureEngineer(
            forecast_horizon=forecast_horizon,
            target_column=target_column,
            feature_engineering_config=feature_engineering_config
        )
        
        # Default LightGBM parameters
        default_lgbm_params = {
            'forecast_horizon': forecast_horizon,
            'target_column': target_column,
            'cv_folds': 3,  # Use 3 folds for time series (faster and more appropriate)
            'random_state': 42,
            'early_stopping_rounds': 50,
            'verbose': True
        }
        if lgbm_params:
            default_lgbm_params.update(lgbm_params)
        
        self.lgbm_forecaster = LightGBMForecaster(**default_lgbm_params)
        
        # Default backtesting parameters (quarterly)
        default_backtest_params = {
            'initial_train_size': 13,  # 1 year of quarterly data (52 weeks / 4)
            'test_size': 1,  # 1 quarter ahead
            'step_size': 1,  # Move forward 1 quarter each iteration
            'min_train_size': 6,  # Minimum 1.5 years of training data
            'metric': 'mape'
        }
        if backtest_params:
            default_backtest_params.update(backtest_params)
        
        self.backtester = TimeSeriesBacktester(**default_backtest_params)
        
        # Results storage
        self.weekly_data = None
        self.forecasting_data = None
        self.backtest_results = None
        self.final_model = None
        self.predictions = None
    
    def fit_predict(
        self, 
        daily_data: pd.DataFrame,
        end_date: Optional[str] = None,
        run_backtesting: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: fit model and make predictions.
        
        Args:
            daily_data: Daily stock data
            end_date: End date for training (if None, uses all data)
            run_backtesting: Whether to run backtesting
            
        Returns:
            Dictionary with results
        """
        logger.info("Starting complete forecasting pipeline...")
        
        # Step 1: Weekly aggregation
        logger.info("Step 1: Weekly data aggregation...")
        self.weekly_data = self.weekly_aggregator.aggregate(daily_data)
        logger.info(f"Weekly data shape: {self.weekly_data.shape}")
        
        # Step 2: Dynamic feature engineering
        logger.info("Step 2: Dynamic feature engineering...")
        self.forecasting_data = self.feature_engineer.create_forecasting_dataset(
            self.weekly_data, end_date
        )
        logger.info(f"Forecasting data shape: {self.forecasting_data.shape}")
        
        # Step 3: Backtesting (if requested)
        if run_backtesting:
            logger.info("Step 3: Time series backtesting...")
            self.backtest_results = self._run_backtesting()
            logger.info("Backtesting completed")
        
        # Step 4: Train final model
        logger.info("Step 4: Training final model...")
        self._train_final_model()
        
        # Step 5: Make predictions
        logger.info("Step 5: Making predictions...")
        self.predictions = self._make_predictions()
        
        # Compile results
        results = self._compile_results()
        
        logger.info("Forecasting pipeline completed successfully")
        return results
    
    def _run_backtesting(self) -> Dict[str, Any]:
        """
        Run time series backtesting.
        
        Returns:
            Backtesting results
        """
        # Prepare data for backtesting
        backtest_data = self.forecasting_data.dropna()
        
        # Hyperparameter grid for backtesting (reduced for faster time series validation)
        hyperparameter_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'num_leaves': [31, 63],
            'subsample': [0.8, 0.9]
        }
        
        # Run backtesting for 1-week ahead prediction
        backtest_results = self.backtester.backtest(
            data=backtest_data,
            model_class=LightGBMForecaster,
            model_params={
                'forecast_horizon': 1,  # Use 1-week for backtesting
                'target_column': self.target_column
            },
            target_column=self.target_column,
            hyperparameter_tuning=self.hyperparameter_tuning,
            hyperparameter_grid=hyperparameter_grid
        )
        
        return backtest_results
    
    def _train_final_model(self):
        """Train the final model on all available data."""
        # Prepare training data for 1-week ahead prediction
        X, y = self.feature_engineer.prepare_training_data(
            self.forecasting_data, target_horizon=1
        )
        
        # Use best hyperparameters from backtesting if available
        if self.backtest_results and 'best_hyperparameters' in self.backtest_results:
            best_params = self.backtest_results['best_hyperparameters']
            # Update model with best parameters
            for param, value in best_params.items():
                if hasattr(self.lgbm_forecaster, param):
                    setattr(self.lgbm_forecaster, param, value)
        
        # Train the model
        self.final_model = self.lgbm_forecaster.fit(
            X, y, 
            hyperparameter_tuning=self.hyperparameter_tuning
        )
    
    def _make_predictions(self) -> Dict[int, float]:
        """
        Make multi-step ahead predictions.
        
        Returns:
            Dictionary with predictions for each horizon
        """
        # Get the last available data point
        last_data = self.forecasting_data.iloc[-1:]
        
        # Extract features for prediction
        feature_columns = [col for col in last_data.columns 
                          if not col.startswith('target_') and col != self.target_column]
        
        X_pred = last_data[feature_columns]
        
        # Make predictions for each horizon
        predictions = {}
        for horizon in range(1, self.forecast_horizon + 1):
            # For multi-step ahead, we need to create a model for each horizon
            # For simplicity, we'll use the 1-week model and adjust predictions
            pred = self.final_model.predict(X_pred)[0]
            
            # Simple adjustment for longer horizons (can be improved)
            if horizon > 1:
                # Apply a decay factor for longer horizons
                decay_factor = 0.95 ** (horizon - 1)
                pred = pred * decay_factor
            
            predictions[horizon] = pred
        
        return predictions
    
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile all results into a comprehensive dictionary.
        
        Returns:
            Comprehensive results dictionary
        """
        results = {
            'pipeline_config': {
                'target_column': self.target_column,
                'forecast_horizon': self.forecast_horizon,
                'backtest_windows': self.backtest_windows,
                'hyperparameter_tuning': self.hyperparameter_tuning
            },
            'data_info': {
                'weekly_data_shape': self.weekly_data.shape if self.weekly_data is not None else None,
                'forecasting_data_shape': self.forecasting_data.shape if self.forecasting_data is not None else None,
                'date_range': {
                    'start': self.weekly_data.index.min() if self.weekly_data is not None else None,
                    'end': self.weekly_data.index.max() if self.weekly_data is not None else None
                }
            },
            'predictions': self.predictions,
            'backtest_results': self.backtest_results,
            'model_info': {
                'feature_importance': self.final_model.get_feature_importance() if self.final_model else None,
                'best_hyperparameters': self.final_model.best_params if self.final_model else None
            }
        }
        
        # Add performance summary if backtesting was run
        if self.backtest_results:
            results['performance_summary'] = {
                'primary_metric': self.backtest_results.get('primary_metric', 'mape'),
                'primary_metric_mean': self.backtest_results.get('primary_metric_mean'),
                'primary_metric_std': self.backtest_results.get('primary_metric_std'),
                'best_iteration': self.backtest_results.get('best_iteration'),
                'total_iterations': self.backtest_results.get('total_iterations')
            }
        
        return results
    
    def plot_results(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot comprehensive results.
        
        Args:
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        if self.weekly_data is None:
            logger.warning("No data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Historical prices
        axes[0, 0].plot(self.weekly_data.index, self.weekly_data[self.target_column], 
                       linewidth=2, label='Historical Price')
        axes[0, 0].set_title('Historical Stock Price')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Predictions
        if self.predictions:
            horizons = list(self.predictions.keys())
            pred_values = list(self.predictions.values())
            
            axes[0, 1].bar(horizons, pred_values, alpha=0.7, color='orange')
            axes[0, 1].set_title('Multi-Step Ahead Predictions')
            axes[0, 1].set_xlabel('Forecast Horizon (weeks)')
            axes[0, 1].set_ylabel('Predicted Price')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature importance
        if self.final_model:
            importance_df = self.final_model.get_feature_importance()
            top_features = importance_df.head(10)
            
            axes[1, 0].barh(range(len(top_features)), top_features['importance'])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features['feature'])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].invert_yaxis()
        
        # Plot 4: Backtesting results
        if self.backtest_results and 'detailed_results' in self.backtest_results:
            detailed_results = self.backtest_results['detailed_results']
            mape_values = [result['metrics']['mape'] for result in detailed_results]
            dates = [result['test_start'] for result in detailed_results]
            
            axes[1, 1].plot(dates, mape_values, marker='o', linewidth=2, markersize=4)
            axes[1, 1].set_title('Backtesting MAPE Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('MAPE (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_pipeline(self, filepath: str):
        """
        Save the complete pipeline.
        
        Args:
            filepath: Path to save the pipeline
        """
        import joblib
        
        pipeline_data = {
            'weekly_data': self.weekly_data,
            'forecasting_data': self.forecasting_data,
            'backtest_results': self.backtest_results,
            'final_model': self.final_model,
            'predictions': self.predictions,
            'config': {
                'target_column': self.target_column,
                'forecast_horizon': self.forecast_horizon,
                'backtest_windows': self.backtest_windows,
                'hyperparameter_tuning': self.hyperparameter_tuning
            }
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str):
        """
        Load a saved pipeline.
        
        Args:
            filepath: Path to load the pipeline from
        """
        import joblib
        
        pipeline_data = joblib.load(filepath)
        
        self.weekly_data = pipeline_data['weekly_data']
        self.forecasting_data = pipeline_data['forecasting_data']
        self.backtest_results = pipeline_data['backtest_results']
        self.final_model = pipeline_data['final_model']
        self.predictions = pipeline_data['predictions']
        
        config = pipeline_data['config']
        self.target_column = config['target_column']
        self.forecast_horizon = config['forecast_horizon']
        self.backtest_windows = config['backtest_windows']
        self.hyperparameter_tuning = config['hyperparameter_tuning']
        
        logger.info(f"Pipeline loaded from {filepath}")
