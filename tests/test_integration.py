"""Integration tests for end-to-end workflows."""

import pytest
import pandas as pd

from src.data_preprocess import (
    validate_stock_data,
    clean_stock_data,
    calculate_technical_indicators,
    create_features,
)
from src.forecasting import WeeklyAggregator, ForecastingPipeline


class TestDataPreprocessIntegration:
    """Integration: validate -> clean -> indicators -> features."""

    def test_full_preprocess_workflow(self, sample_stock_df_long):
        """validate -> clean -> indicators -> create_features runs end-to-end."""
        validation = validate_stock_data(sample_stock_df_long)
        assert validation["has_data"] is True

        cleaned = clean_stock_data(sample_stock_df_long)
        assert len(cleaned) > 0

        with_indicators = calculate_technical_indicators(cleaned)
        assert "SMA_20" in with_indicators.columns

        features = create_features(with_indicators, lookback_days=3)
        assert len(features) > 0
        assert "Close_lag_1" in features.columns


class TestForecastingIntegration:
    """Integration: daily data -> weekly aggregation -> pipeline components."""

    def test_aggregate_then_feature_engineer(self, sample_daily_stock_long):
        """Weekly aggregation produces valid input for downstream."""
        agg = WeeklyAggregator()
        weekly = agg.aggregate(sample_daily_stock_long, date_column="Date")
        assert len(weekly) > 0
        assert "Close" in weekly.columns

    @pytest.mark.slow
    def test_pipeline_fit_predict_returns_results(self, sample_daily_stock_long):
        """ForecastingPipeline fit_predict returns dict with predictions."""
        pipeline = ForecastingPipeline(
            forecast_horizon=2,
            backtest_windows=2,
            hyperparameter_tuning=False,
        )
        results = pipeline.fit_predict(
            sample_daily_stock_long,
            run_backtesting=False,
        )
        assert "predictions" in results
        assert results["predictions"] is not None
        assert "data_info" in results
        assert "weekly_data_shape" in results["data_info"]
