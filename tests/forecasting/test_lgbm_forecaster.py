"""Unit tests for src.forecasting.lgbm_forecaster."""

import pytest
import pandas as pd
import numpy as np

from src.forecasting.lgbm_forecaster import LightGBMForecaster


@pytest.mark.slow
class TestLightGBMForecaster:
    """Tests for LightGBMForecaster."""

    def test_fit_predict_returns_expected_shape(
        self, sample_features_df, sample_target_series
    ):
        """Forecaster produces predictions with correct shape."""
        X = sample_features_df.loc[sample_target_series.index]
        y = sample_target_series
        forecaster = LightGBMForecaster(forecast_horizon=4, verbose=False)
        forecaster.fit(X, y, hyperparameter_tuning=False)
        preds = forecaster.predict(X)
        assert len(preds) == len(X)
        assert preds.ndim == 1
        assert isinstance(preds, np.ndarray)

    def test_predict_before_fit_raises(self, sample_features_df):
        """Predict before fit should raise."""
        forecaster = LightGBMForecaster(verbose=False)
        with pytest.raises(ValueError, match="must be fitted"):
            forecaster.predict(sample_features_df)

    def test_fit_stores_feature_names(
        self, sample_features_df, sample_target_series
    ):
        """Fit should store feature names."""
        X = sample_features_df.loc[sample_target_series.index]
        y = sample_target_series
        forecaster = LightGBMForecaster(verbose=False)
        forecaster.fit(X, y, hyperparameter_tuning=False)
        assert forecaster.feature_names == list(X.columns)

    def test_predictions_are_finite(
        self, sample_features_df, sample_target_series
    ):
        """Predictions should be finite numbers."""
        X = sample_features_df.loc[sample_target_series.index]
        y = sample_target_series
        forecaster = LightGBMForecaster(verbose=False)
        forecaster.fit(X, y, hyperparameter_tuning=False)
        preds = forecaster.predict(X)
        assert np.isfinite(preds).all()
