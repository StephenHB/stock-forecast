# Testing Agent — Examples

## Unit Test Example

```python
# tests/forecasting/test_lgbm_forecaster.py
import pytest
from src.forecasting.lgbm_forecaster import LightGBMForecaster

def test_forecaster_fit_predict_returns_expected_shape(sample_features_df, sample_target_series):
    """New feature: forecaster produces predictions with correct shape."""
    X = sample_features_df.loc[sample_target_series.index]
    y = sample_target_series
    forecaster = LightGBMForecaster(forecast_horizon=4, verbose=False)
    forecaster.fit(X, y, hyperparameter_tuning=False)
    preds = forecaster.predict(X)
    assert len(preds) == len(X)
    assert preds.ndim == 1

def test_forecaster_requires_fit_before_predict(sample_features_df):
    """Edge case: predict before fit raises ValueError."""
    forecaster = LightGBMForecaster(verbose=False)
    with pytest.raises(ValueError, match="must be fitted"):
        forecaster.predict(sample_features_df)
```

## Integration Test Example

```python
# tests/test_integration.py
import pytest
from src.forecasting import ForecastingPipeline

def test_pipeline_fit_predict_returns_results(sample_daily_stock_long):
    """Existing workflow: daily data → ForecastingPipeline fit_predict runs end-to-end."""
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
```

## Fixture Example

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_df():
    """Minimal daily stock DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": np.linspace(100, 130, 30),
        "High": np.linspace(105, 135, 30),
        "Low": np.linspace(98, 128, 30),
        "Close": np.linspace(103, 133, 30),
        "Volume": np.full(30, 1e6),
    })
```
