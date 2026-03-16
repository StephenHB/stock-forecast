# Testing Agent — Examples

## Unit Test Example

```python
# test/forecasting/test_lgbm_forecaster.py
import pytest
import pandas as pd
from src.forecasting.lgbm_forecaster import LGBMForecaster

def test_forecaster_fit_predict_returns_expected_shape():
    """New feature: forecaster produces predictions with correct shape."""
    X = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    y = pd.Series([0.1, 0.2, 0.3])
    forecaster = LGBMForecaster()
    forecaster.fit(X, y)
    preds = forecaster.predict(X)
    assert len(preds) == len(X)
    assert preds.ndim == 1

def test_forecaster_handles_empty_input():
    """Edge case: empty input raises or returns expected result."""
    forecaster = LGBMForecaster()
    X_empty = pd.DataFrame()
    with pytest.raises(ValueError):
        forecaster.predict(X_empty)
```

## Integration Test Example

```python
# test/test_integration.py
import pytest
from src.data_preprocess import StockDataLoader
from src.forecasting import ForecastingPipeline

def test_data_to_forecast_pipeline_integration(config_stocks, tmp_path):
    """Existing workflow: load → preprocess → forecast runs end-to-end."""
    loader = StockDataLoader(data_dir=str(tmp_path))
    # Use fixture or minimal real data
    data = loader.download_stock_data(
        stock_symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-06-30"
    )
    pipeline = ForecastingPipeline()
    result = pipeline.run(data["AAPL"])
    assert "predictions" in result
    assert len(result["predictions"]) > 0
```

## Fixture Example

```python
# test/conftest.py
import pytest

@pytest.fixture
def sample_stock_df():
    """Minimal DataFrame for testing."""
    return pd.DataFrame({
        "Open": [100, 101, 102],
        "High": [105, 106, 107],
        "Low": [99, 100, 101],
        "Close": [104, 105, 106],
        "Volume": [1e6, 1.1e6, 1.2e6],
    }, index=pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03"]))
```
