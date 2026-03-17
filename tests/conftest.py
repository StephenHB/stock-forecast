"""Shared pytest fixtures for stock-forecast tests."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_stock_df():
    """Minimal daily stock DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(100, 130, 30),
            "High": np.linspace(105, 135, 30),
            "Low": np.linspace(98, 128, 30),
            "Close": np.linspace(103, 133, 30),
            "Volume": np.full(30, 1e6),
        }
    )


@pytest.fixture
def sample_daily_stock_long():
    """Longer daily stock DataFrame for pipeline integration (1+ year)."""
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(400) * 0.5)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + np.abs(np.random.randn(400)),
            "Low": close - np.abs(np.random.randn(400)),
            "Close": close + np.random.randn(400) * 0.3,
            "Volume": np.abs(np.random.randn(400) * 1e6) + 1e6,
        }
    )


@pytest.fixture
def sample_stock_df_long():
    """Daily stock DataFrame with enough rows for create_features (60+)."""
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(100, 140, 80),
            "High": np.linspace(105, 145, 80),
            "Low": np.linspace(98, 138, 80),
            "Close": np.linspace(103, 143, 80),
            "Volume": np.full(80, 1e6),
        }
    )


@pytest.fixture
def sample_weekly_stock_df():
    """Minimal weekly stock DataFrame for testing."""
    dates = pd.date_range("2023-01-06", periods=20, freq="W-FRI")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(100, 120, 20),
            "High": np.linspace(105, 125, 20),
            "Low": np.linspace(98, 118, 20),
            "Close": np.linspace(103, 123, 20),
            "Volume": np.full(20, 5e6),
        }
    )


@pytest.fixture
def sample_features_df():
    """Feature matrix for LightGBM testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n).cumsum() + 100,
            "feature_2": np.random.randn(n).cumsum() + 50,
            "feature_3": np.random.rand(n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def sample_target_series(sample_features_df):
    """Target series for LightGBM testing (aligned with features)."""
    y = sample_features_df["feature_1"].shift(-4).dropna()
    return y
