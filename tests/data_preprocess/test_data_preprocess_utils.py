"""Unit tests for src.data_preprocess.data_preprocess_utils."""

import pytest
import pandas as pd
import numpy as np

from src.data_preprocess.data_preprocess_utils import (
    validate_stock_data,
    clean_stock_data,
    calculate_technical_indicators,
    create_features,
)


class TestValidateStockData:
    """Tests for validate_stock_data."""

    def test_valid_data_passes_validation(self, sample_stock_df):
        """Valid stock data should pass all validation checks."""
        result = validate_stock_data(sample_stock_df)
        assert result["has_required_columns"]
        assert result["has_data"]
        assert result["has_date_column"]
        assert result["no_duplicate_dates"]
        assert result["no_missing_prices"]
        assert result["positive_prices"]
        assert result["valid_high_low"]
        assert result["valid_volume"]

    def test_empty_data_fails_validation(self):
        """Empty DataFrame should fail has_data."""
        empty_df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        result = validate_stock_data(empty_df)
        assert not result["has_data"]

    def test_missing_columns_fails_validation(self, sample_stock_df):
        """Data missing required columns should fail has_required_columns."""
        incomplete_df = sample_stock_df[["Date", "Close"]]
        result = validate_stock_data(incomplete_df)
        assert result["has_required_columns"] is False

    def test_duplicate_dates_fails_validation(self, sample_stock_df):
        """Duplicate dates should fail no_duplicate_dates."""
        dup_df = pd.concat([sample_stock_df, sample_stock_df.iloc[:1]], ignore_index=True)
        result = validate_stock_data(dup_df)
        assert result["no_duplicate_dates"] is False

    def test_invalid_high_low_fails_validation(self, sample_stock_df):
        """High < Low should fail valid_high_low."""
        invalid_df = sample_stock_df.copy()
        invalid_df.loc[0, "High"] = 90
        invalid_df.loc[0, "Low"] = 100
        result = validate_stock_data(invalid_df)
        assert not result["valid_high_low"]


class TestCleanStockData:
    """Tests for clean_stock_data."""

    def test_returns_dataframe(self, sample_stock_df):
        """clean_stock_data should return a DataFrame."""
        result = clean_stock_data(sample_stock_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_required_columns(self, sample_stock_df):
        """Cleaned data should retain required columns."""
        result = clean_stock_data(sample_stock_df)
        for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_date_converted_to_datetime(self, sample_stock_df):
        """Date column should be converted to datetime."""
        result = clean_stock_data(sample_stock_df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])


class TestCalculateTechnicalIndicators:
    """Tests for calculate_technical_indicators."""

    def test_adds_technical_indicators(self, sample_stock_df):
        """Should add technical indicator columns."""
        result = calculate_technical_indicators(sample_stock_df)
        expected_indicators = ["SMA_20", "EMA_12", "RSI"]
        for ind in expected_indicators:
            assert ind in result.columns

    def test_returns_dataframe(self, sample_stock_df):
        """Should return DataFrame."""
        result = calculate_technical_indicators(sample_stock_df)
        assert isinstance(result, pd.DataFrame)


class TestCreateFeatures:
    """Tests for create_features."""

    def test_adds_lag_features(self, sample_stock_df_long):
        """Should add lag features when lookback_days > 0."""
        data_with_indicators = calculate_technical_indicators(sample_stock_df_long)
        result = create_features(data_with_indicators, lookback_days=3)
        assert isinstance(result, pd.DataFrame)
        assert "Close_lag_1" in result.columns
        assert len(result) > 0
