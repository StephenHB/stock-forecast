"""Unit tests for src.forecasting.weekly_aggregator."""

import pytest
import pandas as pd

from src.forecasting.weekly_aggregator import WeeklyAggregator


class TestWeeklyAggregator:
    """Tests for WeeklyAggregator."""

    def test_aggregate_returns_dataframe(self, sample_stock_df):
        """aggregate should return a DataFrame."""
        agg = WeeklyAggregator()
        result = agg.aggregate(sample_stock_df, date_column="Date")
        assert isinstance(result, pd.DataFrame)

    def test_aggregate_reduces_rows(self, sample_stock_df):
        """Weekly aggregation should have fewer rows than daily."""
        agg = WeeklyAggregator()
        result = agg.aggregate(sample_stock_df, date_column="Date")
        assert len(result) <= len(sample_stock_df)

    def test_aggregate_has_expected_columns(self, sample_stock_df):
        """Aggregated data should have OHLCV columns."""
        agg = WeeklyAggregator()
        result = agg.aggregate(sample_stock_df, date_column="Date")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_aggregate_with_datetime_index(self, sample_stock_df):
        """Should work when Date is index."""
        df = sample_stock_df.set_index("Date")
        agg = WeeklyAggregator()
        result = agg.aggregate(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_aggregate_raises_without_date(self, sample_stock_df):
        """Should raise if no date column and index is not datetime."""
        df = sample_stock_df.drop(columns=["Date"])
        df.index = range(len(df))
        agg = WeeklyAggregator()
        with pytest.raises(ValueError, match="datetime"):
            agg.aggregate(df)
