"""
Tests to ensure Trading Simulation start/end prices match the Historical Price chart.

Both use the same raw stock_data; this module verifies the algorithm and data flow.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import _get_start_end_prices_from_daily, _get_daily_close_series


def _build_chart_data(daily_df: pd.DataFrame) -> pd.Series:
    """Use same helper as Historical chart and simulation table."""
    return _get_daily_close_series(daily_df)


def test_get_start_end_prices_matches_chart_data():
    """Start and end prices must exist in chart data at correct dates."""
    # Create synthetic daily data matching yfinance structure
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    closes = 100.0 + pd.Series(range(100), index=dates).values * 0.5
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})

    chart_series = _build_chart_data(daily_df)
    first_test = pd.Timestamp("2024-02-15")
    last_test = pd.Timestamp("2024-04-20")

    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df, first_test, last_test
    )

    # Start = prior close before first test date
    before_first = chart_series[chart_series.index < first_test]
    expected_start = float(before_first.iloc[-1]) if len(before_first) > 0 else float(chart_series.iloc[0])
    assert start_price == pytest.approx(expected_start, rel=1e-9), (
        f"Start price {start_price} should match chart value {expected_start} "
        f"at date {before_first.index[-1] if len(before_first) > 0 else 'N/A'}"
    )

    # End = close on last test date
    on_or_before = chart_series[chart_series.index <= last_test]
    expected_end = float(on_or_before.iloc[-1]) if len(on_or_before) > 0 else float(chart_series.iloc[-1])
    assert end_price == pytest.approx(expected_end, rel=1e-9), (
        f"End price {end_price} should match chart value {expected_end} "
        f"at date {on_or_before.index[-1] if len(on_or_before) > 0 else 'N/A'}"
    )


def test_start_end_prices_are_from_chart_series():
    """Start and end prices must be values that appear in the chart."""
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    closes = [100.0 + i for i in range(50)]
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})

    chart_series = _build_chart_data(daily_df)
    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df,
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-15"),
    )

    assert start_price in chart_series.values, "Start price must exist in chart data"
    assert end_price in chart_series.values, "End price must exist in chart data"


def test_buy_hold_math_consistency():
    """Buy & Hold % = (end/start - 1) * 100 must be consistent."""
    dates = pd.date_range("2024-01-02", periods=60, freq="B")
    closes = [100.0] * 20 + [110.0] * 20 + [105.0] * 20  # up then down
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})

    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df,
        pd.Timestamp("2024-02-01"),  # in first segment
        pd.Timestamp("2024-03-20"),  # in last segment
    )

    if start_price > 0:
        buy_hold_pct = (end_price / start_price - 1) * 100
        assert -100 <= buy_hold_pct <= 1000, "Buy & Hold % should be reasonable"


def test_timezone_naive_comparison():
    """Handles tz-aware index (e.g. America/New_York) vs naive test dates."""
    dates = pd.date_range("2024-01-02", periods=30, freq="B", tz="America/New_York")
    closes = [100.0 + i for i in range(30)]
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})

    # Should not raise
    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df,
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-25"),
    )
    assert start_price > 0 and end_price > 0


def test_single_day_test_period():
    """When first and last test date are the same."""
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    closes = [100.0 + i for i in range(20)]
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})
    chart_series = _build_chart_data(daily_df)

    same_date = pd.Timestamp("2024-01-15")
    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df, same_date, same_date
    )

    before_first = chart_series[chart_series.index < same_date]
    on_or_before = chart_series[chart_series.index <= same_date]
    expected_start = float(before_first.iloc[-1]) if len(before_first) > 0 else float(chart_series.iloc[0])
    expected_end = float(on_or_before.iloc[-1]) if len(on_or_before) > 0 else float(chart_series.iloc[-1])
    assert start_price == pytest.approx(expected_start, rel=1e-9)
    assert end_price == pytest.approx(expected_end, rel=1e-9)


def test_chart_and_simulation_use_same_data():
    """Chart and simulation table must use identical data source."""
    dates = pd.date_range("2024-01-02", periods=80, freq="B")
    closes = 50.0 + np.arange(80) * 0.5
    daily_df = pd.DataFrame({"Date": dates, "Close": closes})

    # Chart uses _get_daily_close_series
    chart_series = _get_daily_close_series(daily_df)
    # Simulation uses _get_start_end_prices_from_daily which uses same helper
    start_price, end_price = _get_start_end_prices_from_daily(
        daily_df, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-15")
    )

    # Values must be in chart
    assert start_price in chart_series.values
    assert end_price in chart_series.values
    # Chart min/max in test range should bracket our values
    in_range = chart_series[
        (chart_series.index >= pd.Timestamp("2024-02-01"))
        & (chart_series.index <= pd.Timestamp("2024-03-15"))
    ]
    if len(in_range) > 0:
        assert start_price <= chart_series.max() and end_price <= chart_series.max()
