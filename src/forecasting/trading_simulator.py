"""
Trading Simulation Module

Simulates profit/loss from trading according to forecasted price direction.
- Buy when predicted price is going up (pred > current)
- Sell when predicted price is going down (pred < current)
- Each period: either sell all shares or buy with all available cash.
"""

import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Result of a single-stock trading simulation."""

    symbol: str
    initial_cash: float
    final_value: float
    total_return_pct: float
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    n_buys: int
    n_sells: int


def _get_prior_closes(
    dates: List[pd.Timestamp],
    price_series: pd.Series,
) -> List[float]:
    """Get prior close price for each date (price at start of period)."""
    prior_closes = []
    for d in dates:
        earlier = price_series[price_series.index < d]
        if len(earlier) > 0:
            prior_closes.append(float(earlier.iloc[-1]))
        else:
            prior_closes.append(float(price_series.iloc[0]))
    return prior_closes


def run_simulation(
    symbol: str,
    predictions: List[float],
    actuals: List[float],
    dates: List[pd.Timestamp],
    price_series: pd.Series,
    initial_cash: float = 100_000.0,
) -> SimulationResult:
    """
    Run trading simulation for a single stock.

    At each period: if predicted price > prior close → buy with all cash.
    If predicted price < prior close → sell all shares.

    Args:
        symbol: Stock symbol
        predictions: Predicted close prices for each test date
        actuals: Actual close prices for each test date
        dates: Test dates (datetime index)
        price_series: Full price series (datetime index) for prior closes
        initial_cash: Starting cash

    Returns:
        SimulationResult with equity curve, trades, and metrics
    """
    n = len(dates)
    if n == 0 or len(predictions) != n or len(actuals) != n:
        raise ValueError("predictions, actuals, and dates must have same length")

    prior_closes = _get_prior_closes(dates, price_series)

    cash = initial_cash
    shares = 0.0
    equity_curve = []
    trades = []

    for i in range(n):
        ref_price = prior_closes[i]
        pred = predictions[i]
        actual = actuals[i]

        # Decision: buy if forecast up, sell if forecast down
        if pred > ref_price:
            action = "BUY"
        else:
            action = "SELL"

        # Execute at prior close (approximate open price)
        exec_price = ref_price
        if exec_price <= 0:
            exec_price = actual

        if action == "BUY":
            if cash > 0 and exec_price > 0:
                new_shares = cash / exec_price
                trades.append({
                    "date": dates[i],
                    "action": "BUY",
                    "price": exec_price,
                    "shares": new_shares,
                    "value": cash,
                })
                shares += new_shares
                cash = 0.0
        else:  # SELL
            if shares > 0:
                sell_value = shares * exec_price
                trades.append({
                    "date": dates[i],
                    "action": "SELL",
                    "price": exec_price,
                    "shares": shares,
                    "value": sell_value,
                })
                cash += sell_value
                shares = 0.0

        # Portfolio value at end of period (mark-to-market at actual close)
        period_value = cash + shares * actual
        equity_curve.append(period_value)

    # Final value
    final_actual = actuals[-1] if actuals else 0.0
    final_value = cash + shares * final_actual
    total_return_pct = (final_value - initial_cash) / initial_cash * 100

    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(dates))
    n_buys = sum(1 for t in trades if t["action"] == "BUY")
    n_sells = sum(1 for t in trades if t["action"] == "SELL")

    return SimulationResult(
        symbol=symbol,
        initial_cash=initial_cash,
        final_value=final_value,
        total_return_pct=total_return_pct,
        equity_curve=equity_series,
        trades=trades,
        n_buys=n_buys,
        n_sells=n_sells,
    )


def run_multi_stock_simulation(
    backtest_results: Dict[str, Dict],
    price_series_by_symbol: Dict[str, pd.Series],
    initial_cash_per_stock: float = 100_000.0,
) -> Dict[str, SimulationResult]:
    """
    Run trading simulation for multiple stocks.

    Each stock gets its own 100k simulation (user allocates 100k per stock).

    Args:
        backtest_results: Dict from run_backtest (symbol -> {predictions, actuals, dates, ...})
        price_series_by_symbol: Dict of symbol -> Close price Series (datetime index)
        initial_cash_per_stock: Starting cash per stock

    Returns:
        Dict of symbol -> SimulationResult
    """
    results = {}
    for symbol, bt in backtest_results.items():
        if "error" in bt or "predictions" not in bt:
            continue
        if symbol not in price_series_by_symbol:
            continue

        preds = bt["predictions"]
        actuals = bt["actuals"]
        dates = bt["dates"]
        prices = price_series_by_symbol[symbol]

        if isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
            prices = prices["Close"]
        if isinstance(dates[0], str):
            dates = [pd.Timestamp(d) for d in dates]

        try:
            res = run_simulation(
                symbol=symbol,
                predictions=preds,
                actuals=actuals,
                dates=dates,
                price_series=prices,
                initial_cash=initial_cash_per_stock,
            )
            results[symbol] = res
        except (ValueError, KeyError, IndexError):
            results[symbol] = None  # caller can check
    return results
