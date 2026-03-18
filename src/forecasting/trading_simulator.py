"""
Trading Simulation Module

Simulates profit/loss from trading according to forecasted price direction.
Incorporates five research-driven enhancements over the naive baseline:

1. Implied-return signal: derive (pred - Close[T]) / Close[T] as the signal
   magnitude so threshold and sizing use the same scale regardless of stock price.
2. Dead-zone threshold: only trade when |implied_return| > threshold_pct.
   Periods inside the dead zone keep the current position (HOLD), which
   eliminates whipsaw trades on low-confidence signals.
3. Proportional position sizing: BUY allocates confidence% of available cash,
   where confidence scales from 0 → 1 as signal strength grows from the
   threshold to 3× the threshold. Full size only for high-conviction signals.
4. Transaction cost model: each execution is adjusted by cost_per_side
   (buy: exec × (1 + cost), sell: exec × (1 - cost)).
5. Regime filter: SELL signals are suppressed (→ HOLD) when the stock is
   trading above its 200-day MA, preventing premature exits during bull trends.
"""

import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field


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
    n_holds: int
    start_price: float
    end_price: float
    buy_hold_return_pct: float
    total_cost_paid: float


def _get_signal_closes(
    dates: List[pd.Timestamp],
    price_series: pd.Series,
) -> List[float]:
    """Get the close price ON each signal date (Close[T]).

    Used as the directional reference (signal compares pred vs Close[T])
    and as the execution price (order fills at Close[T] as a proxy for
    the next-day open).
    """
    signal_closes = []
    for d in dates:
        on_or_before = price_series[price_series.index <= d]
        if len(on_or_before) > 0:
            signal_closes.append(float(on_or_before.iloc[-1]))
        else:
            signal_closes.append(float(price_series.iloc[0]))
    return signal_closes


def _compute_regime_filter(price_series: pd.Series) -> pd.Series:
    """Return a boolean Series: True when price is above its 200-day MA.

    Uses min_periods=100 so partial history still produces a signal.
    Only suppresses SELL signals — BUY signals are never filtered out.

    Converts to plain float64 first to avoid pandas 2.x nullable-dtype NA
    propagation, then fills any remaining NaN with False so bool() never
    raises "boolean value of NA is ambiguous".
    """
    ps = price_series.astype(float)
    ma_200 = ps.rolling(200, min_periods=100).mean()
    return (ps > ma_200).fillna(False)


def run_simulation(
    symbol: str,
    predictions: List[float],
    actuals: List[float],
    dates: List[pd.Timestamp],
    price_series: pd.Series,
    initial_cash: float = 100_000.0,
    threshold_pct: float = 0.5,
    cost_per_side: float = 0.001,
) -> SimulationResult:
    """
    Run trading simulation for a single stock with all five enhancements.

    At each signal date T:
      - implied_return  = (pred − Close[T]) / Close[T] × 100  [Solution 1]
      - Dead zone:      |implied_return| ≤ threshold_pct → HOLD [Solution 2]
      - Regime filter:  SELL while price > 200-day MA   → HOLD [Solution 5]
      - Confidence:     min(|implied_return| / (3×threshold), 1.0) [Solution 3]
      - BUY:            convert confidence% of cash to shares at
                        exec_price × (1 + cost_per_side)         [Solution 4]
      - SELL:           liquidate all shares at
                        exec_price × (1 − cost_per_side)         [Solution 4]
      - Mark-to-market: actual Close[T+H] at end of hold period.

    Args:
        symbol: Stock symbol.
        predictions: Model-predicted close prices for each signal date.
        actuals: Actual close prices H periods after each signal date.
        dates: Signal dates (one per non-overlapping backtest window).
        price_series: Full Close price series with DatetimeIndex.
        initial_cash: Starting cash.
        threshold_pct: Dead-zone half-width in %. Signals weaker than this
                       are ignored; the current position is held unchanged.
                       Default 0.5 (%). Set to 0.0 to disable.
        cost_per_side: Fractional transaction cost applied per execution.
                       Default 0.001 (0.1 %). Set to 0.0 to disable.

    Returns:
        SimulationResult with equity curve, trades, and metrics.
    """
    n = len(dates)
    if n == 0 or len(predictions) != n or len(actuals) != n:
        raise ValueError("predictions, actuals, and dates must have same length")

    # Ensure price_series has a tz-naive index so comparisons with dates work
    # regardless of what yfinance / pandas version returned.
    if isinstance(price_series.index, pd.DatetimeIndex) and price_series.index.tz is not None:
        price_series = price_series.copy()
        price_series.index = price_series.index.tz_convert(None)

    # Normalise dates to tz-naive pd.Timestamp to match price_series index.
    dates = [
        pd.Timestamp(d).tz_localize(None) if pd.Timestamp(d).tzinfo is not None
        else pd.Timestamp(d)
        for d in dates
    ]

    signal_closes = _get_signal_closes(dates, price_series)
    uptrend = _compute_regime_filter(price_series)

    # Threshold at which confidence reaches 100% (3× dead-zone width)
    full_conviction_threshold = max(threshold_pct * 3.0, 0.3)

    cash = initial_cash
    shares = 0.0
    equity_curve: List[float] = []
    trades: List[Dict[str, Any]] = []
    n_holds = 0
    total_cost_paid = 0.0

    for i in range(n):
        ref_price = signal_closes[i]
        pred = predictions[i]
        actual = actuals[i]

        # --- Solution 1: implied return magnitude as signal ---
        if ref_price > 0:
            implied_return_pct = (pred - ref_price) / ref_price * 100.0
        else:
            implied_return_pct = 0.0

        # --- Solution 2: dead-zone threshold ---
        if implied_return_pct > threshold_pct:
            action = "BUY"
        elif implied_return_pct < -threshold_pct:
            action = "SELL"
        else:
            action = "HOLD"

        # --- Solution 5: regime filter (suppress SELL in uptrend) ---
        if action == "SELL":
            d = dates[i]
            on_or_before_regime = uptrend[uptrend.index <= d]
            if len(on_or_before_regime) > 0:
                _val = on_or_before_regime.iloc[-1]
                in_uptrend = bool(_val) if pd.notna(_val) else False
            else:
                in_uptrend = False
            if in_uptrend:
                action = "HOLD"

        if action == "HOLD":
            n_holds += 1

        # --- Solution 3: confidence-proportional position sizing ---
        if threshold_pct > 0:
            confidence = min(abs(implied_return_pct) / full_conviction_threshold, 1.0)
        else:
            confidence = 1.0

        exec_price = ref_price if ref_price > 0 else actual

        # --- Solution 4 + execute ---
        if action == "BUY" and cash > 0 and exec_price > 0:
            buy_value = cash * confidence
            exec_price_adj = exec_price * (1.0 + cost_per_side)
            new_shares = buy_value / exec_price_adj
            cost_paid = buy_value * cost_per_side
            trades.append({
                "date": dates[i],
                "action": "BUY",
                "price": exec_price_adj,
                "shares": new_shares,
                "value": buy_value,
                "cost": cost_paid,
                "confidence": round(confidence, 3),
            })
            shares += new_shares
            cash -= buy_value
            total_cost_paid += cost_paid

        elif action == "SELL" and shares > 0:
            exec_price_adj = exec_price * (1.0 - cost_per_side)
            sell_value = shares * exec_price_adj
            cost_paid = shares * exec_price * cost_per_side
            trades.append({
                "date": dates[i],
                "action": "SELL",
                "price": exec_price_adj,
                "shares": shares,
                "value": sell_value,
                "cost": cost_paid,
                "confidence": round(confidence, 3),
            })
            cash += sell_value
            shares = 0.0
            total_cost_paid += cost_paid

        # Mark-to-market at actual Close[T+H]
        period_value = cash + shares * actual
        equity_curve.append(period_value)

    final_actual = actuals[-1] if actuals else 0.0
    final_value = cash + shares * final_actual
    total_return_pct = (final_value - initial_cash) / initial_cash * 100.0

    start_price = signal_closes[0] if signal_closes else final_actual
    end_price = final_actual
    buy_hold_return_pct = (
        (end_price / start_price - 1.0) * 100.0 if start_price > 0 else 0.0
    )

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
        n_holds=n_holds,
        start_price=start_price,
        end_price=end_price,
        buy_hold_return_pct=buy_hold_return_pct,
        total_cost_paid=total_cost_paid,
    )


def run_multi_stock_simulation(
    backtest_results: Dict[str, Dict],
    price_series_by_symbol: Dict[str, pd.Series],
    initial_cash_per_stock: float = 100_000.0,
    threshold_pct: float = 0.5,
    cost_per_side: float = 0.001,
) -> Dict[str, "SimulationResult"]:
    """
    Run trading simulation for multiple stocks.

    Args:
        backtest_results: Dict from run_backtest (symbol → {predictions, actuals, dates, …}).
        price_series_by_symbol: Dict of symbol → Close price Series (datetime index).
        initial_cash_per_stock: Starting cash allocated per stock.
        threshold_pct: Dead-zone threshold in % (passed to run_simulation).
        cost_per_side: Per-side transaction cost fraction (passed to run_simulation).

    Returns:
        Dict of symbol → SimulationResult.
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
                threshold_pct=threshold_pct,
                cost_per_side=cost_per_side,
            )
            results[symbol] = res
        except (ValueError, KeyError, IndexError):
            results[symbol] = None
    return results
