"""
Market-Wide Reference Features

Downloads and merges broad market signals (SPY, QQQ, VIX, 10Y yield) into
a stock's feature DataFrame so the model can learn how macro-level conditions
affect individual stock movements.

Usage pattern
─────────────
    # Once per app run (cached)
    market_data = download_market_reference_data(start_date, end_date)

    # Inside the feature-engineering loop for each symbol
    features = add_market_features(features, market_data, symbol=symbol)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reference tickers — downloaded once, reused for every stock's feature set.
_REFERENCE_SYMBOLS: dict[str, str] = {
    "spy": "SPY",    # S&P 500 — broad market direction
    "qqq": "QQQ",    # Nasdaq-100 — tech/growth regime
    "vix": "^VIX",   # CBOE Volatility Index — fear / risk-off gauge
    "tny": "^TNX",   # 10-Year Treasury yield — interest-rate environment
}

# Symbols whose own price is an alias for a reference ticker; skip that feature
# to avoid leaking the target into its own feature set.
_SPY_ALIASES = {"SPY", "VOO", "IVV", "OEF"}
_QQQ_ALIASES = {"QQQ", "ONEQ"}


def download_market_reference_data(start_date: str, end_date: str) -> dict[str, pd.Series]:
    """
    Download daily Close prices for SPY, QQQ, ^VIX, and ^TNX.

    Returns a dict mapping lowercase key → tz-naive Close Series indexed by date.
    Returns an empty dict if yfinance is unavailable or all downloads fail.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; market reference features disabled.")
        return {}

    result: dict[str, pd.Series] = {}
    for key, sym in _REFERENCE_SYMBOLS.items():
        try:
            hist = yf.Ticker(sym).history(
                start=start_date, end=end_date, interval="1d", auto_adjust=True
            )
            if hist.empty:
                logger.warning("No data returned for reference symbol %s", sym)
                continue
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else "Datetime"
            hist[date_col] = pd.to_datetime(hist[date_col])
            if hist[date_col].dt.tz is not None:
                hist[date_col] = hist[date_col].dt.tz_convert(None)
            hist = hist.set_index(date_col)
            result[key] = hist["Close"].astype(float).rename(key)
            logger.info("Downloaded market reference %s (%d rows)", sym, len(result[key]))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to download reference symbol %s: %s", sym, exc)

    return result


def add_market_features(
    data: pd.DataFrame,
    market_data: Optional[dict[str, pd.Series]],
    symbol: str = "",
) -> pd.DataFrame:
    """
    Merge broad-market features into a stock's feature DataFrame.

    All reference series are forward-filled to cover weekends / holidays so
    that every row in the stock DataFrame gets a value.  Look-ahead bias is
    avoided because we use the reference close that is contemporaneous with
    (or prior to) the stock's own date.

    Features added when reference data is present:

    SPY (skipped when forecasting SPY / VOO / IVV / OEF):
      spy_return_1d    — S&P 500 1-day return (market momentum)
      spy_return_5d    — S&P 500 5-day return (weekly trend)
      spy_ma20_ratio   — SPY vs its 20-day MA (bull/bear regime, centred at 0)

    QQQ (skipped when forecasting QQQ):
      qqq_return_1d    — Nasdaq-100 1-day return (tech/growth sentiment)

    VIX:
      vix_level        — raw VIX close (fear gauge)
      vix_change_1d    — 1-day change in VIX
      vix_ma20_ratio   — VIX vs its 20-day MA (spike detector, centred at 0)
      vix_high         — 1 if VIX > 25 (elevated fear regime flag)

    10Y Treasury yield (^TNX):
      yield_10y        — yield level (interest-rate environment)
      yield_10y_chg_1d — 1-day change in yield (rate pressure)
    """
    if not market_data:
        return data

    f = data.copy()
    stock_idx = pd.DatetimeIndex(f.index).floor("D")
    sym_upper = symbol.upper()

    def _align(series: pd.Series) -> pd.Series:
        """Reindex to the stock's DatetimeIndex with forward-fill."""
        s = series.copy()
        s.index = pd.DatetimeIndex(s.index).floor("D")
        return s.reindex(stock_idx, method="ffill")

    # ── SPY ───────────────────────────────────────────────────────────────────
    if "spy" in market_data and sym_upper not in _SPY_ALIASES:
        spy = _align(market_data["spy"])
        f["spy_return_1d"] = spy.pct_change(1)
        f["spy_return_5d"] = spy.pct_change(5)
        spy_ma20 = spy.rolling(20, min_periods=5).mean()
        f["spy_ma20_ratio"] = (spy / spy_ma20.replace(0, np.nan)) - 1.0

    # ── QQQ ───────────────────────────────────────────────────────────────────
    if "qqq" in market_data and sym_upper not in _QQQ_ALIASES:
        qqq = _align(market_data["qqq"])
        f["qqq_return_1d"] = qqq.pct_change(1)

    # ── VIX ───────────────────────────────────────────────────────────────────
    if "vix" in market_data:
        vix = _align(market_data["vix"])
        f["vix_level"] = vix
        f["vix_change_1d"] = vix.diff()
        vix_ma20 = vix.rolling(20, min_periods=5).mean()
        f["vix_ma20_ratio"] = (vix / vix_ma20.replace(0, np.nan)) - 1.0
        f["vix_high"] = (vix > 25).astype(float)

    # ── 10Y Treasury yield ────────────────────────────────────────────────────
    if "tny" in market_data:
        tny = _align(market_data["tny"])
        f["yield_10y"] = tny
        f["yield_10y_chg_1d"] = tny.diff()

    return f
