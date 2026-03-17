"""
Integration of Capital Market Researcher impact features into forecasting.

Adds news/financial report-derived features to improve out-of-sample accuracy.
See docs/research/MODEL_ENHANCEMENT_RESEARCH.md for methodology and caveats.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_research_features_for_symbols(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fetch impact features from Capital Market Researcher for each symbol.

    Args:
        symbols: List of stock tickers

    Returns:
        Dict of symbol -> {feature_name: value}
    """
    try:
        from src.research.capital_market_researcher import CapitalMarketResearcher
    except ImportError:
        logger.warning("CapitalMarketResearcher not available")
        return {}

    cmr = CapitalMarketResearcher()
    result = {}
    for sym in symbols:
        try:
            result[sym] = cmr.get_impact_features_dict(sym)
        except Exception as e:
            logger.warning("Failed to get research features for %s: %s", sym, e)
            result[sym] = {}
    return result


def append_research_features_to_data(
    data: pd.DataFrame,
    symbol: str,
    research_features: Dict[str, float],
) -> pd.DataFrame:
    """
    Append research (impact) features as constant columns to a DataFrame.

    Each row gets the same values (point-in-time snapshot). Use with caution
    in backtesting: these are current-state values, not historical.

    Args:
        data: DataFrame with datetime index (e.g., weekly/daily features)
        symbol: Stock symbol (for lookup in research_features)
        research_features: Dict of symbol -> {feature: value}. If symbol not in
            research_features, returns data unchanged.

    Returns:
        DataFrame with additional columns prefixed by research_
    """
    if symbol not in research_features or not research_features[symbol]:
        return data

    out = data.copy()
    for name, val in research_features[symbol].items():
        out[f"research_{name}"] = float(val) if val is not None else 0.0
    return out


def build_features_with_research(
    price_features: pd.DataFrame,
    symbol: str,
    research_cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Build feature matrix by merging price features with research features.

    Args:
        price_features: DataFrame from create_daily_features or create_weekly_features
        symbol: Stock symbol
        research_cache: Pre-fetched research features. If None, fetches on demand.

    Returns:
        DataFrame with price + research features
    """
    if research_cache is None:
        research_cache = get_research_features_for_symbols([symbol])

    return append_research_features_to_data(price_features, symbol, research_cache)
