"""
Research Agent for Stock Forecasting Algorithms

This module provides:
- ResearchAgent: Searches for forecasting algorithms
- CapitalMarketResearcher: Reads news and financial reports, builds impact features
"""

from .research_agent import ResearchAgent, ForecastingResearch
from .capital_market_researcher import (
    CapitalMarketResearcher,
    ImpactFeatures,
    CompanyNews,
    FinancialReport,
)

__all__ = [
    "ResearchAgent",
    "ForecastingResearch",
    "CapitalMarketResearcher",
    "ImpactFeatures",
    "CompanyNews",
    "FinancialReport",
]
