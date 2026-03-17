"""
Research Agent for Stock Forecasting Algorithms

This module provides:
- ResearchAgent: Searches for forecasting algorithms
- CapitalMarketResearcher: Reads news and financial reports, builds impact features
- news_report_analyzer: Sentiment analysis on news, summarization of SEC filings
"""

from .research_agent import ResearchAgent, ForecastingResearch
from .capital_market_researcher import (
    CapitalMarketResearcher,
    ImpactFeatures,
    CompanyNews,
    FinancialReport,
)
from .news_report_analyzer import (
    analyze_news_sentiment,
    aggregate_news_sentiment,
    summarize_financial_reports,
    get_news_sentiment_features,
    NewsAnalysis,
    ReportSummary,
)

__all__ = [
    "ResearchAgent",
    "ForecastingResearch",
    "CapitalMarketResearcher",
    "ImpactFeatures",
    "CompanyNews",
    "FinancialReport",
    "analyze_news_sentiment",
    "aggregate_news_sentiment",
    "summarize_financial_reports",
    "get_news_sentiment_features",
    "NewsAnalysis",
    "ReportSummary",
]
