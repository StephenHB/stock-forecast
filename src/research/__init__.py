"""
Research Agent for Stock Forecasting Algorithms

This module provides a research agent that can search for and learn
new forecasting algorithms to improve prediction accuracy, especially
for short-horizon (daily) forecasting.
"""

from .research_agent import ResearchAgent, ForecastingResearch

__all__ = ["ResearchAgent", "ForecastingResearch"]
