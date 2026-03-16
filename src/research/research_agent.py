"""
Research Agent for Forecasting Algorithm Discovery

Searches online for forecasting algorithms and features that improve
short-horizon stock price prediction, especially for reducing prediction
smoothness and capturing daily volatility.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ForecastingResearch:
    """Structured research findings for forecasting improvements."""

    algorithms: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    query: str = ""


class ResearchAgent:
    """
    Agent that searches for forecasting algorithms and features.

    Uses online search (when available) to discover algorithms that
    capture daily volatility and reduce prediction smoothness for
    short-horizon (<= 5 days) forecasting.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_online_search: bool = True,
    ):
        """
        Initialize the research agent.

        Args:
            cache_dir: Directory to cache research findings. If None, uses default.
            use_online_search: Whether to attempt online search (requires duckduckgo-search).
        """
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_online_search = use_online_search
        self._search_available = self._check_search_available()

    def _check_search_available(self) -> bool:
        """Check if online search backend is available."""
        if not self.use_online_search:
            return False
        try:
            from duckduckgo_search import DDGS

            return True
        except ImportError:
            logger.info(
                "duckduckgo-search not installed. Install with: pip install duckduckgo-search"
            )
            return False

    def search(self, query: str, max_results: int = 5) -> ForecastingResearch:
        """
        Search for forecasting algorithms and features.

        Args:
            query: Search query (e.g., "daily stock forecasting volatility features")
            max_results: Maximum number of results to return.

        Returns:
            ForecastingResearch with algorithms, features, and sources.
        """
        cache_path = self.cache_dir / f"research_{hash(query) % 10**8}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                return ForecastingResearch(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        research = ForecastingResearch(query=query)

        if self._search_available:
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))
                for r in results:
                    research.sources.append(r.get("href", ""))
                    title = r.get("title", "")
                    body = r.get("body", "")
                    research.algorithms.append(f"{title}: {body[:200]}...")
            except Exception as e:
                logger.warning("Online search failed: %s", e)

        if not research.algorithms:
            research = self._get_builtin_research(query)

        try:
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "algorithms": research.algorithms,
                        "features": research.features,
                        "sources": research.sources,
                        "query": research.query,
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            logger.warning("Could not cache research: %s", e)

        return research

    def _get_builtin_research(self, query: str) -> ForecastingResearch:
        """
        Return built-in research findings (from prior research).

        Used when online search is unavailable or returns no results.
        """
        research = ForecastingResearch(query=query)

        # Findings from research on daily forecasting and volatility capture
        research.algorithms = [
            "Decomposition: Separate trend and fluctuation components; model fluctuation separately",
            "Volatility-focused: Use OHLC-based estimators (Parkinson, Garman-Klass) for daily range",
            "Daily features: Rolling std of returns, high-low range, average daily range (ADR)",
            "Short-horizon: Use daily data (not weekly) when forecast horizon <= 5 days",
        ]
        research.features = [
            "daily_returns_1d, 2d, 3d, 5d (pct_change)",
            "rolling_volatility_5d, 10d, 20d (std of returns)",
            "high_low_range_pct ((High-Low)/Close)",
            "parkinson_vol (log(High/Low) - intraday volatility proxy)",
            "average_daily_range_20d (rolling mean of High-Low)",
            "volume_price_dynamics (volume * price change)",
        ]
        research.sources = [
            "Built-in research (arxiv, alpharithms, volstats)",
        ]

        return research

    def get_daily_features_for_short_horizon(self) -> List[str]:
        """
        Get recommended features for short-horizon (<= 5 days) forecasting.

        Returns list of feature names to implement based on research.
        """
        research = self.search(
            "short-term stock forecasting daily volatility features machine learning"
        )
        return research.features if research.features else [
            "daily_returns_1d",
            "rolling_volatility_5d",
            "high_low_range_pct",
            "average_daily_range_20d",
        ]
