"""
Capital Market Researcher Agent

Reads news and financial reports like real capital market researchers,
understanding key features that drive stock prices. Uses ResearchAgent
for report discovery and builds short-run and long-run impact features.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompanyNews:
    """Single news item for a company."""

    title: str
    description: str
    pub_date: Optional[str]
    provider: str
    url: Optional[str] = None


@dataclass
class FinancialReport:
    """Financial report reference (from search)."""

    title: str
    url: str
    report_type: str  # e.g., "10-K", "10-Q", "earnings"


@dataclass
class ImpactFeatures:
    """Short-run and long-run impact features for analysis."""

    short_run: Dict[str, float] = field(default_factory=dict)
    long_run: Dict[str, float] = field(default_factory=dict)
    news: List[CompanyNews] = field(default_factory=list)
    reports: List[FinancialReport] = field(default_factory=list)


class CapitalMarketResearcher:
    """
    Agent that researches company news and financial reports to build
    impact features for stock price analysis.
    """

    def __init__(
        self,
        research_agent=None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the capital market researcher.

        Args:
            research_agent: ResearchAgent instance for report search. If None, creates one.
            cache_dir: Directory for caching. If None, uses default.
        """
        from .research_agent import ResearchAgent

        self.research_agent = research_agent or ResearchAgent(use_online_search=True)
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_company_news(self, symbol: str, count: int = 10) -> List[CompanyNews]:
        """
        Fetch top N news articles for the company.

        Args:
            symbol: Stock ticker symbol (e.g., AAPL)
            count: Number of news articles (default 10)

        Returns:
            List of CompanyNews
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            raw = ticker.get_news(count=count)
            news = []
            for item in raw or []:
                content = item.get("content", {})
                if isinstance(content, dict):
                    news.append(
                        CompanyNews(
                            title=content.get("title", ""),
                            description=content.get("description", "")[:500]
                            if content.get("description")
                            else "",
                            pub_date=content.get("pubDate"),
                            provider=content.get("provider", {}).get(
                                "displayName", "Unknown"
                            ),
                            url=content.get("canonicalUrl") or content.get(
                                "clickThroughUrl"
                            ),
                        )
                    )
            return news
        except Exception as e:
            logger.warning("Failed to fetch news for %s: %s", symbol, e)
            return []

    def search_financial_reports(
        self, symbol: str, max_results: int = 10
    ) -> List[FinancialReport]:
        """
        Search for financial reports (10-K, 10-Q, earnings) of the company.

        Uses yfinance sec_filings when available, then falls back to ResearchAgent search.

        Args:
            symbol: Stock ticker symbol
            max_results: Maximum search results

        Returns:
            List of FinancialReport
        """
        reports = []
        seen = set()

        # 1. Try yfinance sec_filings (10-K, 10-Q, 8-K, etc.)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            filings = getattr(ticker, "sec_filings", None)
            if filings and isinstance(filings, list):
                for item in filings[:max_results]:
                    if isinstance(item, dict):
                        form_type = item.get("type", "filing")
                        title = item.get("title", form_type)
                        url = item.get("edgarUrl", item.get("link", ""))
                        key = (form_type, str(item.get("date", "")), title[:40])
                        if key not in seen:
                            seen.add(key)
                            reports.append(
                                FinancialReport(
                                    title=f"{form_type}: {title}"[:100],
                                    url=url or "",
                                    report_type=form_type,
                                )
                            )
        except Exception as e:
            logger.debug("yfinance sec_filings not available: %s", e)

        # 2. Fallback: ResearchAgent search
        if len(reports) < max_results:
            query = f"{symbol} 10-K 10-Q annual report earnings SEC filing"
            research = self.research_agent.search(query, max_results=max_results)
            for i, (src, algo) in enumerate(
                zip(research.sources, research.algorithms)
            ):
                report_type = "10-K" if "10-K" in algo or "annual" in algo.lower() else "10-Q" if "10-Q" in algo else "earnings"
                title = algo.split(":")[0] if ":" in algo else algo[:80]
                key = (report_type, title[:50])
                if key not in seen:
                    seen.add(key)
                    reports.append(
                        FinancialReport(
                            title=title[:100],
                            url=src or "",
                            report_type=report_type,
                        )
                    )
        return reports[:max_results]

    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get key financial metrics from yfinance (income, balance sheet, earnings).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict of financial metrics
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info
            income = ticker.income_stmt
            earnings = ticker.earnings_dates

            metrics = {}
            if info:
                metrics["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
                metrics["market_cap"] = info.get("marketCap")
                metrics["revenue"] = info.get("totalRevenue")
                metrics["profit_margin"] = info.get("profitMargins")

            if income is not None and not income.empty:
                if "Total Revenue" in income.index:
                    rev = income.loc["Total Revenue"]
                    if len(rev) >= 2:
                        metrics["revenue_growth_yoy"] = float(
                            (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1] * 100
                        )
                if "Net Income" in income.index:
                    ni = income.loc["Net Income"]
                    if len(ni) >= 2:
                        metrics["net_income_growth_yoy"] = float(
                            (ni.iloc[0] - ni.iloc[1]) / abs(ni.iloc[1]) * 100
                            if ni.iloc[1] != 0
                            else 0
                        )

            if earnings is not None and not earnings.empty:
                reported = earnings[earnings["Reported EPS"].notna()]
                if not reported.empty:
                    surprise = reported["Surprise(%)"].dropna()
                    metrics["avg_earnings_surprise_pct"] = (
                        float(surprise.mean()) if len(surprise) > 0 else None
                    )

            return metrics
        except Exception as e:
            logger.warning("Failed to fetch financials for %s: %s", symbol, e)
            return {}

    def _parse_pub_date(self, pub_date: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string to datetime."""
        if not pub_date:
            return None
        try:
            from datetime import datetime as dt

            return dt.fromisoformat(
                pub_date.replace("Z", "+00:00").replace("z", "+00:00")
            )
        except (ValueError, TypeError):
            return None

    def research(self, symbol: str) -> ImpactFeatures:
        """
        Full research: news, financial reports, and impact features.

        Args:
            symbol: Stock ticker symbol

        Returns:
            ImpactFeatures with short-run, long-run features, news, and reports
        """
        news = self.get_company_news(symbol, count=10)
        reports = self.search_financial_reports(symbol, max_results=10)
        metrics = self.get_financial_metrics(symbol)

        now = datetime.now(timezone.utc)
        short_run = {}
        long_run = {}

        # Short-run impact features
        short_run["news_count"] = float(len(news))
        if news:
            dates = [
                self._parse_pub_date(n.pub_date)
                for n in news
                if n.pub_date
            ]
            dates = [d for d in dates if d is not None]
            if dates:
                most_recent = max(dates)
                short_run["news_recency_days"] = (
                    now - most_recent.replace(tzinfo=timezone.utc)
                ).days if most_recent.tzinfo else (now - most_recent).days
            else:
                short_run["news_recency_days"] = 999.0
        else:
            short_run["news_recency_days"] = 999.0

        # Earnings-related short-run
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            earnings = ticker.earnings_dates
            if earnings is not None and not earnings.empty:
                reported = earnings[earnings["Reported EPS"].notna()]
                if not reported.empty:
                    last_earnings = reported.index[0]
                    if hasattr(last_earnings, "to_pydatetime"):
                        last_dt = last_earnings.to_pydatetime()
                    else:
                        last_dt = last_earnings
                    if hasattr(last_dt, "replace"):
                        last_dt = last_dt.replace(
                            tzinfo=last_dt.tzinfo or timezone.utc
                        )
                    delta = now - last_dt
                    short_run["earnings_announcement_days_ago"] = (
                        delta.days if hasattr(delta, "days") else 999.0
                    )
                    surprise = reported["Surprise(%)"].iloc[0]
                    short_run["last_earnings_surprise_pct"] = (
                        float(surprise)
                        if surprise is not None
                        and str(surprise) != "nan"
                        else 0.0
                    )
                else:
                    short_run["earnings_announcement_days_ago"] = 999.0
                    short_run["last_earnings_surprise_pct"] = 0.0
            else:
                short_run["earnings_announcement_days_ago"] = 999.0
                short_run["last_earnings_surprise_pct"] = 0.0
        except Exception:
            short_run["earnings_announcement_days_ago"] = 999.0
            short_run["last_earnings_surprise_pct"] = 0.0

        # Long-run impact features
        long_run["revenue_growth_yoy"] = metrics.get("revenue_growth_yoy") or 0.0
        long_run["net_income_growth_yoy"] = (
            metrics.get("net_income_growth_yoy") or 0.0
        )
        long_run["pe_ratio"] = metrics.get("pe_ratio") or 0.0
        long_run["profit_margin"] = (
            float(metrics["profit_margin"] * 100)
            if metrics.get("profit_margin") is not None
            else 0.0
        )
        long_run["avg_earnings_surprise_pct"] = (
            metrics.get("avg_earnings_surprise_pct") or 0.0
        )
        long_run["financial_reports_found"] = float(len(reports))

        # News sentiment (from news_report_analyzer)
        try:
            from .news_report_analyzer import get_news_sentiment_features, analyze_news_sentiment
            analyses = analyze_news_sentiment([n.title for n in news])
            for k, v in get_news_sentiment_features(analyses).items():
                short_run[k] = v
        except Exception as e:
            logger.debug("News sentiment analysis skipped: %s", e)

        return ImpactFeatures(
            short_run=short_run,
            long_run=long_run,
            news=news,
            reports=reports,
        )

    def get_impact_features_dict(self, symbol: str) -> Dict[str, float]:
        """
        Get impact features as a flat dict for use in forecasting pipelines.

        Keys are prefixed with short_run_ or long_run_.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict of feature name -> value
        """
        result = self.research(symbol)
        features = {}
        for k, v in result.short_run.items():
            features[f"short_run_{k}"] = v
        for k, v in result.long_run.items():
            features[f"long_run_{k}"] = v
        return features
