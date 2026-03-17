"""
News and financial report analysis for stock forecasting.

Provides sentiment analysis on news titles and summarization of financial reports
to derive features for model enhancement.
"""

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Simple finance-oriented sentiment lexicon (subset for titles)
POSITIVE_WORDS = {
    "surge", "soar", "rally", "gain", "growth", "beat", "outperform", "upgrade",
    "bullish", "record", "strong", "profit", "revenue", "success", "launch",
    "breakthrough", "deal", "partnership", "buyback", "dividend", "raise",
}
NEGATIVE_WORDS = {
    "fall", "drop", "plunge", "decline", "miss", "cut", "downgrade", "bearish",
    "loss", "layoff", "warning", "concern", "risk", "ban", "lawsuit", "probe",
    "recall", "default", "bankruptcy", "recession",
}


@dataclass
class NewsAnalysis:
    """Analysis result for a single news item."""

    title: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # "positive", "negative", "neutral"
    matched_positive: List[str] = field(default_factory=list)
    matched_negative: List[str] = field(default_factory=list)


@dataclass
class ReportSummary:
    """Summary of financial reports found."""

    total_count: int
    by_type: dict  # report_type -> count
    report_titles: List[str] = field(default_factory=list)
    has_10k: bool = False
    has_10q: bool = False
    has_earnings: bool = False


def analyze_news_sentiment(
    titles: List[str],
    positive_words: Optional[set] = None,
    negative_words: Optional[set] = None,
) -> List[NewsAnalysis]:
    """
    Analyze sentiment of news titles using a simple keyword-based approach.

    Args:
        titles: List of news titles
        positive_words: Set of positive keywords (default: finance-oriented)
        negative_words: Set of negative keywords (default: finance-oriented)

    Returns:
        List of NewsAnalysis per title
    """
    pos = positive_words or POSITIVE_WORDS
    neg = negative_words or NEGATIVE_WORDS
    results = []

    for title in titles:
        text = title.lower()
        words = set(re.findall(r"\b\w+\b", text))
        matched_pos = [w for w in words if w in pos]
        matched_neg = [w for w in words if w in neg]

        n_pos = len(matched_pos)
        n_neg = len(matched_neg)
        total = n_pos + n_neg
        if total == 0:
            score = 0.0
            label = "neutral"
        else:
            score = (n_pos - n_neg) / total
            score = max(-1.0, min(1.0, score))
            label = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")

        results.append(
            NewsAnalysis(
                title=title,
                sentiment_score=score,
                sentiment_label=label,
                matched_positive=matched_pos,
                matched_negative=matched_neg,
            )
        )
    return results


def aggregate_news_sentiment(analyses: List[NewsAnalysis]) -> dict:
    """
    Aggregate sentiment across news items into summary stats.

    Returns:
        Dict with mean_sentiment, positive_count, negative_count, neutral_count
    """
    if not analyses:
        return {"mean_sentiment": 0.0, "positive_count": 0, "negative_count": 0, "neutral_count": 0}

    scores = [a.sentiment_score for a in analyses]
    labels = [a.sentiment_label for a in analyses]
    return {
        "mean_sentiment": sum(scores) / len(scores),
        "positive_count": sum(1 for l in labels if l == "positive"),
        "negative_count": sum(1 for l in labels if l == "negative"),
        "neutral_count": sum(1 for l in labels if l == "neutral"),
        "n_news": len(analyses),
    }


def summarize_financial_reports(
    reports: List,
    report_type_attr: str = "report_type",
    title_attr: str = "title",
) -> ReportSummary:
    """
    Summarize financial reports (10-K, 10-Q, earnings).

    Args:
        reports: List of FinancialReport or similar objects
        report_type_attr: Attribute name for report type
        title_attr: Attribute name for title

    Returns:
        ReportSummary
    """
    by_type = {}
    titles = []
    has_10k = has_10q = has_earnings = False

    for r in reports:
        rtype = getattr(r, report_type_attr, "unknown")
        title = getattr(r, title_attr, "")
        by_type[rtype] = by_type.get(rtype, 0) + 1
        titles.append(title)
        if "10-K" in rtype or "annual" in rtype.lower():
            has_10k = True
        if "10-Q" in rtype:
            has_10q = True
        if "earnings" in rtype.lower():
            has_earnings = True

    return ReportSummary(
        total_count=len(reports),
        by_type=by_type,
        report_titles=titles,
        has_10k=has_10k,
        has_10q=has_10q,
        has_earnings=has_earnings,
    )


def get_news_sentiment_features(analyses: List[NewsAnalysis]) -> dict:
    """
    Convert news sentiment analysis to numeric features for the model.

    Returns:
        Dict with news_sentiment_mean, news_sentiment_std, news_positive_ratio, etc.
    """
    agg = aggregate_news_sentiment(analyses)
    n = agg["n_news"]
    if n == 0:
        return {
            "news_sentiment_mean": 0.0,
            "news_sentiment_std": 0.0,
            "news_positive_ratio": 0.0,
            "news_negative_ratio": 0.0,
        }

    scores = [a.sentiment_score for a in analyses]
    return {
        "news_sentiment_mean": agg["mean_sentiment"],
        "news_sentiment_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "news_positive_ratio": agg["positive_count"] / n,
        "news_negative_ratio": agg["negative_count"] / n,
    }
