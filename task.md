# Stock Forecast Tasks

## Daily Forecasting Accuracy (merged)

- [x] Create branch feat/daily-forecasting-accuracy
- [x] Build research agent with online search for forecasting algorithms
- [x] Add daily features for horizon <= 5 days
- [x] Integrate improvements into forecasting pipeline

## Summary

When forecast horizon <= 5 days, the app now uses:
- **Daily data** (not weekly aggregation)
- **Daily volatility features**: returns, rolling vol, high-low range, Parkinson vol, ADR
- **Research agent** for algorithm discovery (optional: `pip install duckduckgo-search`)

## Files Changed

- `src/research/` - Research agent module
- `src/feature_engineering/daily_volatility_features.py` - Daily volatility features
- `src/forecasting/feature_factory.py` - Feature creation for daily vs weekly
- `app.py` - Dual-path backtest/forecast (daily when horizon <= 5)
- `pyproject.toml` - Optional research dependency

## Capital Market Researcher Agent (feat/capital-market-researcher-agent)

- [x] Create branch
- [x] Search financial reports (10-K, 10-Q, SEC) via ResearchAgent
- [x] Fetch top 10 company news via yfinance
- [x] Build short-run impact features: news_count, news_recency_days, earnings_announcement_days_ago, last_earnings_surprise_pct
- [x] Build long-run impact features: revenue_growth_yoy, net_income_growth_yoy, pe_ratio, profit_margin, avg_earnings_surprise_pct, financial_reports_found
- [x] Integrate into app (expander section)
