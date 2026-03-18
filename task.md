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

## Model Enhancement Research (feat/model-enhancement-research)

- [x] Add news sentiment (keyword-based lexicon)
- [x] Add SEC filings (10-K, 10-Q, 8-K) via yfinance
- [x] Add feature importance (gain, SHAP, permutation)
- [x] Document research findings in `docs/research/MODEL_ENHANCEMENT_RESEARCH.md`
- [x] PR: Merge `feat/model-enhancement-research` → `main`

**Note:** Capital Market Research expander uses yfinance; expanding it may cause connection issues on some systems. Research features are UI-only; pipeline integration (`research_features.py`) is optional and not enabled by default.

## LGBM 2-Stage (feat/lgbm-2stage)

- [x] Create branch
- [x] Add Prophet trend/seasonality features for LGBM input
- [x] Add MA fallback when data too small (< 60 rows)
- [x] Integrate into create_daily_features and create_weekly_features
- [x] Add get_forecast_trend_seasonality for prediction phase (future dates)
- [x] Information leakage audit: `docs/LGBM_INFORMATION_LEAKAGE_AUDIT.md`
- [ ] PR: Merge `feat/lgbm-2stage` → `main`

**Optional:** `pip install prophet` or `pip install stock-forecast[prophet]` for Prophet; otherwise MA is used.
