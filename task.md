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
- [x] Integrate research features (news, SEC, financials) into run_backtest and run_forecast
- [x] Add sidebar checkbox "Include research features (news, SEC, financials)"
- [x] Add optional FinBERT for news sentiment (falls back to keyword when not installed)
- [ ] PR: Merge `feat/lgbm-2stage` → `main`

**Optional:** `pip install stock-forecast[prophet]` for Prophet; `pip install stock-forecast[finbert]` for FinBERT. Research features use yfinance; on some systems this can cause crashes—use checkbox only if stable.

## Trading Simulation Strategy Review (review-trading-simulation-strategy)

- [x] Create branch `review-trading-simulation-strategy`
- [x] Audit existing buy/sell logic: identified 3 bugs (wrong signal reference price, overlapping forecast windows, signal vs execution price mismatch)
- [x] Fix 1: Signal reference — use Close[T] (not Close[T-1]) as reference price in `trading_simulator.py`
- [x] Fix 2: Non-overlapping windows — set `step_size=forecast_days` in `app.py` `run_backtest` for daily mode
- [x] Fix 3: Update simulation tab caption to accurately describe corrected logic
- [x] Research agent investigation: identified 5 root causes why ML forecast strategy struggles vs buy-and-hold (level prediction, equity drift, all-in/all-out whipsaw, no threshold, no costs, no regime awareness)
- [x] Enhancement 1: Implied-return signal — derive `(pred − Close[T]) / Close[T]` as the signal magnitude
- [x] Enhancement 2: Dead-zone threshold — configurable parameter (default 0.5%), HOLD inside dead zone
- [x] Enhancement 3: Proportional position sizing — BUY allocates confidence% of cash (scales threshold → 3× threshold)
- [x] Enhancement 4: Transaction cost model — configurable per-side cost (default 0.1%)
- [x] Enhancement 5: 200-day MA regime filter — suppress SELL signals during uptrend
- [x] UI: Add "Simulation Settings" expander with threshold and transaction cost sliders
- [x] UI: Add "How This Strategy Works" expander in Simulation tab with 5-rule table
- [x] UI: Add "What to Do Now" live recommendation cards (BUY / SELL / HOLD) per stock
- [x] UI: Rename sidebar "Backtest Summary" → "Forecast Summary"; show predicted price + % change (colour-coded)
- [x] UI: Add Holds and Costs columns to simulation results table
- [x] Docs: Update README with Trading Simulation Strategy section and 5-rule table
- [x] Docs: Update task.md with this task log
- [ ] PR: Merge `review-trading-simulation-strategy` → `main`
