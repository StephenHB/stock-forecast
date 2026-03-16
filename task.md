# Daily Forecasting Accuracy

## Status

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
