---
name: stock-forecast-analysis
description: >-
  Guides data analysis, ML, and visualization work in the stock-forecast project.
  Use when analyzing stock data, building forecasting models, creating visualizations,
  developing Jupyter notebooks, or working with pandas, scikit-learn, or LightGBM.
---

# Stock Forecast Analysis

## Quick Start

When working on data analysis, ML, or notebooks in this project:

1. Use pandas for manipulation; prefer method chaining and vectorized operations
2. Use matplotlib for control, seaborn for statistical plots
3. Implement 5-fold stratified cross-validation for ML
4. Define functions in `src/` modules; import into notebooks
5. Follow project structure (see [reference.md](reference.md))

## Key Principles

- **Concise, technical responses** with accurate Python examples
- **Functional over classes** where appropriate
- **Vectorized operations** over explicit loops
- **Enhance in place**: Update existing method names, don't create new ones
- **Statistical validation**: Cross-validation, hyperparameter tuning, significance testing

## Project Structure

| Path | Purpose |
|------|---------|
| `src/` | Production-ready code |
| `src/data_preprocess/` | Data loading, cleaning, feature engineering |
| `src/forecasting/` | LGBM, backtesting, trading simulation, feature importance |
| `src/research/` | Capital market researcher, news sentiment, SEC filings |
| `src/feature_engineering/` | Lag, rolling, technical features |
| `tests/` | Unit and integration tests |
| `notebooks/` | Backtesting, model enhancement research |
| `config/` | S&P 100, market indices, YAML config |
| `docs/research/` | Research documentation |

## ML Validation Checklist

- [ ] 5-fold stratified cross-validation
- [ ] GridSearchCV (small params) or RandomizedSearchCV (large params)
- [ ] Paired t-tests for model comparison
- [ ] 95% CI for performance estimates
- [ ] Shallow trees, focused hyperparameter ranges for speed

## Additional Resources

- For detailed conventions and examples, see [reference.md](reference.md)
