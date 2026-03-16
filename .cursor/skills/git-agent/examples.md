# Git Agent — Examples

## Commit Message Examples

```
feat(forecasting): add weekly aggregator for multi-horizon predictions
fix(data_preprocess): handle Yahoo Finance API rate limits
refactor(skills): create testing-agent for unit and integration tests
docs(readme): update AI agent setup section
test(forecasting): add LGBMForecaster unit tests
chore(deps): bump pandas to 2.1
```

## PR Title and Body Example

**Title:** `feat(forecasting): add LGBM backtester`

**Body:**
```markdown
## Summary
Adds standalone backtester for evaluating LGBM forecasting performance.

## Changes
- New `StandaloneBacktester` class in `src/forecasting/`
- Integrates with `TimeSeriesBacktester` for walk-forward validation
- Outputs metrics: Sharpe, max drawdown, win rate

## Notes
- Requires `config/stocks_config.yaml` for symbol list
```

## Commands Reference

| Action | Command |
|--------|---------|
| Check status | `git status` |
| Stage all | `git add -A` |
| Stage specific | `git add path/to/file.py` |
| Commit | `git commit -m "type(scope): message"` |
| Push | `git push origin <branch>` |
| Set upstream | `git push -u origin <branch>` |
| Create branch | `git checkout -b feature-name` |
