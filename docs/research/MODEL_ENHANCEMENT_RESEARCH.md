# Model Enhancement Research

**Branch:** `feat/model-enhancement-research`  
**Date:** 2026-03  
**Purpose:** Research DL/ML models using news/financial reports for out-of-sample accuracy, and feature importance methods for price direction.

---

## Part 1: News & Financial Reports for Out-of-Sample Accuracy

### Current State (Implemented)

- **Capital Market Researcher** (`src/research/capital_market_researcher.py`):
  - **Short-run:** `news_count`, `news_recency_days`, `earnings_announcement_days_ago`, `last_earnings_surprise_pct`, `news_sentiment_mean`, `news_sentiment_std`, `news_positive_ratio`, `news_negative_ratio`
  - **Long-run:** `revenue_growth_yoy`, `net_income_growth_yoy`, `pe_ratio`, `profit_margin`, `avg_earnings_surprise_pct`, `financial_reports_found`
- **News Report Analyzer** (`src/research/news_report_analyzer.py`): FinBERT (when installed) or keyword-based sentiment on news titles; aggregates for model features.
- **SEC Filings**: yfinance `sec_filings` returns 10-K, 10-Q, 8-K with links; displayed in UI.
- **UI**: Streamlit shows news with sentiment labels (🟢/🔴/⚪), SEC filings with links, impact features.
- Research features are **displayed in the UI** and **integrated into LGBM** via sidebar checkbox "Include research features (news, SEC, financials)".

### Literature Findings (2024–2025)

| Approach | Data Source | Model | OOS Improvement |
|----------|-------------|-------|-----------------|
| FinBERT-LSTM | Earnings transcripts, news | Hybrid DL | ~38% lower MSE vs LSTM alone |
| DeBERTa / RoBERTa | News sentiment | Transformer | 75% accuracy for direction |
| Ensemble (FinBERT+DeBERTa+RoBERTa) | News | Ensemble | ~80% accuracy |
| GPT-4o few-shot | Financial text | LLM | Competitive with fine-tuned FinBERT |
| QLoRA + earnings + market indices | Reports + external | Fine-tuned LLM | Improved predictive accuracy |

### Recommended Approaches for This Project

1. **Quick win: Integrate existing impact features**
   - Use `get_impact_features_dict()` from `CapitalMarketResearcher`.
   - Append as static (point-in-time) features per symbol to each backtest window.
   - **Caveat:** Impact features are current-state; for backtest we need historical values. Options:
     - Use current values as proxy (simplest, introduces look-ahead bias if not careful).
     - Build a time-series of impact features by re-running research at historical dates (expensive).
     - Use only features that are less time-sensitive (e.g., `pe_ratio`, `profit_margin`).

2. **Medium effort: Sentiment from news titles**
   - Use FinBERT or a lightweight sentiment model on `CompanyNews.title`.
   - Aggregate: `news_sentiment_mean`, `news_sentiment_std`, `news_count`.
   - Add as features to the LightGBM pipeline.

3. **Higher effort: FinBERT-LSTM or Transformer**
   - FinBERT encodes news/report text; LSTM or Transformer for sequence modeling.
   - Requires: FinBERT (`transformers`), PyTorch/TensorFlow, more data and compute.
   - Best for dedicated research; not necessarily a quick pipeline enhancement.

### Implementation Status

- **Done:** News sentiment (FinBERT when installed, else keyword) → `news_sentiment_mean`, `news_positive_ratio`, etc.
- **Done:** SEC filings from yfinance (10-K, 10-Q, 8-K) with links in UI.
- **Done:** `research_features.py` for pipeline integration; sidebar checkbox enables research features in backtest/forecast.
- **Done:** Feature importance (gain, SHAP, permutation) in `feature_importance.py`.
- **Caveat:** yfinance news/SEC can cause stability issues (exit 139) on some systems; use checkbox only if stable.
- **Future:** FinBERT-LSTM or Transformer for sequence modeling (Phase 2–3).

---

## Part 2: Feature Importance for Price Direction

### Goal

Identify which features most drive predicted stock price **increases** vs **decreases**.

### Methods

| Method | Directional? | Use Case |
|--------|--------------|----------|
| **LightGBM gain/split** | No | Global importance ranking |
| **SHAP (TreeExplainer)** | Yes | Per-feature contribution to each prediction (positive/negative) |
| **Permutation importance** | No | Global importance, model-agnostic |
| **SHAP PermutationExplainer** | Yes | Model-agnostic, directional |

### SHAP for Directional Insight

- **TreeExplainer** (LightGBM): Fast, exact for tree models. Use `shap.TreeExplainer(model, X)`.
- **Summary:** Mean absolute SHAP value = global importance; sign of mean SHAP = directional tendency.
- **Per-instance:** `shap_values[i, j] > 0` means feature `j` pushes prediction up for instance `i`.

### Permutation Importance

- Shuffle feature `j`, measure drop in metric (e.g., MAPE, accuracy).
- Does **not** show direction; use in combination with SHAP for robustness.

### Recommended Workflow

1. **Global importance:** LightGBM `get_feature_importance(importance_type='gain')` (already in codebase).
2. **Directional importance:** Add SHAP TreeExplainer:
   - `mean_shap_by_feature`: average SHAP value per feature (positive = tends to increase prediction).
   - `direction_score`: `(pos_count - neg_count) / total` per feature.
3. **Validation:** Compare top features from gain vs SHAP; flag disagreements for further analysis.

### Implementation (Done)

`src/forecasting/feature_importance.py`:
- `compute_shap_importance(model, X, feature_names, max_samples) -> pd.DataFrame`
- `compute_directional_importance(shap_values, feature_names) -> pd.DataFrame`
- `compute_permutation_importance(model, X, y, metric, n_repeats=5) -> pd.DataFrame`
- `get_lightgbm_feature_importance(model, importance_type='gain') -> pd.DataFrame`

Requires `pip install shap` for SHAP. See `notebooks/model_enhancement_research.ipynb`.

---

## References

- FinBERT-LSTM for stock prediction: [Springer](https://link.springer.com/chapter/10.1007/978-981-96-6438-2_24)
- LLM news sentiment for stock movement: [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260200086S/abstract)
- FinBERT + SHAP explainability: [MDPI](https://www.mdpi.com/2227-7390/13/17/2747)
- SHAP vs Permutation importance: [Amazon Science](https://assets.amazon.science/ea/92/35606b124fe89226a23e02cc1956/a-model-explanation-framework-aligning-shapley-contributions-and-permutation-feature-importance.pdf)
