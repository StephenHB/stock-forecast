"""
Feature importance analysis for stock price prediction.

Provides SHAP, permutation importance, and directional analysis to identify
which features drive predicted price increases vs decreases.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_samples: Optional[int] = 500,
) -> pd.DataFrame:
    """
    Compute SHAP values for tree-based models (LightGBM, XGBoost, etc.).

    Returns mean absolute SHAP and mean SHAP (directional) per feature.

    Args:
        model: Fitted tree model with .predict() and tree structure
        X: Feature matrix
        feature_names: List of feature names (default: X.columns)
        max_samples: Cap samples for SHAP (default 500 for speed)

    Returns:
        DataFrame with columns: feature, mean_abs_shap, mean_shap, direction_tendency
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; pip install shap")
        return pd.DataFrame()

    names = feature_names or list(X.columns)
    X_sample = X.head(max_samples) if max_samples and len(X) > max_samples else X

    try:
        # LightGBM/XGBoost: use booster for TreeExplainer
        if hasattr(model, "booster_"):
            explainer = shap.TreeExplainer(model.booster_)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # binary/multi: use first class
    except Exception as e:
        logger.warning("TreeExplainer failed, trying KernelExplainer: %s", e)
        try:
            explainer = shap.KernelExplainer(model.predict, X_sample.head(50))
            shap_values = explainer.shap_values(X_sample.head(100), nsamples=50)
        except Exception as e2:
            logger.warning("KernelExplainer failed: %s", e2)
            return pd.DataFrame()

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    direction = np.where(mean_shap > 0, "up", np.where(mean_shap < 0, "down", "neutral"))

    return pd.DataFrame({
        "feature": names[: len(mean_abs)],
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_shap,
        "direction_tendency": direction,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compute_directional_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute directional importance from precomputed SHAP values.

    direction_score: (pos_contributions - neg_contributions) / total per feature.

    Args:
        shap_values: 2D array (n_samples, n_features)
        feature_names: List of feature names

    Returns:
        DataFrame with feature, direction_score, up_pct, down_pct
    """
    n_features = shap_values.shape[1]
    names = feature_names[:n_features]

    up_count = (shap_values > 0).sum(axis=0)
    down_count = (shap_values < 0).sum(axis=0)
    total = shap_values.shape[0]
    direction_score = (up_count - down_count) / total if total > 0 else np.zeros(n_features)
    up_pct = up_count / total * 100 if total > 0 else np.zeros(n_features)
    down_pct = down_count / total * 100 if total > 0 else np.zeros(n_features)

    return pd.DataFrame({
        "feature": names,
        "direction_score": direction_score,
        "up_pct": up_pct,
        "down_pct": down_pct,
    }).sort_values("direction_score", ascending=False, key=abs).reset_index(drop=True)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int = 5,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Compute permutation feature importance (model-agnostic).

    Shuffles each feature and measures drop in metric. Higher drop = more important.

    Args:
        model: Fitted model with .predict(X)
        X: Feature matrix
        y: Target
        metric: (y_true, y_pred) -> float (lower is better, e.g., MAPE)
        n_repeats: Number of shuffle repeats
        random_state: Random seed

    Returns:
        DataFrame with feature, importance (metric increase when shuffled), std
    """
    rng = np.random.default_rng(random_state)
    baseline_pred = model.predict(X)
    baseline_score = metric(y, baseline_pred)

    results = []
    for col in X.columns:
        X_perm = X.copy()
        increases = []
        for _ in range(n_repeats):
            X_perm[col] = rng.permutation(X_perm[col].values)
            pred = model.predict(X_perm)
            score = metric(y, pred)
            increases.append(score - baseline_score)  # positive = worse when shuffled
        results.append({
            "feature": col,
            "importance": np.mean(increases),
            "std": np.std(increases),
        })

    return pd.DataFrame(results).sort_values("importance", ascending=False).reset_index(drop=True)


def get_lightgbm_feature_importance(
    model: Any,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Get feature importance from a fitted LightGBM model.

    Args:
        model: Fitted LGBMRegressor or Booster
        importance_type: 'gain' or 'split'

    Returns:
        DataFrame with feature, importance, direction_tendency (from gain sign if available)
    """
    if hasattr(model, "booster_"):
        booster = model.booster_
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = model.feature_name_ if hasattr(model, "feature_name_") else [f"f{i}" for i in range(len(imp))]
        return pd.DataFrame({
            "feature": names,
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        booster = model

    if importance_type == "gain":
        imp = booster.feature_importance(importance_type="gain")
    else:
        imp = booster.feature_importance(importance_type="split")
    names = booster.feature_name()

    return pd.DataFrame({
        "feature": names,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
