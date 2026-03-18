"""
Adaptive LGBM Hyperparameter Tuner

Uses the "1-lag-back" window strategy: for a forecast horizon of N days,
the N-day window immediately preceding the current (unknown) window is used
as the validation set to find the best LightGBM hyperparameters.

The tuned parameters are then applied to the full training data, so both
the live forecast and every backtest window benefit from params that were
selected against the most recent market behaviour rather than pre-fixed.

Tuning is kept fast by:
- Fixing n_estimators at 200 during candidate evaluation (avoids sweeping
  hundreds of trees per candidate)
- Using a focused param space over the 5 most impactful LGBM hyperparameters
- Defaulting to n_iter=20 random draws (≈ 2-3 seconds per window)
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)

# ── Search space ─────────────────────────────────────────────────────────────
# n_estimators is NOT tuned here; it is fixed at 200 during evaluation so that
# each candidate is evaluated cheaply.  The winning combo of the other 5 params
# is what matters for generalisation.
PARAM_GRID: dict[str, list] = {
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.03, 0.05, 0.1, 0.15],
    "num_leaves":       [15, 31, 63],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# Fixed settings applied to every candidate and to the final model
_FIXED = {
    "n_estimators":  200,
    "reg_alpha":     0.1,
    "reg_lambda":    0.1,
    "random_state":  42,
    "verbose":       -1,
}

DEFAULT_PARAMS: dict = {**_FIXED, "max_depth": 5, "learning_rate": 0.1,
                        "num_leaves": 31, "subsample": 0.9,
                        "colsample_bytree": 0.9}


def tune_lgbm_params(
    X_tune_train: np.ndarray,
    y_tune_train: np.ndarray,
    X_tune_val: np.ndarray,
    y_tune_val: np.ndarray,
    n_iter: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Random search over PARAM_GRID evaluated on the 1-lag-back validation set.

    Parameters
    ----------
    X_tune_train : array-like, shape (n_tune_train, n_features)
        Scaled feature matrix for the tuning training portion.
    y_tune_train : array-like, shape (n_tune_train,)
        Target values for the tuning training portion.
    X_tune_val : array-like, shape (forecast_horizon, n_features)
        Scaled feature matrix for the 1-lag-back validation window.
    y_tune_val : array-like, shape (forecast_horizon,)
        Known targets for the validation window.
    n_iter : int
        Number of random hyperparameter combinations to evaluate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Best hyperparameter dict (ready to pass to ``lgb.LGBMRegressor``).
        Falls back to DEFAULT_PARAMS when data is too small or all
        candidates fail.
    """
    if len(X_tune_train) < 20 or len(X_tune_val) < 1:
        logger.debug("Tuning skipped: insufficient samples (train=%d, val=%d)",
                     len(X_tune_train), len(X_tune_val))
        return DEFAULT_PARAMS.copy()

    rng = random.Random(random_state)
    best_mape = float("inf")
    best_params = DEFAULT_PARAMS.copy()

    for _ in range(n_iter):
        candidate = {k: rng.choice(v) for k, v in PARAM_GRID.items()}
        candidate.update(_FIXED)
        try:
            model = lgb.LGBMRegressor(**candidate)
            model.fit(X_tune_train, y_tune_train)
            preds = model.predict(X_tune_val)
            mape = mean_absolute_percentage_error(y_tune_val, preds)
            if mape < best_mape:
                best_mape = mape
                best_params = candidate.copy()
        except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
            logger.debug("Candidate failed: %s", exc)

    logger.debug("Tuning done — best MAPE=%.4f  params=%s", best_mape, best_params)
    return best_params


def build_lag_back_splits(
    X: np.ndarray,
    y: np.ndarray,
    forecast_horizon: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split (X, y) into tune-train and tune-val using the 1-lag-back window.

    The validation set is the ``forecast_horizon`` rows immediately before the
    last ``forecast_horizon`` rows (which correspond to the current, unknown
    window).  The tuning training set is everything that precedes it.

    Layout
    ------
    [  tune_train  |  tune_val (lag-back)  |  (current window — targets unknown)  ]
                   ← forecast_horizon →    ←      forecast_horizon        →

    Returns None when there are not enough rows to form both splits.
    """
    n = len(X)
    if n < 3 * forecast_horizon:
        return None

    val_end   = n - forecast_horizon          # exclude the unknown current window
    val_start = val_end - forecast_horizon    # 1-lag-back window

    X_tune_train = X[:val_start]
    y_tune_train = y[:val_start]
    X_tune_val   = X[val_start:val_end]
    y_tune_val   = y[val_start:val_end]

    return X_tune_train, y_tune_train, X_tune_val, y_tune_val
