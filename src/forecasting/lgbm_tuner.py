"""
Adaptive LGBM Hyperparameter Tuner

Uses the "1-lag-back" window strategy: for a forecast horizon of N days,
the N-day window immediately preceding the current (unknown) window is used
as the validation set to find the best LightGBM hyperparameters.

Speed design — HalvingRandomSearchCV
-------------------------------------
``HalvingRandomSearchCV`` (sklearn ≥ 0.24) implements Successive Halving:

    Round 0:  all n_candidates evaluated with min_resources trees  (cheap)
    Round 1:  top 1/factor survive, evaluated with factor×more trees
    Round k:  repeat until max_resources is reached

Compared to plain random search this gives:
- The same or lower total tree-build count as ``n_iter`` random candidates
  evaluated at max budget, because losers are eliminated early.
- Better winner quality: the champion is re-evaluated at progressively larger
  budgets, reducing noise from small-sample estimates.

The fixed ``PredefinedSplit`` ensures our hand-crafted 1-lag-back split is
used verbatim instead of rolling k-fold (no extra CV overhead).

Final model budget
------------------
After tuning, the winning structural params are applied to a fresh
LGBMRegressor trained with ``_FIXED_FINAL`` (n_estimators=100).  Reducing
from 200 → 100 trees halves the dominant cost: the 48 final-model fits run
for every backtest window, dwarfing the 10 tuning windows.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, PredefinedSplit

logger = logging.getLogger(__name__)

# ── Search space ─────────────────────────────────────────────────────────────
PARAM_GRID: dict[str, list] = {
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.03, 0.05, 0.1, 0.15],
    "num_leaves":       [15, 31, 63],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# Shared fixed settings (no n_estimators — controlled separately below)
_SHARED = {
    "reg_alpha":    0.1,
    "reg_lambda":   0.1,
    "random_state": 42,
    "verbose":      -1,
}

# HalvingRandomSearchCV controls n_estimators via resource= during search.
# The base estimator just needs a sensible fallback; it is always overridden.
_BASE_N_ESTIMATORS = 50

# Winner gets the full budget for the final model trained on the whole window.
# 100 is the sweet spot: fast enough that 48 window fits are cheap, yet
# accurate enough for daily price forecasting.
_FIXED_FINAL: dict = {**_SHARED, "n_estimators": 100}

# Halving config: min trees per candidate → factor growth → cap at max
_HALVING_MIN_RESOURCES: int = 10   # cheap first screen
_HALVING_MAX_RESOURCES: int = 50   # winner evaluated up to this many trees
_HALVING_FACTOR: int = 3           # keep top-1/3 each round

# Fallback used when tuning is disabled or data is insufficient
DEFAULT_PARAMS: dict = {
    **_FIXED_FINAL,
    "max_depth": 5,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


def tune_lgbm_params(
    X_tune_train: np.ndarray,
    y_tune_train: np.ndarray,
    X_tune_val: np.ndarray,
    y_tune_val: np.ndarray,
    n_iter: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Successive-halving search over PARAM_GRID on the 1-lag-back validation set.

    Starts with ``n_iter * 3`` candidates evaluated at ``_HALVING_MIN_RESOURCES``
    trees.  Each round keeps only the top-1/``_HALVING_FACTOR`` survivors,
    re-evaluating them with proportionally more trees.  The champion at
    ``_HALVING_MAX_RESOURCES`` trees wins.

    The winner's structural params are returned merged with ``_FIXED_FINAL``
    (n_estimators=100), so the final model trained on the full window is
    always grown to 100 trees regardless of the search budget.

    Parameters
    ----------
    X_tune_train : ndarray, shape (n_tune_train, n_features)
    y_tune_train : ndarray, shape (n_tune_train,)
    X_tune_val   : ndarray, shape (forecast_horizon, n_features)
    y_tune_val   : ndarray, shape (forecast_horizon,)
    n_iter       : Seed pool size is ``n_iter * 3`` (default 10 → 30 candidates).
    random_state : Seed for reproducibility.

    Returns
    -------
    dict
        Best params ready for ``lgb.LGBMRegressor(**params)``.
        Falls back to DEFAULT_PARAMS when data is too small.
    """
    if len(X_tune_train) < 20 or len(X_tune_val) < 1:
        logger.debug(
            "Tuning skipped: insufficient samples (train=%d, val=%d)",
            len(X_tune_train), len(X_tune_val),
        )
        return DEFAULT_PARAMS.copy()

    # ── Build a single fixed train/val split ─────────────────────────────────
    # PredefinedSplit: -1 → always in training fold, 0 → validation fold.
    # This reuses our hand-crafted 1-lag-back split verbatim, with zero extra
    # CV overhead.
    n_train = len(X_tune_train)
    X_all = np.vstack([X_tune_train, X_tune_val])
    y_all = np.concatenate([y_tune_train, y_tune_val])
    test_fold = np.full(len(X_all), -1)
    test_fold[n_train:] = 0
    cv = PredefinedSplit(test_fold)

    base = lgb.LGBMRegressor(**_SHARED, n_estimators=_BASE_N_ESTIMATORS)

    # Start with 3× the n_iter seed pool so halving can eliminate aggressively.
    # Tree-build budget with n_iter=10 → n_candidates=30, factor=3:
    #   Round 0: 30 × 10 = 300   Round 1: 10 × 30 = 300   Round 2: 3 × 50 = 150
    #   Total: ~750  vs.  old manual loop 10 × 200 = 2,000 → ~2.7× faster
    n_candidates = n_iter * 3

    search = HalvingRandomSearchCV(
        base,
        param_distributions=PARAM_GRID,
        resource="n_estimators",
        min_resources=_HALVING_MIN_RESOURCES,
        max_resources=_HALVING_MAX_RESOURCES,
        factor=_HALVING_FACTOR,
        n_candidates=n_candidates,
        cv=cv,
        scoring="neg_mean_absolute_percentage_error",
        refit=False,     # we refit manually with _FIXED_FINAL below
        random_state=random_state,
        n_jobs=1,        # joblib multiprocessing is unsafe inside Streamlit
        verbose=0,
        error_score=np.nan,
    )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Scoring failed",
                category=UserWarning,
            )
            search.fit(X_all, y_all)

        # Extract only the structural hyperparams (n_estimators comes from
        # _FIXED_FINAL, not from the search result).
        best_structural = {
            k: search.best_params_[k]
            for k in PARAM_GRID
            if k in search.best_params_
        }
        best_params = {**best_structural, **_FIXED_FINAL}
        logger.debug(
            "Halving search done — best params=%s", best_params
        )
        return best_params

    except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
        logger.warning("HalvingRandomSearchCV failed (%s), using defaults", exc)
        return DEFAULT_PARAMS.copy()


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

    val_end   = n - forecast_horizon       # exclude the unknown current window
    val_start = val_end - forecast_horizon  # 1-lag-back window

    return X[:val_start], y[:val_start], X[val_start:val_end], y[val_start:val_end]
