"""
models.py
---------
Model training, evaluation, and stacking utilities for the
India Cricket Analytics capstone (RQ1-RQ5).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, log_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "auc_roc", "log_loss"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_prob: np.ndarray) -> dict:
    """Return standard binary classification metrics as a dict."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_true, y_prob), 4),
        "log_loss":  round(log_loss(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------------------
# Baseline logistic regression (all RQs)
# ---------------------------------------------------------------------------
def build_logistic(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """L2-regularised logistic regression with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                   max_iter=1000, random_state=random_state)),
    ])


# ---------------------------------------------------------------------------
# Random Forest (RQ3)
# ---------------------------------------------------------------------------
def build_random_forest(n_estimators: int = 100,
                         random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# XGBoost with DART (RQ3)
# ---------------------------------------------------------------------------
def build_xgboost(params: dict | None = None,
                  random_state: int = 42):
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    default_params = {
        "n_estimators": 200,
        "booster": "dart",
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": random_state,
    }
    if params:
        default_params.update(params)
    return xgb.XGBClassifier(**default_params)


# ---------------------------------------------------------------------------
# Stacking meta-classifier (RQ3)
# ---------------------------------------------------------------------------
def build_stacking(random_state: int = 42) -> StackingClassifier:
    """
    Stacking classifier combining logistic regression, Random Forest,
    and XGBoost base learners with a logistic regression meta-learner.
    """
    estimators = [
        ("lr",  build_logistic(random_state=random_state)),
        ("rf",  build_random_forest(random_state=random_state)),
    ]
    if HAS_XGB:
        estimators.append(("xgb", build_xgboost(random_state=random_state)))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                           random_state=random_state),
        cv=5,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------
def run_cv(model, X: np.ndarray, y: np.ndarray,
           n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation and return per-fold metrics.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    results = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, y_pred, y_prob)
        metrics["fold"] = fold
        results.append(metrics)
    df = pd.DataFrame(results).set_index("fold")
    df.loc["mean"] = df.mean()
    return df


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimisation (RQ3)
# ---------------------------------------------------------------------------
def optimise_xgboost(X: np.ndarray, y: np.ndarray,
                     n_trials: int = 50,
                     random_state: int = 42) -> dict:
    """
    Bayesian hyperparameter optimisation for XGBoost via Optuna.
    Returns the best parameter dictionary.
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna is not installed. Run: pip install optuna")
    if not HAS_XGB:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "booster":          "dart",
            "eval_metric":      "logloss",
            "random_state":     random_state,
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[val_idx])[:, 1]
            scores.append(roc_auc_score(y[val_idx], prob))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params
