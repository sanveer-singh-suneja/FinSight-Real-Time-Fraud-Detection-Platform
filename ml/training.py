"""
FinSight ML Training Pipeline.
Trains multiple classifiers, optimises the best one with Optuna,
evaluates on time-split validation set, and persists all artefacts.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import structlog
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = structlog.get_logger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute standard classification metrics at optimal F1 threshold."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0.0
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    y_pred_best = (y_prob >= best_thresh).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(best_f1),
        "precision": float(precision_score(y_true, y_pred_best, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_best, zero_division=0)),
        "optimal_threshold": float(best_thresh),
    }


def _make_base_models() -> dict[str, Any]:
    """Instantiate all candidate models with sensible defaults."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if HAS_LIGHTGBM:
        models["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if HAS_XGBOOST:
        models["xgboost"] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=10,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )
    return models


def _optuna_objective_xgb(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """XGBoost Optuna objective – maximises PR-AUC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50),
    }
    if not HAS_XGBOOST:
        return 0.0
    model = xgb.XGBClassifier(
        **params, eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, y_prob))


def _optuna_objective_lgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """LightGBM Optuna objective – maximises PR-AUC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "class_weight": "balanced",
    }
    if not HAS_LIGHTGBM:
        return 0.0
    model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, y_prob))


def optimise_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> Any:
    """Run Optuna hyperparameter search for the best model."""
    logger.info("optuna_optimisation_start", model=model_name, n_trials=n_trials)

    if model_name == "xgboost" and HAS_XGBOOST:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _optuna_objective_xgb(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        optimised = xgb.XGBClassifier(
            **best_params, eval_metric="aucpr", random_state=42, n_jobs=-1, verbosity=0
        )
    elif model_name == "lightgbm" and HAS_LIGHTGBM:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _optuna_objective_lgbm(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        optimised = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        logger.warning("no_optuna_support_for_model", model=model_name)
        return _make_base_models().get(model_name)

    optimised.fit(X_train, y_train)
    y_prob = optimised.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, y_prob)
    logger.info("optuna_optimisation_complete", model=model_name, best_pr_auc=pr_auc)
    return optimised


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE oversampling to address class imbalance."""
    fraud_rate = float(y_train.mean())
    if fraud_rate > 0.3:
        logger.info("smote_skipped", reason="already_balanced", fraud_rate=fraud_rate)
        return X_train, y_train

    logger.info("smote_start", fraud_rate=fraud_rate)
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info("smote_complete", new_fraud_rate=float(y_res.mean()), new_size=len(X_res))
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_smote: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """
    Train all candidate models and collect validation metrics.

    Returns:
        (models_dict, metrics_dict) where keys are model names.
    """
    if use_smote:
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    base_models = _make_base_models()
    trained: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    for name, model in base_models.items():
        logger.info("model_training_start", model=name)
        try:
            if name in ("logistic_regression",):
                pipe = Pipeline(
                    [("scaler", StandardScaler()), ("clf", model)]
                )
                pipe.fit(X_train_bal, y_train_bal)
                y_prob = pipe.predict_proba(X_val)[:, 1]
                trained[name] = pipe
            else:
                model.fit(X_train_bal, y_train_bal)
                y_prob = model.predict_proba(X_val)[:, 1]
                trained[name] = model

            m = _compute_metrics(y_val.values, y_prob)
            metrics[name] = m
            logger.info("model_training_complete", model=name, **{k: round(v, 4) for k, v in m.items()})
        except Exception as exc:
            logger.error("model_training_failed", model=name, error=str(exc))

    return trained, metrics


def select_best_model(metrics: dict[str, dict[str, float]]) -> str:
    """Select the model with highest PR-AUC."""
    best_name = max(metrics, key=lambda n: metrics[n].get("pr_auc", 0))
    logger.info(
        "best_model_selected",
        model=best_name,
        pr_auc=round(metrics[best_name]["pr_auc"], 4),
    )
    return best_name


def calibrate_model(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """Wrap model with Platt scaling calibration."""
    logger.info("calibrating_model")
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def log_to_mlflow(
    run_name: str,
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    feature_cols: list[str],
    artifacts_dir: Path,
    tracking_uri: str,
    experiment_name: str,
) -> str:
    """Log model, metrics, and artefacts to MLflow. Returns run_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("feature_list", json.dumps(feature_cols[:50]))  # truncate for UI

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="finsight-fraud-detector",
        )

        # Log artefact files
        for path in artifacts_dir.rglob("*"):
            if path.is_file():
                mlflow.log_artifact(str(path))

        run_id = run.info.run_id
        logger.info("mlflow_run_logged", run_id=run_id)
        return run_id


def persist_model_artifacts(
    model: Any,
    scaler: Any,
    feature_cols: list[str],
    metrics: dict[str, float],
    optimal_threshold: float,
    model_name: str,
    output_dir: Path,
) -> dict[str, str]:
    """Save all artefacts required for inference to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "model": str(output_dir / "model.joblib"),
        "scaler": str(output_dir / "scaler.joblib"),
        "feature_cols": str(output_dir / "feature_cols.json"),
        "metadata": str(output_dir / "metadata.json"),
    }

    joblib.dump(model, paths["model"])
    if scaler is not None:
        joblib.dump(scaler, paths["scaler"])
    with open(paths["feature_cols"], "w") as f:
        json.dump(feature_cols, f)
    with open(paths["metadata"], "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "optimal_threshold": optimal_threshold,
                "metrics": metrics,
                "version": "1.0.0",
            },
            f,
            indent=2,
        )

    logger.info("model_artifacts_saved", paths=paths)
    return paths
