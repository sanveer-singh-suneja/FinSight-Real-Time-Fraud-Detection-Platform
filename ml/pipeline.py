"""
FinSight ML Pipeline Orchestrator.
Runs the complete training pipeline end-to-end:
  1. Load & engineer features
  2. Time-based train/val split
  3. Train all models
  4. Optimise best model
  5. Calibrate
  6. Evaluate & explain
  7. Persist artefacts
  8. Log to MLflow
"""
from __future__ import annotations

import sys
from pathlib import Path

import structlog

# Configure structlog early
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

logger = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.evaluation import generate_evaluation_report
from ml.explainability import generate_global_explanations
from ml.feature_engineering import load_and_engineer, run_eda, time_based_split
from ml.training import (
    calibrate_model,
    log_to_mlflow,
    optimise_best_model,
    persist_model_artifacts,
    select_best_model,
    train_all_models,
)


def run_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
    mlflow_uri: str = "http://mlflow:5000",
    mlflow_experiment: str = "fraud-detection",
    n_optuna_trials: int = 30,
    run_optuna: bool = True,
    skip_eda: bool = False,
) -> dict:
    """
    Execute the full ML pipeline and return a summary dictionary.
    """
    data_dir = data_dir or (PROJECT_ROOT / "data" / "raw")
    output_dir = output_dir or (PROJECT_ROOT / "models")
    reports_dir = reports_dir or (PROJECT_ROOT / "reports")

    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "validation").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "artifacts" / "plots").mkdir(parents=True, exist_ok=True)

    logger.info("pipeline_start", data_dir=str(data_dir))

    # ── Step 1: Load & engineer features ───────────────────────────────────
    logger.info("step_1_feature_engineering")
    X, y, feature_cols = load_and_engineer(data_dir)

    if not skip_eda:
        logger.info("running_eda")
        import pandas as pd
        raw_df = pd.read_csv(data_dir / "train_transaction.csv")
        if "isFraud" in raw_df.columns:
            run_eda(raw_df, reports_dir / "eda")

    # ── Step 2: Time-based split ────────────────────────────────────────────
    logger.info("step_2_train_val_split")
    X_train, X_val, y_train, y_val = time_based_split(X, y, validation_fraction=0.2)

    # Save processed datasets
    X_train.to_parquet(PROJECT_ROOT / "data" / "processed" / "X_train.parquet", index=False)
    X_val.to_parquet(PROJECT_ROOT / "data" / "validation" / "X_val.parquet", index=False)
    y_train.to_frame("isFraud").to_parquet(
        PROJECT_ROOT / "data" / "processed" / "y_train.parquet", index=False
    )
    y_val.to_frame("isFraud").to_parquet(
        PROJECT_ROOT / "data" / "validation" / "y_val.parquet", index=False
    )
    logger.info("processed_data_saved")

    # ── Step 3: Train all models ────────────────────────────────────────────
    logger.info("step_3_model_training")
    trained_models, all_metrics = train_all_models(X_train, y_train, X_val, y_val)

    # ── Step 4: Select best & optimise ─────────────────────────────────────
    logger.info("step_4_model_selection")
    best_name = select_best_model(all_metrics)

    if run_optuna and best_name in ("xgboost", "lightgbm"):
        logger.info("step_4b_optuna_optimisation", model=best_name)
        optimised = optimise_best_model(
            best_name, X_train, y_train, X_val, y_val, n_trials=n_optuna_trials
        )
        if optimised is not None:
            trained_models[best_name] = optimised
            import numpy as np
            from ml.training import _compute_metrics
            y_prob_opt = optimised.predict_proba(X_val)[:, 1]
            all_metrics[best_name] = _compute_metrics(y_val.values, y_prob_opt)
            logger.info("post_optuna_metrics", **{k: round(v, 4) for k, v in all_metrics[best_name].items()})

    best_model = trained_models[best_name]
    best_metrics = all_metrics[best_name]
    threshold = best_metrics.get("optimal_threshold", 0.5)

    # ── Step 5: Calibrate ───────────────────────────────────────────────────
    logger.info("step_5_calibration")
    best_model = calibrate_model(best_model, X_val, y_val)

    # ── Step 6: Final evaluation ────────────────────────────────────────────
    logger.info("step_6_evaluation")
    import numpy as np
    y_prob_final = best_model.predict_proba(X_val)[:, 1]
    eval_report = generate_evaluation_report(
        y_true=y_val.values,
        y_prob=y_prob_final,
        model=best_model,
        model_name=best_name,
        feature_cols=feature_cols,
        threshold=threshold,
        output_dir=reports_dir / "evaluation",
        all_metrics=all_metrics,
    )

    # ── Step 7: SHAP explanations ───────────────────────────────────────────
    logger.info("step_7_shap_explanations")
    shap_results = generate_global_explanations(
        model=best_model,
        X_val=X_val,
        output_dir=PROJECT_ROOT / "artifacts" / "plots" / "shap",
    )

    # ── Step 8: Persist model artefacts ────────────────────────────────────
    logger.info("step_8_persist_artefacts")
    artifact_paths = persist_model_artifacts(
        model=best_model,
        scaler=None,  # scaler embedded in pipeline if needed
        feature_cols=feature_cols,
        metrics=best_metrics,
        optimal_threshold=threshold,
        model_name=best_name,
        output_dir=output_dir,
    )

    # ── Step 9: Log to MLflow ───────────────────────────────────────────────
    logger.info("step_9_mlflow_logging")
    try:
        run_id = log_to_mlflow(
            run_name=f"finsight-{best_name}",
            model=best_model,
            metrics=best_metrics,
            params={
                "model_name": best_name,
                "n_features": len(feature_cols),
                "threshold": threshold,
                "n_optuna_trials": n_optuna_trials if run_optuna else 0,
            },
            feature_cols=feature_cols,
            artifacts_dir=PROJECT_ROOT / "artifacts",
            tracking_uri=mlflow_uri,
            experiment_name=mlflow_experiment,
        )
    except Exception as exc:
        logger.warning("mlflow_logging_failed", error=str(exc))
        run_id = "local"

    summary = {
        "status": "success",
        "best_model": best_name,
        "metrics": best_metrics,
        "threshold": threshold,
        "n_features": len(feature_cols),
        "mlflow_run_id": run_id,
        "artifact_paths": artifact_paths,
        "shap_global_importance": shap_results.get("global_importance", {}),
    }

    logger.info("pipeline_complete", **{k: v for k, v in summary.items() if k not in ("artifact_paths", "shap_global_importance")})
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FinSight ML Training Pipeline")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mlflow-uri", default="http://mlflow:5000")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        data_dir=args.data_dir,
        mlflow_uri=args.mlflow_uri,
        n_optuna_trials=args.n_trials,
        run_optuna=not args.no_optuna,
        skip_eda=args.skip_eda,
    )
    import json
    print(json.dumps({k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v
                      for k, v in result.items()}, indent=2))
