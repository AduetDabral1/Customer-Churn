"""
Master Pipeline Orchestrator for Customer Churn
Runs Data Loading -> Pydantic Validation -> Preprocessing -> Optuna (Recall) -> MLflow Tracking -> Final Pipeline Export
"""
import os
import sys
import time
import argparse
import logging
import pandas as pd

from src.data_pipeline import load_and_clean_data, prepare_splits, build_preprocessor
from src.model_pipeline import tune_xgboost, train_and_log_champion_model

# Configure console and file logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("run_pipeline")


def run(
    data_path: str = "data/raw/Customer_Churn_Data.csv",
    n_trials: int = 20,
    experiment_name: str = "Telco_Customer_Churn_XGBoost",
    model_dir: str = "models"
):
    start_time = time.time()
    logger.info("=" * 65)
    logger.info("STARTING CUSTOMER CHURN PRODUCTION TRAINING PIPELINE")
    logger.info(f"Targeting Metric: RECALL | Optimizer: Optuna | Tracking: MLflow")
    logger.info("=" * 65)

    # 1. Load, clean, and validate data with Pydantic
    logger.info("[Step 1/5] Ingesting and validating raw data with Pydantic...")
    df = load_and_clean_data(data_path, validate_samples=500)

    # 2. Stratified train/test split
    logger.info("[Step 2/5] Creating stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test = prepare_splits(df, test_size=0.20, random_state=42)

    # 3. Assemble and fit preprocessor
    logger.info("[Step 3/5] Building unified ColumnTransformer preprocessor...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)

    # 4. Optuna hyperparameter tuning targeting Recall
    logger.info(f"[Step 4/5] Running Bayesian hyperparameter tuning ({n_trials} trials)...")
    best_params = tune_xgboost(
        X_train_proc, 
        y_train, 
        n_trials=n_trials, 
        experiment_name=experiment_name
    )

    # 5. Train champion model, log to MLflow, and export compressed pipeline
    logger.info("[Step 5/5] Training Champion Pipeline & logging artifacts to MLflow...")
    final_pipeline, test_metrics, threshold = train_and_log_champion_model(
        preprocessor=preprocessor,
        best_params=best_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_dir=model_dir,
        experiment_name=experiment_name
    )

    # 6. Benchmark latency & artifact size
    model_path = os.path.join(model_dir, "final_churn_pipeline.joblib")
    artifact_size_kb = os.path.getsize(model_path) / 1024

    sample_input = X_test.head(1)
    t0 = time.perf_counter()
    _ = final_pipeline.predict_proba(sample_input)
    latency_ms = (time.perf_counter() - t0) * 1000

    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.1f}s!")
    logger.info(f"Model Artifact Size : {artifact_size_kb:.1f} KB")
    logger.info(f"Single-Sample Latency: {latency_ms:.2f} ms (< 1 ms serving target)")
    logger.info(f"Test Recall Score    : {test_metrics['test_recall']:.2%}")
    logger.info(f"Test ROC-AUC Score   : {test_metrics['test_roc_auc']:.4f}")
    logger.info(f"Calibrated Threshold : {threshold:.2f}")
    logger.info(f"Exported Model Path  : {model_path}")
    logger.info("To view MLflow dashboard, run: 'uv run mlflow ui'")
    logger.info("=" * 65)

    return final_pipeline, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Training & Tracking Pipeline")
    parser.add_argument("--data-path", type=str, default="data/raw/Customer_Churn_Data.csv", help="Path to raw CSV")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna tuning trials")
    parser.add_argument("--experiment-name", type=str, default="Telco_Customer_Churn_XGBoost", help="MLflow experiment name")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save final model artifact")

    args = parser.parse_args()
    run(
        data_path=args.data_path,
        n_trials=args.trials,
        experiment_name=args.experiment_name,
        model_dir=args.model_dir
    )
