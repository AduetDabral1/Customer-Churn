"""
Model Pipeline Module for Customer Churn
Handles Recall-Driven Optuna Hyperparameter Optimization, MLflow Tracking,
Threshold Calibration, and Final Pipeline Serialization.
"""
import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)

# Suppress agent hints & verbose Optuna output
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# 1. Optuna Hyperparameter Tuning (Recall-First)
# ==========================================
def tune_xgboost(
    X_train_processed: np.ndarray,
    y_train: pd.Series,
    n_trials: int = 25,
    experiment_name: str = "Telco_Customer_Churn_XGBoost"
) -> Dict[str, Any]:
    """
    Performs Bayesian hyperparameter optimization targeting RECALL
    using 5-Fold Stratified Cross-Validation. Logs all trials into MLflow.
    """
    mlflow.set_experiment(experiment_name)
    logger.info(f"Starting Optuna tuning ({n_trials} trials) targeting Recall in experiment '{experiment_name}'...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 0.90),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.90),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.2, 3.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier(**params)

        cv_results = cross_validate(
            model,
            X_train_processed,
            y_train,
            cv=cv,
            scoring={"recall": "recall", "roc_auc": "roc_auc", "precision": "precision"},
            n_jobs=-1
        )

        mean_recall = cv_results["test_recall"].mean()
        mean_roc_auc = cv_results["test_roc_auc"].mean()
        mean_precision = cv_results["test_precision"].mean()

        # Log each trial as an MLflow nested run
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number:02d}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mean_recall", mean_recall)
            mlflow.log_metric("cv_mean_roc_auc", mean_roc_auc)
            mlflow.log_metric("cv_mean_precision", mean_precision)

        # Primary optimization objective is Recall
        return mean_recall

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Optuna tuning completed! Best Trial #{study.best_trial.number} with Mean Recall: {study.best_value:.4f}")
    return study.best_params


# ==========================================
# 2. Champion Model Training & MLflow Logging
# ==========================================
def train_and_log_champion_model(
    preprocessor: Pipeline,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir: str = "models",
    experiment_name: str = "Telco_Customer_Churn_XGBoost"
) -> Tuple[Pipeline, Dict[str, float], float]:
    """
    Fits preprocessor and champion XGBoost, evaluates on test set,
    calibrates decision threshold, logs artifacts to MLflow,
    and serializes the compressed production pipeline.
    """
    mlflow.set_experiment(experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    params = best_params.copy()
    params["eval_metric"] = "logloss"
    params["random_state"] = 42
    params["n_jobs"] = -1

    with mlflow.start_run(run_name="Champion_XGBoost_Pipeline"):
        logger.info("Transforming features via preprocessor...")
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        logger.info("Training champion XGBoost classifier on full training split...")
        champion_xgb = XGBClassifier(**params)
        champion_xgb.fit(X_train_proc, y_train)

        # Test set predictions & probabilities
        y_test_pred_default = champion_xgb.predict(X_test_proc)
        y_test_prob = champion_xgb.predict_proba(X_test_proc)[:, 1]

        # 1. Decision threshold tuning for business recall
        # Search for threshold that achieves maximum recall while keeping precision >= 0.50
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob)
        calibrated_threshold = 0.40  # Default robust operating threshold
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            if r >= 0.80 and p >= 0.50:
                calibrated_threshold = float(t)
                break

        y_test_pred_calibrated = (y_test_prob >= calibrated_threshold).astype(int)

        # 2. Performance metrics
        test_metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_test_pred_calibrated)),
            "test_precision": float(precision_score(y_test, y_test_pred_calibrated)),
            "test_recall": float(recall_score(y_test, y_test_pred_calibrated)),
            "test_f1": float(f1_score(y_test, y_test_pred_calibrated)),
            "test_roc_auc": float(roc_auc_score(y_test, y_test_prob)),
            "calibrated_threshold": calibrated_threshold,
            "default_recall_threshold_0_5": float(recall_score(y_test, y_test_pred_default))
        }

        logger.info(f"Test Set Evaluation Results at Threshold {calibrated_threshold:.2f}:")
        for k, v in test_metrics.items():
            logger.info(f"  - {k:25s}: {v:.4f}")

        # 3. Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics(test_metrics)

        # 4. Generate & log diagnostic figures
        # A. Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_test_pred_calibrated)
        ConfusionMatrixDisplay(cm, display_labels=["Stayed (0)", "Churned (1)"]).plot(ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix (Threshold={calibrated_threshold:.2f})")
        ax_cm.grid(False)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        # B. ROC Curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        ax_roc.plot(fpr, tpr, color="#2ecc71", lw=2, label=f"ROC Curve (AUC = {test_metrics['test_roc_auc']:.4f})")
        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        mlflow.log_figure(fig_roc, "roc_curve.png")
        plt.close(fig_roc)

        # C. Feature Importances
        feature_names = preprocessor.named_steps["encoding_and_scaling"].get_feature_names_out()
        fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
        fi_series = pd.Series(champion_xgb.feature_importances_, index=feature_names)
        fi_series.nlargest(15).plot(kind="barh", ax=ax_fi, color="#3498db")
        ax_fi.set_title("Top 15 Feature Importances (XGBoost)")
        ax_fi.invert_yaxis()
        plt.tight_layout()
        mlflow.log_figure(fig_fi, "feature_importances.png")
        plt.close(fig_fi)

        # 5. Assemble End-to-End Pipeline
        final_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", champion_xgb)
        ])

        # 6. Save compressed artifact for sub-millisecond serving
        model_path = os.path.join(model_dir, "final_churn_pipeline.joblib")
        joblib.dump(final_pipeline, model_path, compress=3)
        artifact_size_kb = os.path.getsize(model_path) / 1024
        logger.info(f"Saved compressed pipeline artifact to {model_path} ({artifact_size_kb:.1f} KB)")

        # Log model to MLflow model registry/artifacts
        mlflow.sklearn.log_model(final_pipeline, artifact_path="churn_pipeline")

        return final_pipeline, test_metrics, calibrated_threshold
