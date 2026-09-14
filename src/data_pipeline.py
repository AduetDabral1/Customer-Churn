"""
Data Pipeline Module for Customer Churn
Contains Pydantic Schemas, Data Cleaning, Validation, and Feature Preprocessing.
"""
import os
import logging
from typing import Tuple
import pandas as pd
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.validation import (
    CustomerInputSchema,
    CustomerPredictionResponse,
    validate_dataframe
)

logger = logging.getLogger(__name__)

# Re-export schemas for backward compatibility
__all__ = [
    "CustomerInputSchema",
    "CustomerPredictionResponse",
    "validate_dataframe",
    "load_and_clean_data",
    "prepare_splits",
    "engineer_domain_features",
    "build_preprocessor"
]


# ==========================================
# 1. Data Loading, Cleaning & Validation
# ==========================================
def load_and_clean_data(file_path: str, validate_samples: int = 500) -> pd.DataFrame:
    """
    Loads raw customer churn CSV, parses TotalCharges, drops identifier column,
    and validates records against CustomerInputSchema using Pydantic TypeAdapter.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    logger.info(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")

    # 1. Clean TotalCharges (blank spaces to numeric, fill 0 where tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 2. Drop pure identifier column
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Pydantic validation via validation module
    try:
        validate_dataframe(df, sample_size=validate_samples)
    except ValidationError:
        logger.warning("Dataset had validation warnings, continuing with cleaned data.")

    return df


def prepare_splits(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates target y ('Churn') from features X, performs stratified 80/20 train/test split.
    """
    if "Churn" not in df.columns:
        raise ValueError("DataFrame does not contain 'Churn' target column.")

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Stratified split completed - Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train Churn Rate: {y_train.mean():.2%}, Test Churn Rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


# ==========================================
# 3. Native Feature Engineering & ColumnTransformer
# ==========================================
def engineer_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Stateless transformation function for scikit-learn's FunctionTransformer.
    Simplifies redundant categories and adds domain interaction features using vectorized pandas.
    """
    X = X.copy()

    # 1. Simplify redundant 'No internet service' and 'No phone service'
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ]
    existing_services = [c for c in service_cols if c in X.columns]
    if existing_services:
        X[existing_services] = X[existing_services].replace({
            "No internet service": "No",
            "No phone service": "No"
        })

    # 2. Add domain features: Total count of active services & contract commitment
    active_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    existing_active = [c for c in active_cols if c in X.columns]
    if existing_active:
        total_active = (X[existing_active] == "Yes").sum(axis=1)
        if "InternetService" in X.columns:
            total_active = total_active + (X["InternetService"] != "No").astype(int)
        X["TotalServices"] = total_active

    if "Contract" in X.columns:
        X["IsLongTermContract"] = (X["Contract"] != "Month-to-month").astype(int)

    return X


def build_preprocessor() -> Pipeline:
    """
    Builds the feature engineering and encoding/scaling pipeline using
    scikit-learn's native FunctionTransformer and ColumnTransformer.
    Eliminates custom class boilerplate and guarantees 100% standard serialization.
    """
    cat_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod"
    ]
    num_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", 
        "TotalServices", "IsLongTermContract", "SeniorCitizen"
    ]

    encoding_scaling_block = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    full_preprocessor = Pipeline(steps=[
        ("feature_engineering", FunctionTransformer(engineer_domain_features)),
        ("encoding_and_scaling", encoding_scaling_block)
    ])

    return full_preprocessor
