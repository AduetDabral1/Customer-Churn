"""
Data Pipeline Module for Customer Churn
Contains Pydantic Schemas, Data Cleaning, Validation, and Feature Preprocessing.
"""
import os
import logging
from typing import Literal, Optional, Tuple, List
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ==========================================
# 1. Pydantic Schemas (Unified for Batch & API)
# ==========================================
class CustomerInputSchema(BaseModel):
    """
    Schema representing all customer attributes required for churn prediction.
    Used for both dataset validation and FastAPI request payload verification.
    """
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0.0, le=250.0, description="Monthly charges amount")
    TotalCharges: float = Field(ge=0.0, description="Total charges accumulated")
    Churn: Optional[Literal["Yes", "No"]] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1026.0
            }
        }
    )


class CustomerPredictionResponse(BaseModel):
    """Output schema returned by the model pipeline / FastAPI endpoint."""
    churn_prediction: int = Field(description="Binary prediction: 1 = At risk of churn, 0 = Stay")
    churn_probability: float = Field(description="Calibrated probability of churn (0.0 to 1.0)")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Business risk tier for retention action")
    threshold_used: float = Field(description="Decision threshold used to assign the prediction")


# ==========================================
# 2. Data Loading, Cleaning & Pydantic Validation
# ==========================================
def load_and_clean_data(file_path: str, validate_samples: int = 500) -> pd.DataFrame:
    """
    Loads raw customer churn CSV, parses TotalCharges, drops identifier column,
    and validates records against CustomerInputSchema.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    logger.info(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw dataset with shape: {df.shape}")

    # 1. Clean TotalCharges (blank spaces to numeric, fill 0 where tenure == 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().sum() > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 2. Drop pure identifier column
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Pydantic validation check on sample records
    logger.info(f"Validating {validate_samples} records with Pydantic CustomerInputSchema...")
    sample_records = df.head(validate_samples).to_dict(orient="records")
    validation_errors = 0

    for i, record in enumerate(sample_records):
        try:
            CustomerInputSchema(**record)
        except ValidationError as e:
            validation_errors += 1
            if validation_errors <= 3:
                logger.warning(f"Validation error in record {i}: {e.errors()}")

    if validation_errors > 0:
        logger.warning(f"Detected {validation_errors} validation discrepancies out of {validate_samples} records.")
    else:
        logger.info("All sampled records passed Pydantic validation successfully!")

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
# 3. Feature Engineering & ColumnTransformer
# ==========================================
class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies row-level domain features
    and simplifies redundant categories. Safe from data leakage.
    """
    def __init__(self):
        self.replace_no_internet = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Simplify redundant categories
        for col in self.replace_no_internet:
            if col in X.columns:
                X[col] = X[col].replace({"No internet service": "No"})

        if "MultipleLines" in X.columns:
            X["MultipleLines"] = X["MultipleLines"].replace({"No phone service": "No"})

        # 2. Add domain features
        if "PhoneService" in X.columns and "InternetService" in X.columns:
            X["TotalServices"] = (
                (X["PhoneService"] == "Yes").astype(int) +
                (X["MultipleLines"] == "Yes").astype(int) +
                (X["InternetService"] != "No").astype(int) +
                (X["OnlineSecurity"] == "Yes").astype(int) +
                (X["OnlineBackup"] == "Yes").astype(int) +
                (X["DeviceProtection"] == "Yes").astype(int) +
                (X["TechSupport"] == "Yes").astype(int) +
                (X["StreamingTV"] == "Yes").astype(int) +
                (X["StreamingMovies"] == "Yes").astype(int)
            )

        if "Contract" in X.columns:
            X["IsLongTermContract"] = (X["Contract"] != "Month-to-month").astype(int)

        return X


def build_preprocessor() -> Pipeline:
    """
    Builds the unified feature engineering and encoding/scaling pipeline.
    Output features are completely numeric and ready for XGBoost.
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
        ("feature_engineering", ChurnFeatureEngineer()),
        ("encoding_and_scaling", encoding_scaling_block)
    ])

    return full_preprocessor
