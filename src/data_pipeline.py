"""
Data Pipeline Module for Customer Churn
Contains Pydantic Schemas, Data Cleaning, Validation, and Feature Preprocessing.
"""
import os
import logging
from typing import Literal, Optional, Tuple, List
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict, TypeAdapter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
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


# Cached Pydantic TypeAdapter for batch validation performance
CUSTOMER_LIST_ADAPTER = TypeAdapter(List[CustomerInputSchema])


# ==========================================
# 2. Data Loading, Cleaning & Pydantic Validation
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

    # 3. Native Pydantic validation via TypeAdapter
    logger.info(f"Validating {validate_samples} records with Pydantic TypeAdapter...")
    sample_records = df.head(validate_samples).to_dict(orient="records")
    try:
        CUSTOMER_LIST_ADAPTER.validate_python(sample_records)
        logger.info(f"All {validate_samples} sampled records passed Pydantic validation successfully!")
    except ValidationError as e:
        logger.warning(f"Pydantic validation detected schema discrepancies: {e.errors()[:3]}")

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
