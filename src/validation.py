"""
Data Validation & Contract Schema Module
Contains unified Pydantic schemas for batch dataset validation and FastAPI request/response payloads.
"""
import logging
from typing import Literal, Optional, List
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, ConfigDict, TypeAdapter

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


def validate_dataframe(df: pd.DataFrame, sample_size: int = 500) -> None:
    """
    Validates sample records from a DataFrame against CustomerInputSchema using Pydantic TypeAdapter.
    """
    logger.info(f"Validating {sample_size} records with Pydantic TypeAdapter...")
    sample_records = df.head(sample_size).to_dict(orient="records")
    try:
        CUSTOMER_LIST_ADAPTER.validate_python(sample_records)
        logger.info(f"All {sample_size} sampled records passed Pydantic validation successfully!")
        return True
    except ValidationError as e:
        logger.warning(f"Pydantic validation detected schema discrepancies: {e.errors()[:3]}")
        raise e
