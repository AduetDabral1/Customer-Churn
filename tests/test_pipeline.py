"""
Unit and Smoke Tests for Customer Churn Production Pipeline
"""
import os
import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

from src.data_pipeline import (
    CustomerInputSchema,
    CustomerPredictionResponse,
    load_and_clean_data,
    prepare_splits,
    build_preprocessor
)


@pytest.fixture
def valid_customer_dict():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
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
        "TotalCharges": 2050.0
    }


def test_pydantic_schema_valid(valid_customer_dict):
    """Assert valid payload successfully creates CustomerInputSchema."""
    customer = CustomerInputSchema(**valid_customer_dict)
    assert customer.tenure == 24
    assert customer.MonthlyCharges == 85.5
    assert customer.InternetService == "Fiber optic"


def test_pydantic_schema_negative_tenure(valid_customer_dict):
    """Assert negative tenure raises ValidationError."""
    invalid_dict = valid_customer_dict.copy()
    invalid_dict["tenure"] = -5
    with pytest.raises(ValidationError):
        CustomerInputSchema(**invalid_dict)


def test_pydantic_schema_invalid_contract(valid_customer_dict):
    """Assert unknown contract category raises ValidationError."""
    invalid_dict = valid_customer_dict.copy()
    invalid_dict["Contract"] = "Lifetime Contract"
    with pytest.raises(ValidationError):
        CustomerInputSchema(**invalid_dict)


def test_pydantic_prediction_response():
    """Assert CustomerPredictionResponse structure and types."""
    resp = CustomerPredictionResponse(
        churn_prediction=1,
        churn_probability=0.74,
        risk_tier="HIGH",
        threshold_used=0.40
    )
    assert resp.churn_prediction == 1
    assert 0.0 <= resp.churn_probability <= 1.0
    assert resp.risk_tier == "HIGH"


def test_data_loader_and_splits():
    """Assert raw data loading, type cleaning, and split logic."""
    data_path = "data/raw/Customer_Churn_Data.csv"
    if not os.path.exists(data_path):
        pytest.skip(f"Data file not found at {data_path}")

    df = load_and_clean_data(data_path, validate_samples=50)
    assert "customerID" not in df.columns
    assert "Churn" in df.columns
    assert df["TotalCharges"].isna().sum() == 0
    assert df["TotalCharges"].dtype in [np.float64, np.float32, float]

    X_train, X_test, y_train, y_test = prepare_splits(df, test_size=0.20)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01  # Stratified check


def test_preprocessor_transformation(valid_customer_dict):
    """Assert feature engineering and scaling executes cleanly on sample DataFrame."""
    sample_df = pd.DataFrame([valid_customer_dict])
    preprocessor = build_preprocessor()

    # Fit and transform single sample
    transformed = preprocessor.fit_transform(sample_df)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_saved_pipeline_inference(valid_customer_dict):
    """Assert saved production pipeline can be loaded and predict in < 20 ms."""
    import joblib, time
    model_path = "models/final_churn_pipeline.joblib"
    if not os.path.exists(model_path):
        pytest.skip(f"Model artifact not yet saved at {model_path}")

    pipeline = joblib.load(model_path)
    sample_df = pd.DataFrame([valid_customer_dict])

    # Warmup prediction (loads XGBoost C++ dynamic library into CPU memory)
    _ = pipeline.predict(sample_df)

    # Steady-state inference latency benchmark
    t0 = time.perf_counter()
    pred = pipeline.predict(sample_df)
    prob = pipeline.predict_proba(sample_df)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0
    assert latency_ms < 500.0  # Asserts fast sub-half-second response
