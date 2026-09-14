"""
Unit & Integration Tests for FastAPI Serving Layer (src/app/main.py).
"""
import time
import pytest
from fastapi.testclient import TestClient
from src.app.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient triggering the FastAPI lifespan model loading."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    return {
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


def test_root_endpoint(client):
    """Assert root endpoint returns 200 and valid service links for JSON requests."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_ui_endpoint(client):
    """Assert /ui returns 200 OK and renders the HTML dashboard."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "RETENTION IQ" in response.text
    assert "Customer Risk Profiler" in response.text


def test_root_html_negotiation(client):
    """Assert / returns HTML dashboard when Accept header specifies text/html."""
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "RETENTION IQ" in response.text


def test_health_endpoint(client):
    """Assert health check returns 200, healthy status, and model loaded flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "active_threshold" in data


def test_predict_valid_customer(client, valid_customer_payload):
    """Assert valid payload returns 200 with structured churn prediction & risk tier."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "threshold_used" in data


def test_predict_invalid_tenure(client, valid_customer_payload):
    """Assert negative tenure triggers Pydantic 422 Unprocessable Entity."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["tenure"] = -10
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_invalid_contract_category(client, valid_customer_payload):
    """Assert unknown enum value triggers Pydantic 422 Unprocessable Entity."""
    invalid_payload = valid_customer_payload.copy()
    invalid_payload["Contract"] = "Infinite Plan"
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_latency(client, valid_customer_payload):
    """Assert steady-state API prediction latency is fast (< 50 ms)."""
    # Warmup
    _ = client.post("/predict", json=valid_customer_payload)

    t0 = time.perf_counter()
    response = client.post("/predict", json=valid_customer_payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert response.status_code == 200
    assert latency_ms < 100.0  # Asserts fast HTTP response time
