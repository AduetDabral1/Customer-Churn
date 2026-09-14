"""
FastAPI Serving Application for Customer Churn Prediction.
Provides /predict and /health endpoints with sub-millisecond inference and Pydantic validation.
"""
import os
import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from src.validation import CustomerInputSchema, CustomerPredictionResponse

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/final_churn_pipeline.joblib")
DEFAULT_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.36"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading and warming up the ML model on startup.
    Ensures zero cold-start latency on the first incoming user request.
    """
    logger.info(f"Loading production pipeline from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        # Fallback to alternate model if final artifact not yet generated
        fallback_path = "models/best_model_pipeline.joblib"
        if os.path.exists(fallback_path):
            logger.warning(f"Primary model not found at '{MODEL_PATH}', falling back to '{fallback_path}'.")
            app.state.model = joblib.load(fallback_path)
        else:
            raise FileNotFoundError(f"No trained model artifact found at '{MODEL_PATH}' or '{fallback_path}'.")
    else:
        app.state.model = joblib.load(MODEL_PATH)

    app.state.threshold = DEFAULT_THRESHOLD

    # Warmup prediction: loads dynamic C++ XGBoost libraries into memory cache
    warmup_sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5, "TotalCharges": 1026.0
    }
    _ = app.state.model.predict_proba(pd.DataFrame([warmup_sample]))
    logger.info(f"Model successfully loaded and warmed up! Active threshold: {app.state.threshold:.2f}")

    yield

    logger.info("Shutting down churn prediction serving service...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="High-performance, recall-optimized Telco Customer Churn inference service.",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files directory for frontend UI
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["General"])
async def root(request: Request):
    """
    Serves interactive HTML Web UI to browsers, or JSON metadata to API clients.
    """
    accept = request.headers.get("accept", "")
    index_file = os.path.join(STATIC_DIR, "index.html")
    if "text/html" in accept and os.path.exists(index_file):
        return FileResponse(index_file)

    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "web_ui": "/ui",
        "health_check": "/health"
    }


@app.get("/ui", tags=["General"])
@app.get("/dashboard", tags=["General"])
async def serve_ui():
    """Customer Churn Interactive Web UI."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="UI frontend template not found.")
    return FileResponse(index_file)


@app.get("/health", tags=["General"])
async def health_check():
    """Operational health check endpoint for container orchestrators (ECS, Docker, K8s)."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    if not model_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": "Model artifact not loaded"}
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_threshold": app.state.threshold
    }


@app.post(
    "/predict",
    response_model=CustomerPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_churn(payload: CustomerInputSchema) -> CustomerPredictionResponse:
    """
    Predicts customer churn probability and risk tier for retention action.
    
    - **churn_prediction**: Binary flag (1 = At risk of churn, 0 = Stay)
    - **churn_probability**: Calibrated churn likelihood (0.0 to 1.0)
    - **risk_tier**: Business risk category (HIGH, MEDIUM, LOW)
    - **threshold_used**: Decision threshold used to assign the prediction
    """
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for serving."
        )

    try:
        # 1. Convert validated Pydantic payload to single-row DataFrame
        input_data = payload.model_dump()
        input_df = pd.DataFrame([input_data])

        # 2. Run model pipeline inference (preprocessing + XGBoost classification)
        prob = float(app.state.model.predict_proba(input_df)[0, 1])
        threshold = app.state.threshold

        # 3. Apply calibrated operating threshold for business recall
        churn_pred = int(prob >= threshold)

        # 4. Determine actionable customer risk tier
        if prob >= 0.70:
            risk_tier = "HIGH"
        elif prob >= threshold:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        return CustomerPredictionResponse(
            churn_prediction=churn_pred,
            churn_probability=round(prob, 4),
            risk_tier=risk_tier,
            threshold_used=threshold
        )
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
