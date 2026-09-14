# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for XGBoost/LightGBM (OpenMP runtime) and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency specification first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy the application source code into the image
COPY . .

# Explicitly ensure model artifacts are copied (in case excluded by any rule)
COPY models/ /app/models/

# Environment configurations:
# - PYTHONUNBUFFERED=1: ensures logs are streamed in real-time without buffering
# - PYTHONPATH: allows importing modules using either `from src.validation...` or `from validation...`
# - MODEL_PATH & DECISION_THRESHOLD: configurable production defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src \
    MODEL_PATH=/app/models/final_churn_pipeline.joblib \
    DECISION_THRESHOLD=0.36

# 7. Expose FastAPI serving port
EXPOSE 8000

# 8. Container health check using the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 9. Run the FastAPI application using Uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
