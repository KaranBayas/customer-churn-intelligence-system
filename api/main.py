from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
)
from api.model_utils import load_pipeline
from api.predictor import predict_from_pipeline
from src.config import PIPELINE_PATH

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="A simple FastAPI service for churn prediction using a trained customer churn model.",
    version="0.1.0",
)


def get_pipeline():
    """Load and cache the trained pipeline for inference."""
    return load_pipeline(PIPELINE_PATH)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Return a simple health check response."""
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    """Predict churn and return a clean JSON response."""
    try:
        pipeline = get_pipeline()
        prediction = predict_from_pipeline(pipeline, request.dict())
        return ChurnPredictionResponse(**prediction)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed.")
