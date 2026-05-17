from typing import Dict, Any
import pandas as pd

from src.preprocessing.preprocessing_pipeline import clean_data, engineer_features


def prepare_input(payload: Dict[str, Any]) -> pd.DataFrame:
    """Convert JSON payload to a cleaned and feature-engineered DataFrame."""
    raw_df = pd.DataFrame([payload])
    cleaned = clean_data(raw_df)
    engineered = engineer_features(cleaned)
    return engineered


def predict_from_pipeline(pipeline, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a prediction and return a simple JSON-ready response."""
    X = prepare_input(payload)
    prediction = pipeline.predict(X)
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(X)[:, 1].tolist()[0]

    return {
        "churn_prediction": int(prediction.tolist()[0]),
        "churn_probability": float(probability) if probability is not None else None,
    }
