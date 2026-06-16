import joblib
import pandas as pd
from pathlib import Path

from src.config import PIPELINE_PATH
from src.preprocessing.clean_data import clean_data
from src.preprocessing.feature_engineering import engineer_features


def load_dashboard_model():
    """Load the trained churn prediction pipeline."""
    model_file = Path(PIPELINE_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"Missing model file: {model_file}")
    return joblib.load(model_file)


def predict_customer(pipeline, customer_data: dict) -> dict:
    """Predict churn label and probability for a single customer record."""
    payload = pd.DataFrame([customer_data])
    payload = clean_data(payload)
    payload = engineer_features(payload)

    if "Churn" in payload.columns:
        payload = payload.drop(columns=["Churn"], errors="ignore")

    prediction = pipeline.predict(payload)[0]
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(payload)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability) if probability is not None else None,
    }


def risk_label(probability: float | None) -> str:
    """Convert probability into a risk label for presentation."""
    if probability is None:
        return "Unavailable"
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def get_feature_importance(pipeline) -> pd.DataFrame | None:
    """Extract feature importance from a fitted pipeline classifier."""
    if not hasattr(pipeline, "named_steps"):
        return None

    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")
    if classifier is None:
        return None

    importance = None
    coefficient_values = None

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        coefficient_values = classifier.coef_
        if coefficient_values.ndim > 1:
            coefficient_values = coefficient_values[0]
        importance = abs(coefficient_values)

    if importance is None:
        return None

    feature_names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    if len(feature_names) != len(importance):
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    if coefficient_values is not None:
        importance_df["coefficient"] = coefficient_values

    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return importance_df
