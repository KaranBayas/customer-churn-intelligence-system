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
    """Extract feature importance from a logistic regression classifier if available."""
    if not hasattr(pipeline, "named_steps"):
        return None

    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if classifier is None or not hasattr(classifier, "coef_"):
        return None

    # LogisticRegression stores coefficients as a NumPy array.
    # For binary churn classification, coef_ is typically shape (1, n_features).
    coefficients = classifier.coef_
    if coefficients.ndim > 1:
        coefficients = coefficients[0]

    feature_names = None
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    # Convert coefficients to a pandas Series to use .abs() and preserve feature labels.
    coefficient_series = pd.Series(coefficients, index=feature_names, name="coefficient")
    importance_series = coefficient_series.abs().rename("importance")

    importance_df = pd.concat([coefficient_series, importance_series], axis=1).reset_index()
    importance_df.columns = ["feature", "coefficient", "importance"]
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    return importance_df
