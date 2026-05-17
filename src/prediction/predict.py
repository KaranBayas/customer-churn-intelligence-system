import joblib
import pandas as pd
from pathlib import Path
from typing import Union


def load_model(model_path: Union[str, Path]):
    """Load a saved sklearn pipeline or model from disk."""
    return joblib.load(Path(model_path))


def predict_batch(model, X_new):
    """Make predictions for a batch or table of records."""
    if isinstance(X_new, dict):
        X_new = pd.DataFrame([X_new])
    elif not hasattr(X_new, "shape"):
        X_new = pd.DataFrame(X_new)

    return model.predict(X_new)


def predict_proba(model, X_new):
    """Compute probability predictions for a batch of records."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support probability predictions.")

    if isinstance(X_new, dict):
        X_new = pd.DataFrame([X_new])
    elif not hasattr(X_new, "shape"):
        X_new = pd.DataFrame(X_new)

    return model.predict_proba(X_new)


def predict_single(model, record: dict):
    """Predict churn for a single customer dictionary."""
    X_new = pd.DataFrame([record])
    return predict_batch(model, X_new)
