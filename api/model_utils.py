import joblib
from pathlib import Path


def load_pipeline(model_path: str | Path):
    """Load a trained sklearn pipeline from disk."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    return joblib.load(model_file)
