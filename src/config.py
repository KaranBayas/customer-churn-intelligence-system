from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR.parent / "model"
PIPELINE_FILENAME = "customer_churn_pipeline.pkl"
PIPELINE_PATH = MODEL_DIR / PIPELINE_FILENAME
