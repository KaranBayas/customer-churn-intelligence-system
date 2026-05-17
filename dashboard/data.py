import pandas as pd

from src.config import RAW_DATA_PATH
from src.preprocessing.clean_data import clean_data
from src.preprocessing.feature_engineering import engineer_features


def load_dashboard_data() -> pd.DataFrame:
    """Load the original customer dataset and prepare it for dashboard charts."""
    df = pd.read_csv(RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    df["Churn"] = df["Churn"].astype(int)
    return df
