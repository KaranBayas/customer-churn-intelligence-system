from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer

from .clean_data import clean_data
from .feature_engineering import engineer_features
from .data_encoding import build_preprocessor
from .split_data import split_data
from ..config import PROCESSED_DATA_DIR


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_data(df)
    df = engineer_features(df)
    return df


def create_preprocessor(X) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    return preprocessor


def split_and_save_data(df: pd.DataFrame, target_column: str = "Churn", test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test
