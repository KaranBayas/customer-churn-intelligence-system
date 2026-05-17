import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data and prepare it for feature engineering."""
    df = df.copy()
    df = df.drop(columns=["customerID", "customerId"], errors="ignore")

    df = df.replace(" ", pd.NA)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").astype("Int64")

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype("Int64")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    df = df.ffill().bfill()

    return df
