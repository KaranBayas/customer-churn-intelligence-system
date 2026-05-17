import pandas as pd

from src.preprocessing.clean_data import clean_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.split_data import split_data


def test_clean_data_and_feature_engineering():
    df = pd.DataFrame(
        {
            "customerID": ["A001", "A002"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 5],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["No", "No"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "Contract": ["Month-to-month", "One year"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["29.85", "284.75"],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned = clean_data(df)
    assert cleaned.shape[0] == 2
    assert cleaned["customerID"].nunique() == 2 if "customerID" in cleaned else True
    assert cleaned["TotalCharges"].dtype.kind in "fi"
    assert cleaned["Churn"].isin([0, 1]).all()

    engineered = engineer_features(cleaned)
    assert "TotalServices" in engineered.columns
    assert "RiskScore" in engineered.columns
    assert engineered.loc[1, "HighValueCustomer"] == 0 or engineered.loc[1, "HighValueCustomer"] == 1


def test_split_data():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "Churn": [0, 0, 1, 1],
        }
    )

    X_train, X_test, y_train, y_test = split_data(df, target_column="Churn", test_size=0.5, random_state=1)
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert set(y_train.unique()) == {0, 1}
