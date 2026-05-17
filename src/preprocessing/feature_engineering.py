def engineer_features(df):
    """Create derived features that capture churn risk for each customer."""
    df = df.copy()
    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    service_columns = [col for col in services if col in df.columns]
    df["TotalServices"] = (df[service_columns] == "Yes").sum(axis=1) if service_columns else 0

    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    else:
        df["AvgMonthlySpend"] = 0

    df["HighValueCustomer"] = (df.get("MonthlyCharges", 0) > 80).astype(int)
    df["FiberPremiumRisk"] = (
        (df.get("InternetService", "") == "Fiber optic")
        & (df.get("MonthlyCharges", 0) > 80)
    ).astype(int)
    df["SeniorNoSupport"] = (
        (df.get("SeniorCitizen", 0) == 1)
        & (df.get("TechSupport", "") == "No")
    ).astype(int)

    df["RiskScore"] = (
        (df.get("Contract", "") == "Month-to-month").astype(int)
        + (df.get("InternetService", "") == "Fiber optic").astype(int)
        + (df.get("TechSupport", "") == "No").astype(int)
        + (df.get("PaymentMethod", "") == "Electronic check").astype(int)
    )

    return df
