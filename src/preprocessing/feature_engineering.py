def engineer_features(df):
    services = [
    'PhoneService',
    'MultipleLines',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

    df['TotalServices'] = (
    df[services] == 'Yes'
    ).sum(axis=1)

    df['AvgMonthlySpend'] = (
    df['TotalCharges'] / (df['tenure'] + 1)
    )

    df['HighValueCustomer'] = (
    df['MonthlyCharges'] > 80
    ).astype(int)

    df['FiberPremiumRisk'] = (
    (df['InternetService'] == 'Fiber optic') &
    (df['MonthlyCharges'] > 80)
    ).astype(int)

    df['SeniorNoSupport'] = (
    (df['SeniorCitizen'] == 1) &
    (df['TechSupport'] == 'No')
    ).astype(int)

    df['RiskScore'] = (
    (df['Contract'] == 'Month-to-month').astype(int) +
    (df['InternetService'] == 'Fiber optic').astype(int) +
    (df['TechSupport'] == 'No').astype(int) +
    (df['PaymentMethod'] == 'Electronic check').astype(int)
    )


    return df