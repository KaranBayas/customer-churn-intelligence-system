def data_encoding(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Placeholder for data encoding code
    print("Encoding data:", X)
    # Simulate encoding process
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns
    preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(drop='first'),
            categorical_cols
        ),
        (
            'num',
            StandardScaler(),
            numerical_cols
        )
      ]
    )
    X_train_encoded = preprocessor.fit_transform(X)

    return X_train_encoded