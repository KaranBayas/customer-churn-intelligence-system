def split_data(df, target_column, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test