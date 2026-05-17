from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(categorical_cols, numerical_cols):
    """Create a preprocessing transformer that encodes categorical and scales numerical data."""
    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False),
                categorical_cols,
            )
        )

    if numerical_cols:
        transformers.append(
            (
                "num",
                StandardScaler(),
                numerical_cols,
            )
        )

    if not transformers:
        raise ValueError("No feature columns were provided for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="passthrough")
