from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def build_classifier(model_name="logistic", **kwargs):
    """Build a classifier for supervised churn prediction."""
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=42, **kwargs)
    if model_name == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **kwargs)

    raise ValueError(f"Unsupported model_name: {model_name}")


def train_model(X_train, y_train, estimator=None):
    """Train a classifier with training data."""
    if estimator is None:
        estimator = build_classifier()

    estimator.fit(X_train, y_train)
    return estimator
