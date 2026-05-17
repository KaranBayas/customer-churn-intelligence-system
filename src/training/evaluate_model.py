from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test, verbose=True):
    """Evaluate a trained churn model and return standard classification metrics."""
    y_pred = model.predict(X_test)
    y_probs = None
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probs) if y_probs is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

    if verbose:
        print("Model evaluation results:")
        print(metrics["classification_report"])
        print("Confusion matrix:\n", metrics["confusion_matrix"])

    return metrics
    