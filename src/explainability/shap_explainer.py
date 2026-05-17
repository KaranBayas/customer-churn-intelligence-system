import shap


def create_explainer(model, X_train):
    """Instantiate a SHAP explainer for the trained pipeline."""
    return shap.Explainer(model, X_train)


def compute_shap_values(explainer, X):
    """Compute SHAP values for a set of examples."""
    return explainer(X)


def plot_shap_summary(shap_values, feature_names=None):
    """Plot a summary of SHAP values."""
    shap.summary_plot(shap_values, feature_names=feature_names)


def plot_shap_waterfall(shap_values, sample_index=0):
    """Plot a waterfall explanation for a single sample."""
    shap.plots.waterfall(shap_values[sample_index])
