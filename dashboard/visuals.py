import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Churn", data=df, palette=["#4c78a8", "#f58518"], ax=ax)
    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of customers")
    ax.set_xticklabels(["No", "Yes"])
    fig.tight_layout()
    return fig


def plot_contract_churn(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    contract_summary = df.groupby(["Contract", "Churn"]).size().reset_index(name="count")
    sns.barplot(
        x="Contract",
        y="count",
        hue="Churn",
        data=contract_summary,
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )
    ax.set_title("Contract Type vs Churn")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Customer Count")
    ax.legend(title="Churn", labels=["No", "Yes"])
    fig.tight_layout()
    return fig


def plot_internet_service_churn(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    internet_summary = df.groupby(["InternetService", "Churn"]).size().reset_index(name="count")
    sns.barplot(
        x="InternetService",
        y="count",
        hue="Churn",
        data=internet_summary,
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )
    ax.set_title("Internet Service vs Churn")
    ax.set_xlabel("Internet Service")
    ax.set_ylabel("Customer Count")
    ax.legend(title="Churn", labels=["No", "Yes"])
    fig.tight_layout()
    return fig


def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["MonthlyCharges"], bins=25, kde=True, color="#4c78a8", ax=ax)
    ax.set_title("Monthly Charges Distribution")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Number of customers")
    fig.tight_layout()
    return fig


def plot_logistic_coefficients(importance_df, top_n=10):
    """Plot the most important logistic regression coefficients for churn prediction."""
    if importance_df is None or importance_df.empty:
        raise ValueError("importance_df must be a non-empty DataFrame.")
    if not {"feature", "coefficient"}.issubset(importance_df.columns):
        raise ValueError("importance_df must contain 'feature' and 'coefficient' columns.")

    plot_data = (
        importance_df.copy()
        .assign(abs_coef=lambda df: df["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
    )

    colors = plot_data["coefficient"].apply(lambda x: "#d73027" if x > 0 else "#1a9850")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_data["feature"], plot_data["coefficient"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_title("Logistic Regression Feature Effects on Churn Risk")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()

    for idx, coef in enumerate(plot_data["coefficient"]):
        offset = 0.01 if coef >= 0 else -0.01
        ha = "left" if coef >= 0 else "right"
        ax.text(coef + offset, idx, f"{coef:.3f}", va="center", ha=ha, color="#111111", fontsize=9)

    fig.tight_layout()
    return fig


def plot_logistic_coefficients(importance_df, top_n=10):
    """Plot the most important logistic regression coefficients for churn prediction."""
    if importance_df is None or importance_df.empty:
        raise ValueError("importance_df must be a non-empty DataFrame.")
    if not {"feature", "coefficient"}.issubset(importance_df.columns):
        raise ValueError("importance_df must contain 'feature' and 'coefficient' columns.")

    plot_data = (
        importance_df.copy()
        .assign(abs_coef=lambda df: df["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(top_n)
    )

    colors = plot_data["coefficient"].apply(lambda x: "#d73027" if x > 0 else "#1a9850")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_data["feature"], plot_data["coefficient"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_title("Logistic Regression Feature Effects on Churn Risk")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Feature")
    ax.invert_yaxis()

    for idx, coef in enumerate(plot_data["coefficient"]):
        offset = 0.01 if coef >= 0 else -0.01
        ha = "left" if coef >= 0 else "right"
        ax.text(coef + offset, idx, f"{coef:.3f}", va="center", ha=ha, color="#111111", fontsize=9)

    fig.tight_layout()
    return fig
