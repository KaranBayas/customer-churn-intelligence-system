import matplotlib.pyplot as plt
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
