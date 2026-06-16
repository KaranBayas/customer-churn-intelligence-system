import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dashboard.data import load_dashboard_data
from dashboard.model import (
    load_dashboard_model,
    predict_customer,
    risk_label,
    get_feature_importance,
)
from dashboard.utils import SAMPLE_CUSTOMERS, risk_color, risk_description
from dashboard.visuals import (
    plot_churn_distribution,
    plot_contract_churn,
    plot_internet_service_churn,
    plot_monthly_charges,
    plot_logistic_coefficients,
)


def style_page():
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #111111; }
        .stApp .block-container { background-color: #f8f9fa; }
        .big-font { font-size: 1.5rem; font-weight: 600; }
        .small-text { color: #5a5a5a; }
        .metric-label { font-size: 0.95rem; color: #5a5a5a; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_sidebar() -> str:
    st.sidebar.title("Customer Churn Dashboard")
    st.sidebar.markdown(
        "A beginner-friendly interface for customer churn analysis and prediction."
    )
    return st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Prediction",
            "Exploration",
            "Model Insights",
            "Sample Profiles",
        ],
    )


def show_home(df: pd.DataFrame):
    st.title("Customer Churn Intelligence")
    st.markdown(
        "This dashboard presents a clean churn prediction workflow with data insights, a prediction interface, and model explainability for a customer retention project."
    )

    churn_rate = df["Churn"].mean()
    st.metric(label="Dataset size", value=f"{len(df)} customers")
    st.metric(label="Churn rate", value=f"{churn_rate:.1%}")

    st.subheader("Key dataset insights")
    st.markdown(
        """
- Most churn occurs in month-to-month contracts.
- Customers without stable support and premium services are likelier to churn.
- Monthly charges and tenure are strong signals for churn risk.
        """
    )


def show_prediction(pipeline, df: pd.DataFrame):
    st.title("Churn Prediction")
    st.markdown("Use the form below to predict churn risk for a single customer profile.")

    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        with col2:
            device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=1500.0)

        submit_button = st.form_submit_button("Predict churn")

    if submit_button:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges,
        }

        with st.spinner("Running prediction..."):
            result = predict_customer(pipeline, customer_data)

        label = risk_label(result["probability"])
        style = risk_color(label)

        st.markdown("### Prediction result")
        st.metric(label="Churn risk", value=label, delta=f"{result['probability']:.1%}", delta_color="off")
        st.markdown(f"**Model prediction:** {result['prediction']}  ")
        st.markdown(f"**Churn probability:** {result['probability']:.1%}")

        st.markdown(f"<div style='padding:12px; background:{style}; border-radius:8px;'>{risk_description(label)}</div>", unsafe_allow_html=True)


def show_exploration(df: pd.DataFrame):
    st.title("Dashboard Explorer")
    st.markdown("Use these charts to understand churn patterns in the customer dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_churn_distribution(df))
        st.pyplot(plot_contract_churn(df))
    with col2:
        st.pyplot(plot_internet_service_churn(df))
        st.pyplot(plot_monthly_charges(df))


def show_model_insights(pipeline, df: pd.DataFrame):
    st.title("Model Insights")
    st.markdown(
        "This section shows feature importance and explainability for the trained churn model."
    )

    importance_df = get_feature_importance(pipeline)
    if importance_df is None or importance_df.empty:
        st.warning(
            "Feature importance is not available for this model. The model must expose either feature_importances_ or coef_."
        )
        return

    st.subheader("Top 20 most important features")
    top_features = importance_df.head(20).copy().reset_index(drop=True)
    st.dataframe(top_features)
    st.bar_chart(top_features.set_index("feature")["importance"])

    classifier = pipeline.named_steps.get("classifier")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if hasattr(classifier, "coef_"):
        st.subheader("Logistic regression coefficient sign")
        st.pyplot(plot_logistic_coefficients(top_features.head(10)))
        st.markdown(
            "Positive coefficients increase predicted churn risk, while negative coefficients reduce churn probability."
        )

    if hasattr(classifier, "feature_importances_"):
        st.subheader("SHAP explainability")
        try:
            raw_features = df.drop(columns=["Churn"], errors="ignore")
            if preprocessor is not None:
                transformed_features = preprocessor.transform(raw_features)
            else:
                transformed_features = raw_features

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(transformed_features)
            fig = plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_values,
                feature_names=importance_df["feature"].tolist(),
                show=False,
            )
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.warning(
                "SHAP explainability could not be generated for this model. Please check that the model and dashboard data are compatible."
            )
            st.write(f"Details: {exc}")


def show_sample_profiles():
    st.title("Sample Customer Profiles")
    st.markdown("Use these example profiles to explore how the model responds to typical customer situations.")
    sample_df = pd.DataFrame(SAMPLE_CUSTOMERS)
    st.dataframe(sample_df)


def main():
    style_page()
    page = get_sidebar()
    df = load_dashboard_data()
    pipeline = load_dashboard_model()

    if page == "Home":
        show_home(df)
    elif page == "Prediction":
        show_prediction(pipeline, df)
    elif page == "Exploration":
        show_exploration(df)
    elif page == "Model Insights":
        show_model_insights(pipeline, df)
    elif page == "Sample Profiles":
        show_sample_profiles()


if __name__ == "__main__":
    main()
