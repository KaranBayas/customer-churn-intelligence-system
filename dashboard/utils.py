SAMPLE_CUSTOMERS = [
    {
        "Profile": "New fiber month-to-month",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "MonthlyCharges": 99.65,
        "TotalCharges": 200.0,
        "Risk": "High Risk",
    },
    {
        "Profile": "Stable one-year customer",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "Contract": "One year",
        "InternetService": "DSL",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1000.0,
        "Risk": "Low Risk",
    },
    {
        "Profile": "Support-seeking customer",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "MonthlyCharges": 75.35,
        "TotalCharges": 650.0,
        "Risk": "Medium Risk",
    },
]


def risk_color(label: str) -> str:
    if label == "High Risk":
        return "#f8d7da"
    if label == "Medium Risk":
        return "#fff3cd"
    if label == "Low Risk":
        return "#d4edda"
    return "#e2e3e5"


def risk_description(label: str) -> str:
    if label == "High Risk":
        return "This customer has a strong probability of churn. Consider retention actions such as targeted offers, support outreach, or better contract terms."
    if label == "Medium Risk":
        return "This customer is at moderate churn risk. Monitor engagement and customer service interactions while offering loyalty incentives."
    if label == "Low Risk":
        return "This customer appears stable. Continue good service and consider growth opportunities through upsells or premium services."
    return "Prediction could not be completed. Please check the inputs and try again."
