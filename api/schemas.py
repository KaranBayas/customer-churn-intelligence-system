from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    gender: str = Field(..., description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen flag: 0 or 1")
    Partner: str = Field(..., description="Does the customer have a partner? Yes or No")
    Dependents: str = Field(..., description="Does the customer have dependents? Yes or No")
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    PhoneService: str = Field(..., description="Does the customer have phone service? Yes or No")
    MultipleLines: str = Field(..., description="Does the customer have multiple lines? Yes, No, or No phone service")
    InternetService: str = Field(..., description="Type of internet service: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Online security subscription: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Online backup subscription: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Device protection subscription: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Tech support subscription: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Streaming TV subscription: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Streaming movies subscription: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Paperless billing option: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method used by the customer")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges for the customer")
    TotalCharges: float = Field(..., ge=0.0, description="Total charges for the customer")


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="Predicted churn label: 0 means no churn, 1 means churn")
    churn_probability: float = Field(..., description="Predicted probability of churn")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status of the API")
    version: str = Field(..., description="API semantic version")
