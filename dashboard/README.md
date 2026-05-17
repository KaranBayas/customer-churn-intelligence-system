# Dashboard

This dashboard folder contains a Streamlit application for the Customer Churn Intelligence System.

## Structure

- `dashboard/app.py` - main Streamlit app entry point
- `dashboard/data.py` - functions to load and prepare the dataset
- `dashboard/model.py` - model loading and prediction helpers
- `dashboard/visuals.py` - plotting helpers for charts and dashboards
- `dashboard/utils.py` - reusable constants and UI helpers

## Run locally

From the repository root:

```bash
streamlit run dashboard/app.py
```

Then open the URL shown by Streamlit, usually `http://localhost:8501`.

## Notes

- The dashboard loads the saved model pipeline from `model/customer_churn_pipeline.pkl`.
- It uses the same preprocessing code in `src/preprocessing/` so prediction behavior matches the training workflow.
