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

## Theme configuration

Place the `.streamlit` folder at the repository root, alongside `README.md` and `requirements.txt`.

The file `.streamlit/config.toml` locks the dashboard to a fixed light theme and enforces professional colors for text, background, and controls. This avoids dark mode rendering issues and keeps colors readable for all users.

## How Streamlit theme configuration works

Streamlit looks for `.streamlit/config.toml` in the current project folder when the app starts. The `[theme]` section defines:
- `base`: light or dark theme base
- `primaryColor`: the main accent color for buttons and active elements
- `backgroundColor`: main background for app panels
- `secondaryBackgroundColor`: panel/section background
- `textColor`: default text color
- `font`: font family for the app

Because we use `base = "light"`, Streamlit will ignore the operating system dark mode setting and keep the dashboard consistently light.

## Styling best practices

- Keep color contrast high: dark text on light backgrounds is easiest to read.
- Avoid custom CSS that depends on Streamlit internal class names as they may change.
- Use Streamlit theme settings for global colors rather than ad-hoc per-component styling.
- Keep the sidebar simple with clear headings, concise instructions, and visible controls.
- Use default component styles where possible; custom styles should be minimal and tested for readability.
- When displaying charts, use light plot backgrounds and dark grid/text colors.
- Prefer `st.markdown` and `st.metric` defaults for accessibility and clean appearance.

## Dashboard Preview

Use the `dashboard/assets/` folder to store screenshot images for the dashboard. Add images to the folder and reference them in this README with GitHub-style markdown links.

### Home page

![Home page preview](dashboard/assets/home-page.png)

A concise overview with dataset summary metrics, churn rate, and navigation to the prediction and insights sections.

### Prediction page

![Prediction page preview](dashboard/assets/prediction-page.png)

A clean single-customer prediction form with probability output and risk labels for churn decision support.

### Model insights page

![Model insights page preview](dashboard/assets/model-insights-page.png)

A professional view of feature effects and coefficient-based model explainability for logistic regression.

## Screenshot instructions

1. Capture dashboard screenshots locally using Streamlit.
2. Save images in `dashboard/assets/`.
3. Update the markdown image paths if you rename the files.
4. Keep screenshot filenames descriptive and lowercase, for example:
   - `home-page.png`
   - `prediction-page.png`
   - `model-insights-page.png`

## Notes

- The dashboard loads the saved model pipeline from `model/customer_churn_pipeline.pkl`.
- It uses the same preprocessing code in `src/preprocessing/` so prediction behavior matches the training workflow.
