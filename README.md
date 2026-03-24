# Crop Yield Prediction System

This project is a crop yield prediction product built with:

- `Streamlit` for the public web app
- `FastAPI` for API architecture
- `scikit-learn` for model training and inference
- `SHAP` for explainability

The cleaned dataset is stored at [cleaned_crop_data.csv](/c:/Users/Arul/Desktop/c_y_p/data/cleaned_crop_data.csv).

## What This App Does

- Predicts crop yield from `state`, `crop`, `season`, `area`, `rainfall`, and `temperature`
- Shows prediction confidence
- Visualizes feature importance and SHAP explanations
- Includes scenario simulation for weather changes
- Uses real agriculture photos and a custom agri-tech UI

## Project Structure

```text
c_y_p/
├── .streamlit/
│   └── config.toml
├── backend/
│   ├── __init__.py
│   ├── explain.py
│   ├── main.py
│   ├── model.py
│   ├── preprocess.py
│   ├── saved_model.pkl
│   └── train.py
├── data/
│   ├── cleaned/
│   │   ├── cleaned_data.csv
│   │   ├── data_analysis_report.json
│   │   └── ml_ready_data.csv
│   └── cleaned_crop_data.csv
├── frontend/
│   ├── assets/
│   └── app.py
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## Notes About Training

The dataset contains some columns that are intentionally excluded from prediction input:

- `Production_in_tons` is excluded because it can leak target information
- `N`, `P`, `K`, and `pH` are excluded from the public prediction form to keep inference aligned with the UI

The production model is a `RandomForestRegressor`.

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

### Streamlit App

```powershell
streamlit run streamlit_app.py
```

Local URL:

```text
http://localhost:8501
```

### FastAPI Backend

The repo still includes the FastAPI backend for architecture and API usage:

```powershell
uvicorn backend.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

## Resume-Friendly Deployment

The easiest way to get one public link for your resume is to deploy the Streamlit app as a single site.

This repo is already prepared for that:

- `streamlit_app.py` is the root app entrypoint
- the Streamlit app can run without a separately deployed FastAPI backend
- the trained model file is already included

### Recommended Platform

- Streamlit Community Cloud

Official docs:

- https://docs.streamlit.io/deploy/streamlit-community-cloud

### Deploy Steps

1. Push this project to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app and choose your repository.
4. Set the main file path to `streamlit_app.py`.
5. Deploy.

Your public link will look like:

```text
https://your-app-name.streamlit.app
```

That is the link you can place in:

- your resume
- GitHub profile
- portfolio
- LinkedIn projects section

## If You Want A Two-Service Deployment

If you want to showcase the architecture as separate frontend and backend services:

- Deploy FastAPI on Render or Railway
- Deploy Streamlit separately

Official docs:

- https://docs.render.com/deploy-fastapi

For a resume, though, one Streamlit link is usually cleaner and easier to maintain.

## Example Prediction Input

```json
{
  "crop": "rice",
  "state": "tamil nadu",
  "season": "kharif",
  "area": 2.5,
  "rainfall": 120,
  "temperature": 30
}
```

## Important Note

Creating the final public deployment URL still requires:

- your GitHub repository
- your Streamlit Cloud account login

So the app is now deployment-ready, but the actual public URL must be created through your account.
