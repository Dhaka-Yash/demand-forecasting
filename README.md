## Demand Forecasting Project

Time-series demand forecasting project using Walmart weekly sales data.  
The repository includes data preprocessing, ARIMA/SARIMA/Prophet model training, evaluation utilities, and a Streamlit dashboard for interactive forecasts.

## Features

- Weekly sales preprocessing pipeline (`Date` -> weekly indexed time series)
- Forecasting models:
- ARIMA (`statsmodels`)
- SARIMA (`statsmodels`)
- Prophet (`prophet`)
- Basic evaluation metrics: MAE and RMSE
- Streamlit dashboard to visualize forecast output from saved models

## Project Structure

```text
demand-forecasting/
|-- app/
|   `-- dashboard.py
|-- data/
|   `-- walmart_sales.csv
|-- models/
|   |-- arima_model.pkl
|   |-- sarima_model.pkl
|   `-- prophet_model.pkl
|-- notebooks/
|   `-- 01_Demand_Forecasting.ipynb
|-- src/
|   |-- data_preprocessing.py
|   |-- train_arima.py
|   |-- train_sarima.py
|   |-- train_prophet.py
|   |-- forecast.py
|   `-- evaluate.py
|-- main.py
|-- requirements.txt
`-- README.md
```

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train Models

You can train and save each model individually from Python:

```python
from src.data_preprocessing import load_and_preprocess
from src.train_arima import train_arima
from src.train_sarima import train_sarima
from src.train_prophet import train_prophet

data = load_and_preprocess("data/walmart_sales.csv")
train = data[:-20]

train_arima(train)
train_sarima(train)
train_prophet(data)
```

Saved model artifacts:
- `models/arima_model.pkl`
- `models/sarima_model.pkl`
- `models/prophet_model.pkl`

## Run Dashboard

Start the Streamlit app:

```powershell
streamlit run app/dashboard.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Streamlit Cloud Deployment (Without Committing Models)

Model `.pkl` files are ignored in git, so configure remote model URLs for deployment.

1. Upload model files to external storage (for example: Hugging Face, S3, or Google Drive direct-download links).
2. In Streamlit app settings, add secrets:

```toml
[model_urls]
arima_model.pkl = "https://<your-url>/arima_model.pkl"
sarima_model.pkl = "https://<your-url>/sarima_model.pkl"
prophet_model.pkl = "https://<your-url>/prophet_model.pkl"
```

3. Redeploy. On first run, the app downloads models into `models/` at runtime.

You can also use environment variables instead of secrets:
- `MODEL_URL_ARIMA`
- `MODEL_URL_SARIMA`
- `MODEL_URL_PROPHET`

## Evaluate Forecasts

The `src/evaluate.py` module provides:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Data

- Input dataset: `data/walmart_sales.csv`
- Required columns used by preprocessing:
- `Date`
- `Weekly_Sales`

## Known Issues

- `main.py` currently imports `src/train_lstm.py`, but that file does not exist in this repository.
- Because of this, running `python main.py` will fail unless LSTM code is added or the LSTM section is removed.

## Notes

- `requirements.txt` is the full environment list.
- `Requirement.txt` is a smaller dependency list; prefer `requirements.txt` for reproducibility.
