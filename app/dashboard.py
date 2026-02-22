import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import joblib
import pandas as pd
import streamlit as st


st.title("Demand Forecasting Dashboard")
st.subheader("Business Insights")
st.write("Forecasting Walmart Weekly Sales using ARIMA, SARIMA, and Prophet models")
st.write("""
- Accurate demand forecasting reduces stockouts.
- Helps optimize inventory levels.
- Improves supply chain efficiency.
- Reduces holding cost.
""")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL_ENV_KEYS = {
    "arima_model.pkl": "MODEL_URL_ARIMA",
    "sarima_model.pkl": "MODEL_URL_SARIMA",
    "prophet_model.pkl": "MODEL_URL_PROPHET",
}

MODEL_URL_SECRET_ALIASES = {
    "arima_model.pkl": ["arima_model.pkl", "arima", "ARIMA"],
    "sarima_model.pkl": ["sarima_model.pkl", "sarima", "SARIMA"],
    "prophet_model.pkl": ["prophet_model.pkl", "prophet", "PROPHET"],
}


def _resolve_secret_url(model_filename: str) -> str | None:
    secret_urls = st.secrets.get("model_urls", {})
    aliases = MODEL_URL_SECRET_ALIASES.get(model_filename, [model_filename])
    for alias in aliases:
        value = secret_urls.get(alias) if hasattr(secret_urls, "get") else None
        if value:
            return str(value)

    env_key = MODEL_URL_ENV_KEYS.get(model_filename)
    if env_key:
        # Some deployments place these values in Streamlit secrets but not OS env vars.
        secret_env_value = st.secrets.get(env_key)
        if secret_env_value:
            return str(secret_env_value)

    return None


def get_model_url(model_filename: str) -> str | None:
    secret_url = _resolve_secret_url(model_filename)
    if secret_url:
        return secret_url

    env_key = MODEL_URL_ENV_KEYS.get(model_filename)
    if env_key:
        return os.getenv(env_key)
    return None


def ensure_model_available(model_filename: str) -> Path:
    model_path = MODEL_DIR / model_filename
    if model_path.exists():
        return model_path

    model_url = get_model_url(model_filename)
    if not model_url:
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Configure model URL in "
            "Streamlit secrets under [model_urls] using keys arima_model.pkl / "
            "sarima_model.pkl / prophet_model.pkl (or arima/sarima/prophet), or set "
            "MODEL_URL_ARIMA / MODEL_URL_SARIMA / MODEL_URL_PROPHET."
        )

    try:
        with urlopen(model_url, timeout=60) as response:
            model_path.write_bytes(response.read())
    except URLError as exc:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download {model_filename} from {model_url}") from exc

    return model_path


@st.cache_resource
def load_model(model_filename: str):
    model_path = ensure_model_available(model_filename)
    return joblib.load(model_path)


model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA", "Prophet"])
periods = st.slider("Forecast Weeks", 1, 52, 20)

try:
    if model_choice == "ARIMA":
        model = load_model("arima_model.pkl")
        forecast_values = model.forecast(steps=periods)
        forecast_df = pd.DataFrame(
            {
                "week": range(1, periods + 1),
                "forecast": pd.Series(forecast_values).values,
            }
        )

    elif model_choice == "SARIMA":
        model = load_model("sarima_model.pkl")
        forecast_values = model.forecast(steps=periods)
        forecast_df = pd.DataFrame(
            {
                "week": range(1, periods + 1),
                "forecast": pd.Series(forecast_values).values,
            }
        )

    else:
        model = load_model("prophet_model.pkl")
        future = model.make_future_dataframe(periods=periods, freq="W")
        prophet_forecast = model.predict(future)
        forecast_df = prophet_forecast[["ds", "yhat"]].tail(periods).rename(
            columns={"ds": "week", "yhat": "forecast"}
        )

    st.subheader(f"{model_choice} Forecast")
    st.line_chart(forecast_df.set_index("week")["forecast"])
    st.dataframe(forecast_df, width="stretch")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run training locally and upload models to external storage, then configure model URLs.")
except Exception as e:
    st.error(f"Dashboard error: {e}")
