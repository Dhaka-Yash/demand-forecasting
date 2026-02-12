import streamlit as st
import os
import joblib
import pandas as pd
from pathlib import Path


st.title("Demand Forecasting Dashboard")
st.subheader("Business Insights")
st.write("Forecasting Walmart Weekly Sales using ARIMA, SARIMA, and Prophet models")
st.write("""
- Accurate demand forecasting reduces stockouts.
- Helps optimize inventory levels.
- Improves supply chain efficiency.
- Reduces holding cost.
""")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "arima_model.pkl")

model = joblib.load(model_path)


@st.cache_resource
def load_model(model_filename: str):
    model_path = MODEL_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
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
    st.info("Run training first to generate model files in the models folder.")
except Exception as e:
    st.error(f"Dashboard error: {e}")
