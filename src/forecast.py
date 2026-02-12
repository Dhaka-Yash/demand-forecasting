import joblib
import numpy as np


def forecast_arima(steps):
    model = joblib.load("models/arima_model.pkl")
    return model.forecast(steps=steps)


def forecast_sarima(steps):
    model = joblib.load("models/sarima_model.pkl")
    return model.forecast(steps=steps)


def forecast_prophet(steps):
    model = joblib.load("models/prophet_model.pkl")
    future = model.make_future_dataframe(periods=steps, freq='W')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(steps)
