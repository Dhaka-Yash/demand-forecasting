import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # p-value


def train_arima(train_data):

    print("Checking stationarity...")
    p_value = check_stationarity(train_data)

    if p_value > 0.05:
        print("Data not stationary. Differencing applied.")
        d = 1
    else:
        print("Data stationary.")
        d = 0

    model = ARIMA(train_data, order=(2, d, 2))
    model_fit = model.fit()

    joblib.dump(model_fit, "models/arima_model.pkl")

    print("ARIMA Model Saved.")
    return model_fit
