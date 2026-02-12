import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX


def train_sarima(train_data):

    print("Training SARIMA...")

    model = SARIMAX(
        train_data,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    joblib.dump(model_fit, "models/sarima_model.pkl")

    print("SARIMA Model Saved.")
    return model_fit
