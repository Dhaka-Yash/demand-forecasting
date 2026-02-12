from prophet import Prophet
import pandas as pd
import joblib


def train_prophet(df):

    print("Training Prophet...")

    prophet_df = df.reset_index()
    prophet_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    joblib.dump(model, "models/prophet_model.pkl")

    print("Prophet Model Saved.")
    return model

model = Prophet()
model.add_country_holidays(country_name='US')
