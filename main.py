from src.data_preprocessing import load_and_preprocess
from src.train_arima import train_arima
from src.train_sarima import train_sarima
from src.train_prophet import train_prophet
from src.train_lstm import train_lstm
from src.evaluate import evaluate

print("Loading Data...")
data = load_and_preprocess("data/walmart_sales.csv")

train = data[:-20]
test = data[-20:]

print("Training ARIMA...")
arima_model = train_arima(train)

arima_forecast = arima_model.forecast(steps=20)
arima_metrics = evaluate(test, arima_forecast)

print("ARIMA Results:", arima_metrics)

print("Training SARIMA...")
sarima_model = train_sarima(train)
sarima_forecast = sarima_model.forecast(steps=20)
sarima_metrics = evaluate(test, sarima_forecast)

print("SARIMA Results:", sarima_metrics)

print("Training LSTM...")
lstm_model, lstm_pred = train_lstm(train, test)
lstm_results = evaluate(test, lstm_pred)
print("LSTM Results:", lstm_results)


print("Training Prophet...")
prophet_model = train_prophet(data)

print("Training Completed Successfully.")

print("\nModel Comparison")
print("-------------------")
print("ARIMA RMSE:", arima_metrics)
print("SARIMA RMSE:", sarima_metrics)
print("LSTM RMSE:", lstm_results)