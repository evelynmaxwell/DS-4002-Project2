import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Description:
    This script fits and evaluates a Seasonal ARIMA (SARIMA) model to forecast 
    daily motor vehicle collisions in New York City using cleaned crash data.
Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing a 
        'timestamp' column representing crash date and time.
Process:
    1. Reads and parses the collision data.
    2. Aggregates crash counts at a daily frequency.
    3. Splits the time series into training and testing sets (80/20).
    4. Fits a SARIMA(0,1,0)(1,1,1,7) model to capture weekly seasonality.
    5. Forecasts daily crashes and evaluates model performance.
Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE (expected ~ RMSE: 33.02, MAE: 25.77, MAPE: 11.90%)
"""

def time_series_split(series, train_ratio=0.8):
    n_train = int(len(series) * train_ratio)
    train = series.iloc[:n_train]
    test = series.iloc[n_train:]
    return train, test

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return rmse, mae, mape

# Read collision data
df = pd.read_csv('../DATA/collisions_cleaned.csv',
                 low_memory=False,
                 dtype={'BOROUGH': 'string'})
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Prepare daily data
daily = df.set_index('timestamp').resample('D').size()

# 80/20 train-test split
daily_train, daily_test = time_series_split(daily, 0.8)

# Fit model
sarima_daily = SARIMAX(
    daily_train,
    order=(0, 1, 0),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Evaluate model
daily_forecast = sarima_daily.get_forecast(steps=len(daily_test))
daily_pred = daily_forecast.predicted_mean
daily_ci = daily_forecast.conf_int()

daily_rmse, daily_mae, daily_mape = evaluate_forecast(daily_test, daily_pred)
print(f"\nDAILY MODEL RESULTS:")
print(f"RMSE: {daily_rmse:.2f} | MAE: {daily_mae:.2f} | MAPE: {daily_mape:.2f}%")
