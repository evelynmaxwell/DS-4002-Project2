import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Description:
    This script fits and evaluates a Seasonal ARIMA (SARIMA) model to forecast 
    hourly motor vehicle collisions in New York City using cleaned crash data.
Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing a 
        'timestamp' column representing crash date and time.
Process:
    1. Reads and parses the collision data.
    2. Aggregates crash counts at an hourly frequency.
    3. Trims the dataset to the most recent two years for computational efficiency.
    4. Splits the time series into training and testing sets (80/20).
    5. Fits a SARIMA(1,0,1)(0,1,1,24) model to capture daily seasonality.
    6. Forecasts hourly crashes and evaluates model performance.
Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE (expected ~ RMSE: 3.80, MAE: 2.95, MAPE: 46.24%)
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

# Prepare hourly data
hourly = df.set_index('timestamp').resample('h').size()
cut = hourly.index.max() - pd.Timedelta(days=365*2)
hourly = hourly.loc[cut:]   # Only include data within the last two years

# 80/20 train-test split
hourly_train, hourly_test = time_series_split(hourly, 0.8)

# Fit model
hourly_model = SARIMAX(
    hourly_train,
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Evaluate model
hourly_forecast = hourly_model.get_forecast(steps=len(hourly_test))
hourly_pred = hourly_forecast.predicted_mean
hourly_ci = hourly_forecast.conf_int()

hourly_rmse, hourly_mae, hourly_mape = evaluate_forecast(hourly_test, hourly_pred)
print(f"\nHOURLY MODEL:")
print(f"RMSE: {hourly_rmse:.2f} | MAE: {hourly_mae:.2f} | MAPE: {hourly_mape:.2f}%")
