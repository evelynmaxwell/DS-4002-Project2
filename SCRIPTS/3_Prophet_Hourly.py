import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Description:
    This script fits and evaluates a Prophet model to forecast **daily** motor vehicle
    collisions in New York City using cleaned crash data.

Inputs:
    - ../DATA/collisions_cleaned.csv.zip
        A cleaned dataset with a 'timestamp' column (crash datetime).

Process:
    1. Reads and parses the collision data.
    2. Aggregates crash counts at a daily frequency.
    3. Creates a simple calendar regressor (weekend).
    4. Splits the series into training and testing sets (80/20).
    5. Fits a Prophet model with US holidays and a weekend regressor.
    6. Forecasts daily crashes and evaluates performance.

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, and MAPE
"""

# ---------- Helpers ----------
def time_series_split(df, train_ratio=0.8):
    """Split a dataframe with columns ['ds','y', ...] chronologically."""
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def evaluate_forecast(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    # Safe MAPE: avoid div-by-zero; treat 0 as 1 in denominator
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return rmse, mae, mape


# ---------- Load & Aggregate Data ----------
usecols = ["timestamp"]
df = pd.read_csv("../DATA/collisions_cleaned.csv.zip", usecols=usecols, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# Daily collision counts (each row is an incident)
daily = (
    df.sort_values("timestamp")
    .set_index("timestamp")
    .resample("D")  
      .size()
      .rename("y")
      .to_frame()
      .reset_index()
      .rename(columns={"timestamp": "ds"})
)

# Simple regressor: weekend flag
daily["is_weekend"] = (daily["ds"].dt.dayofweek >= 5).astype(int)

# ---------- Train/Test Split ----------
train, test = time_series_split(daily, train_ratio=0.8)

# ---------- Fit Prophet ----------
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,   # daily data, no intra-day seasonality
    n_changepoints=80,
    changepoint_range=0.95,
    changepoint_prior_scale=0.5,
    seasonality_prior_scale=15.0,
    holidays_prior_scale=0.01,
    seasonality_mode="multiplicative",
    uncertainty_samples=0
)
m.add_country_holidays(country_name="US")
m.add_regressor("is_weekend")

m.fit(train[["ds", "y", "is_weekend"]])

# ---------- Forecast ----------
fcst = m.predict(test[["ds", "is_weekend"]])

# ---------- Evaluation ----------
eval_df = test[["ds", "y"]].merge(fcst[["ds", "yhat"]], on="ds", how="left")
rmse, mae, mape = evaluate_forecast(eval_df["y"].values, eval_df["yhat"].values)

print("\nDAILY PROPHET MODEL:")
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
