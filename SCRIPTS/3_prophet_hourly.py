"""
Description:
    This script fits and evaluates a Prophet model to forecast **hourly** motor vehicle
    collisions in New York City using cleaned crash data.

Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset with a 'timestamp' column (crash datetime).

Process:
    1. Reads and parses the collision data.
    2. Aggregates crash counts at an hourly frequency.
    3. Filters to the LAST TWO YEARS of data.
    4. Creates a simple calendar regressor (weekend).
    5. Splits the series into training and testing sets (80/20) after the 2y filter.
    6. Fits a Prophet model with tuned weekly (7d) and daily (24h) seasonalities.
    7. Forecasts hourly crashes and evaluates performance (hourly + daily-aggregated MAPE).
    8. Saves a PNG plot to ../OUTPUT/daily_prophet_model.png

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, Hourly MAPE, and Daily MAPE (expected ~ RMSE: 3.61, MAE: 2.81, MAPE (hourly): 38.88%, MAPE (daily agg): 11.70%)
    - PNG file of prediction plot
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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
df = pd.read_csv("../DATA/collisions_cleaned.csv", usecols=usecols,parse_dates=["timestamp"])

hourly = (
    df.sort_values("timestamp")
    .set_index("timestamp")
      .resample("h")
      .size()
      .rename("y")
      .to_frame()
      .reset_index()
      .rename(columns={"timestamp": "ds"})
)

# ---------- Filter to the LAST TWO YEARS FIRST ----------
if not hourly.empty:
    two_year_start = hourly["ds"].max() - pd.Timedelta(days=365*2)
    hourly = hourly.loc[hourly["ds"] >= two_year_start].reset_index(drop=True)

# Add regressors
hourly["hour"] = hourly["ds"].dt.hour
scaler = StandardScaler()
hourly["hour_scaled"] = scaler.fit_transform(hourly[["hour"]])
hourly["is_weekend"] = (hourly["ds"].dt.dayofweek >= 5).astype(int)

# ---------- Train/Test Split (80/20 AFTER 2y filter) ----------
train, test = time_series_split(hourly, train_ratio=0.8)


m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,    # will add tuned weekly below
    daily_seasonality=False,     # will add tuned 24h below
    n_changepoints=50,
    changepoint_range=0.90,
    changepoint_prior_scale=0.20,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=0.01,
    seasonality_mode="additive",
    uncertainty_samples=0,
)

m.add_country_holidays(country_name="US")
m.add_seasonality(name='daily_24h', period=24/24, fourier_order=12)  # captures within-day pattern
m.add_seasonality(name='weekly', period=7, fourier_order=15)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
m.add_regressor("is_weekend")
m.add_regressor("hour_scaled")

# ---------- Fit on Capped Train ----------
m.fit(train[["ds", "y", "is_weekend","hour_scaled"]])

# ---------- Forecast on Test (unchanged 20%) ----------
fcst = m.predict(test[["ds", "is_weekend","hour_scaled"]])

# ---------- Evaluation (Hourly + Daily) ----------
eval_df = test[["ds", "y"]].merge(fcst[["ds", "yhat"]], on="ds", how="left")

rmse, mae, mape = evaluate_forecast(eval_df["y"].values, eval_df["yhat"].values)

# Aggregate to daily for daily MAPE
tmp = eval_df.copy()
tmp["date"] = tmp["ds"].dt.date
agg = (
    tmp.groupby("date", as_index=False)
       .agg(y_day=("y", "sum"), yhat_day=("yhat", "sum"))
)
# daily MAPE (ignore days with y_day == 0)
daily_den = agg["y_day"].replace(0, np.nan)
mape_daily = (np.abs(agg["y_day"] - agg["yhat_day"]) / daily_den).mean() * 100

print("\nHOURLY PROPHET MODEL (train capped to last 2y post-split):")
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE (hourly): {mape:.2f}% | MAPE (daily agg): {mape_daily:.2f}%")

# ---------- Prophet Hourly Visual: Train vs Test & Predictions ----------
split_dt = test["ds"].min()

plt.figure(figsize=(12, 5))

# 1) Train data (light gray)
plt.plot(train["ds"], train["y"], color="#bfbfbf", linewidth=0.7, label="Train Data")

# 2) Test actuals (black)
plt.plot(test["ds"], test["y"], color="black", linewidth=0.9, label="Actual (Test)")

# 3) Prophet predictions (blue)
plt.plot(fcst["ds"], fcst["yhat"], color="royalblue", linewidth=0.9, label="Prophet Prediction")

# 4) Vertical split line
plt.axvline(split_dt, color="red", linestyle="--", linewidth=1.0, label="Train/Test Split")

# Cosmetics
plt.title("Hourly Prophet Model — Observed vs Predicted")
plt.xlabel("Date")
plt.ylabel("Hourly Collisions")
plt.legend(frameon=True)
plt.tight_layout()

# Save plot
output_path = "../OUTPUT/hourly_prophet_model.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nPlot saved to: {output_path}")
