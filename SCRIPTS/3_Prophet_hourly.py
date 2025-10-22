import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

Outputs:
    - Printed model evaluation metrics:
        RMSE, MAE, Hourly MAPE, and Daily MAPE
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

# Simple regressor: weekend flag (applies to each hour)
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
m.add_seasonality(name="weekly", period=7, fourier_order=10, prior_scale=10)
m.add_seasonality(name="daily",  period=1, fourier_order=12, prior_scale=10)
m.add_regressor("is_weekend")

# ---------- Fit on Capped Train ----------
m.fit(train_cap[["ds", "y", "is_weekend"]])

# ---------- Forecast on Test (unchanged 20%) ----------
fcst = m.predict(test[["ds", "is_weekend"]])

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
import matplotlib.pyplot as plt
import pandas as pd

# Merge predictions onto full hourly series
full_h = hourly[["ds","y"]].copy()
full_h = full_h.merge(fcst[["ds","yhat"]], on="ds", how="left")  # yhat only on test rows

split_dt = test["ds"].min()

plt.figure(figsize=(12, 5))

# 1) Train data (light gray)
mask_train = full_h["ds"] < split_dt
plt.plot(full_h.loc[mask_train, "ds"], full_h.loc[mask_train, "y"],
         color="#bfbfbf", linewidth=0.7, label="Train Data")

# 2) Test actuals (black)
mask_test = full_h["ds"] >= split_dt
plt.plot(full_h.loc[mask_test, "ds"], full_h.loc[mask_test, "y"],
         color="black", linewidth=0.9, label="Actual (Test)")

# 3) Prophet predictions on test window (blue)
plt.plot(full_h.loc[mask_test, "ds"], full_h.loc[mask_test, "yhat"],
         color="royalblue", linewidth=0.9, label="Prophet Prediction")

# 4) Vertical split line
plt.axvline(split_dt, color="red", linestyle="--", linewidth=1.2, label="Train/Test Split")

# Cosmetics
plt.title("Hourly Prophet Model — Observed vs Predicted")
plt.xlabel("Date")
plt.ylabel("Hourly Collisions")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
