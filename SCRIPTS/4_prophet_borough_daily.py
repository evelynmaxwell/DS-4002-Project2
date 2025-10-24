"""
Forecast Daily Motor Vehicle Collisions per NYC Borough using Prophet
---------------------------------------------------------------------

Description:
    This script fits separate Prophet models for each NYC borough to forecast
    **daily collision counts** for a specified date using historical data.
    It uses geographic bounding boxes to assign each record to boroughs.
    The script prints a clean summary of evaluation metrics and a single-day forecast
    (with uncertainty intervals) for each borough and the overall city total.

Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing:
          • 'timestamp' — datetime of each crash.
          • 'location'  — text coordinates formatted as "(latitude, longitude)".

Process:
    1. Parse and clean input data.
    2. Convert "(lat, lon)" strings into numeric latitude and longitude.
    3. Assign each record to one of five boroughs using coarse geographic bounding boxes.
    4. Aggregate data into daily counts per borough and flag weekends.
    5. For each borough:
        • Train Prophet on 80% of the series.
        • Evaluate RMSE, MAE, MAPE on the remaining 20%.
        • Store the trained model for forecasting.
    6. Forecast a specified future date (default: 2025-11-01) for all boroughs.
    7. Print summary metrics and forecast tables to the console.

Outputs:
    - Summary table of per-borough RMSE, MAE, MAPE.
    - Forecast table (yhat, yhat_lower, yhat_upper) for each borough.
    - Citywide aggregated forecast for the same day.

Notes:
    • Confidence interval lower bounds are capped at 0 (no negative collisions).
    • Prophet uncertainty intervals are 80% by default.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------- Helper Functions --------------------------------------------------
def time_series_split(df, train_ratio=0.8):
    """Chronologically split a dataframe with ['ds','y', ...] into train/test segments."""
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def evaluate_forecast(actual, predicted):
    """Compute RMSE, MAE, and MAPE for forecast evaluation."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
    return rmse, mae, mape

# ---------- NYC Borough Bounding Boxes (Lat, Lon) -----------------------------
# Approximate geographic boundaries for fast borough tagging
BOROUGH_BBOX = {
    "Manhattan":     {"lat": (40.68, 40.90), "lon": (-74.05, -73.90)},
    "Brooklyn":      {"lat": (40.55, 40.75), "lon": (-74.10, -73.83)},
    "Queens":        {"lat": (40.53, 40.82), "lon": (-73.98, -73.68)},
    "Bronx":         {"lat": (40.79, 40.92), "lon": (-73.95, -73.75)},
    "Staten Island": {"lat": (40.48, 40.66), "lon": (-74.28, -74.00)},
}

def assign_borough_vectorized(df):
    """Assign borough labels to each row based on latitude and longitude ranges."""
    conds = [
        df["latitude"].between(*BOROUGH_BBOX["Manhattan"]["lat"])     & df["longitude"].between(*BOROUGH_BBOX["Manhattan"]["lon"]),
        df["latitude"].between(*BOROUGH_BBOX["Brooklyn"]["lat"])      & df["longitude"].between(*BOROUGH_BBOX["Brooklyn"]["lon"]),
        df["latitude"].between(*BOROUGH_BBOX["Queens"]["lat"])        & df["longitude"].between(*BOROUGH_BBOX["Queens"]["lon"]),
        df["latitude"].between(*BOROUGH_BBOX["Bronx"]["lat"])         & df["longitude"].between(*BOROUGH_BBOX["Bronx"]["lon"]),
        df["latitude"].between(*BOROUGH_BBOX["Staten Island"]["lat"]) & df["longitude"].between(*BOROUGH_BBOX["Staten Island"]["lon"]),
    ]
    choices = np.array(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], dtype=object)
    return np.select(conds, choices, default=None).astype("object")

# ---------- Load and Preprocess Data ------------------------------------------
usecols = ["timestamp", "location"]
df = pd.read_csv("../DATA/collisions_cleaned.csv", usecols=usecols)

# Convert timestamps and drop invalid/missing rows
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp", "location"])

# Parse "(lat, lon)" text into numeric latitude and longitude
m = (
    df["location"].astype(str).str.strip()
      .str.extract(r".*?\(?\s*([+-]?\d+(?:\.\d+)?)\s*[, ]\s*([+-]?\d+(?:\.\d+)?)\s*\)?", expand=True)
)
df["latitude"]  = pd.to_numeric(m[0], errors="coerce")
df["longitude"] = pd.to_numeric(m[1], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])

# Filter to plausible NYC bounds
NYC_LAT = (40.45, 40.95)
NYC_LON = (-74.30, -73.65)
df = df[df["latitude"].between(*NYC_LAT) & df["longitude"].between(*NYC_LON)]

# Tag by borough
df["borough"] = assign_borough_vectorized(df)
df = df.dropna(subset=["borough"])

# ---------- Aggregate to Daily Counts -----------------------------------------
df["ds"] = df["timestamp"].dt.floor("D")  # Prophet expects 'ds' column
daily = (
    df.groupby(["borough", "ds"], as_index=False)
      .size()
      .rename(columns={"size": "y"})      # Prophet expects 'y' target column
)
# Add a simple binary weekend regressor
daily["is_weekend"] = (daily["ds"].dt.dayofweek >= 5).astype(int)

# ---------- Train Prophet Model for Each Borough ------------------------------
results = []
models_by_borough = {}

for boro, series in daily.groupby("borough"):
    series = series.sort_values("ds").reset_index(drop=True)
    if len(series) < 100:
        # Skip boroughs with insufficient data to fit a stable model
        continue

    # Split chronologically (80% train / 20% test)
    train, test = time_series_split(series, train_ratio=0.8)

    # Initialize Prophet model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        n_changepoints=80,
        changepoint_range=0.95,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=15.0,
        holidays_prior_scale=0.01,
        seasonality_mode="multiplicative",
        uncertainty_samples=500,
        interval_width=0.8
    )
    m.add_country_holidays(country_name="US")
    m.add_regressor("is_weekend")

    # Fit the model on the training portion
    m.fit(train[["ds", "y", "is_weekend"]])
    models_by_borough[boro] = m

    # Evaluate on the held-out 20% of data
    fcst = m.predict(test[["ds", "is_weekend"]])
    eval_df = test[["ds", "y"]].merge(fcst[["ds", "yhat"]], on="ds", how="left")

    rmse, mae, mape = evaluate_forecast(eval_df["y"].values, eval_df["yhat"].values)
    results.append({"borough": boro, "rmse": rmse, "mae": mae, "mape": mape})

# ---------- Print Evaluation Summary ------------------------------------------
if results:
    summary = pd.DataFrame(results).sort_values("mape")
    print("\n=== SUMMARY (per-borough) ===")
    print(summary.to_string(index=False))

# ---------- Forecast Specific Future Day --------------------------------------
target_date = pd.Timestamp("2025-11-01")  # Change this date as needed
rows = []

# Predict for each trained borough model
for boro, m in models_by_borough.items():
    future = pd.DataFrame({"ds": [target_date]})
    future["is_weekend"] = (future["ds"].dt.dayofweek >= 5).astype(int)
    f = m.predict(future).loc[:, ["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # Cap lower bound at 0 (no negative collisions)
    f["yhat_lower"] = f["yhat_lower"].clip(lower=0)
    f["yhat"]        = f["yhat"].clip(lower=0)
    f["yhat_upper"]  = f["yhat_upper"].clip(lower=0)

    f.insert(0, "borough", boro)
    rows.append(f)

# Combine and print per-borough and citywide forecasts
if rows:
    day_forecasts = pd.concat(rows, ignore_index=True).sort_values("borough")
    print(f"\n=== FORECASTS for {target_date.date()} (per borough) ===")
    print(day_forecasts.to_string(index=False))

    # Summed citywide total (sum of central and interval estimates)
    total = day_forecasts[["yhat", "yhat_lower", "yhat_upper"]].sum()
    print(f"\nCITYWIDE TOTAL on {target_date.date()}: "
          f"yhat={total['yhat']:.1f}, lower={total['yhat_lower']:.1f}, upper={total['yhat_upper']:.1f}")
