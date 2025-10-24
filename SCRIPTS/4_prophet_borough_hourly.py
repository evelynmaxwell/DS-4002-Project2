"""
Forecast Hourly Motor Vehicle Collisions per NYC Borough using Prophet
----------------------------------------------------------------------

Description:
    This script fits a Prophet model for a single NYC borough to forecast
    **hourly collision counts** for a specified date using recent historical data.
    It uses geographic bounding boxes to assign collisions to boroughs.
    The script prints a concise evaluation summary (RMSE, MAE, MAPE) and a clean
    24-hour forecast table for the selected borough.

Inputs:
    - ../DATA/collisions_cleaned.csv
        A cleaned dataset of NYC motor vehicle collisions containing:
          • 'timestamp' — datetime of each crash.
          • 'location'  — text coordinates formatted as "(latitude, longitude)".

Process:
    1. Parse and clean input data.
    2. Convert "(lat, lon)" strings into numeric latitude and longitude.
    3. Assign each record to one of five boroughs using coarse geographic bounding boxes.
    4. Aggregate data into hourly counts per borough and flag weekend hours.
    5. For the selected borough:
        • Cap the data to the most recent N years (default = 2).
        • Split the capped data into 80% train and 20% test sets.
        • Fit a Prophet model with tuned weekly and daily seasonalities.
        • Evaluate test RMSE, MAE, and MAPE for hourly accuracy.
        • Forecast the 24 hourly counts for a target date (default example: 2025-11-01).

Outputs:
    - A printed evaluation line showing RMSE, MAE, and MAPE for the capped test set.
    - A 24-row forecast table for the target date with columns:
          ['borough', 'ds', 'yhat'] (and 'yhat_lower', 'yhat_upper' if intervals enabled).

Notes:
    • Prophet uncertainty intervals are 80% by default and disabled when
      `uncertainty_samples = 0` for speed.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

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
df = df.dropna(subset=["timestamp", "location"]).sort_values("timestamp")

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

# Aggregate to hourly counts
df["ds"] = df["timestamp"].dt.floor("h")
hourly_all = (
    df.groupby(["borough", "ds"], as_index=False)
       .size()
       .rename(columns={"size": "y"})
)
hourly_all["is_weekend"] = (hourly_all["ds"].dt.dayofweek >= 5).astype(int)

# ----------------- Function: fit & forecast for a single borough/day ----------
def fit_forecast_hourly_for_day(
    borough: str,
    target_date,
    cap_years: int = 2,
    train_ratio: float = 0.8,
    uncertainty_samples: int = 0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit an hourly Prophet model for a single borough and forecast the 24 hours of target_date.

    Steps:
      1. Select the borough’s hourly series.
      2. Cap it to the last N years (default=2).
      3. Split capped data into 80/20 train/test.
      4. Fit Prophet with weekly + daily seasonality and weekend regressor.
      5. Predict the 24 hourly counts for target_date.
    """
    target_date = pd.Timestamp(target_date).normalize()
    series = hourly_all.loc[hourly_all["borough"] == borough, ["ds", "y", "is_weekend"]].sort_values("ds").reset_index(drop=True)
    if series.empty:
        raise ValueError(f"No hourly data found for borough '{borough}'.")

    # Cap to last N years before splitting
    cap_start = series["ds"].max() - pd.Timedelta(days=365 * cap_years)
    series = series.loc[series["ds"] >= cap_start].reset_index(drop=True)

    # Add regressors
    series["hour"] = series["ds"].dt.hour
    scaler = StandardScaler()
    series["hour_scaled"] = scaler.fit_transform(series[["hour"]])   # scale AFTER 2y filter
    series["is_weekend"]  = (series["ds"].dt.dayofweek >= 5).astype(int)

    train, test = time_series_split(series, train_ratio=train_ratio)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=50,
        changepoint_range=0.90,
        changepoint_prior_scale=0.20,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=0.01,
        seasonality_mode="additive",
        uncertainty_samples=uncertainty_samples,
        interval_width=0.8
    )
    m.add_country_holidays(country_name="US")
    m.add_seasonality(name='daily_24h', period=24/24, fourier_order=12)  # captures within-day pattern
    m.add_seasonality(name='weekly', period=7, fourier_order=15)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_regressor("is_weekend")
    m.add_regressor("hour_scaled")

    m.fit(train[["ds", "y", "is_weekend","hour_scaled"]])

    # Evaluate on test (hourly + daily-agg MAPE)
    fcst_test = m.predict(test[["ds", "is_weekend", "hour_scaled"]])
    eval_df = test[["ds", "y"]].merge(fcst_test[["ds", "yhat"]], on="ds", how="left")
    rmse, mae, mape = evaluate_forecast(eval_df["y"].values, eval_df["yhat"].values)

    tmp = eval_df.copy()
    tmp["date"] = tmp["ds"].dt.date
    agg = tmp.groupby("date", as_index=False).agg(y_day=("y", "sum"),
                                                 yhat_day=("yhat", "sum"))
    daily_den = agg["y_day"].replace(0, np.nan)
    mape_daily = (np.abs(agg["y_day"] - agg["yhat_day"]) / daily_den).mean() * 100

    print(f"\nHOURLY PROPHET MODEL — {borough}")
    print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE (hourly): {mape:.2f}% | MAPE (daily agg): {mape_daily:.2f}%")

    # Build the 24-hour forecast frame for the requested date (using same scaler)
    hours = pd.date_range(start=target_date, periods=24, freq="h")
    future = pd.DataFrame({"ds": hours})
    future["is_weekend"]  = (future["ds"].dt.dayofweek >= 5).astype(int)
    future["hour"]        = future["ds"].dt.hour
    future["hour_scaled"] = scaler.transform(future[["hour"]])

    fcst = m.predict(future)
    cols = ["ds", "yhat"] + [c for c in ("yhat_lower", "yhat_upper") if c in fcst.columns]
    fcst = fcst.loc[:, cols]
    fcst.insert(0, "borough", borough)

    # Print just the clean 24-row table
    print("\n24-HOUR FORECAST TABLE:")
    print(fcst.to_string(index=False))

    return fcst

# --------------------------- Example run ---------------------------------
fc = fit_forecast_hourly_for_day("Brooklyn", "2025-11-01", uncertainty_samples=0)
print(fc.to_string(index=False))
