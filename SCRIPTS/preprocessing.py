"""
This script preprocesses raw NYC motor vehicle collision data for time-series modeling.
It combines date and time into a single timestamp, removes irrelevant columns,
cleans missing location data, and standardizes column names for consistency.

Input:
    - Raw collision dataset loaded in the dataset directory.
          ../DATA/collisions_raw.csv

Processing steps:
    1. Combine 'CRASH DATE' and 'CRASH TIME' into a unified timestamp column.
    2. Remove irrelevant or redundant columns.
    3. Drop rows missing location or timestamp.
    4. Rename and reorder columns for a standardized schema.
    5. Convert persons_injured / persons_killed to integers.

Output:
    - `collisions_cleaned.csv` containing the following columns:
          - `collision_id`: unique identifier for the collision
          - `timestamp`: combined crash date and time
          - `location`: text location (in coordinate pair) of the crash
          - `persons_injured`: number of persons injured
          - `persons_killed`: number of persons killed
"""

import pandas as pd
from pathlib import Path

# ----- File paths -----
RAW_PATH = Path("../DATA/collisions_raw.csv")
OUTPUT_PATH = Path("../DATA/collisions_cleaned.csv")

# ----- Load raw data -----
df = pd.read_csv(RAW_PATH)

# ----- Combine crash date and time into a unified timestamp -----
df["timestamp"] = df["CRASH DATE"] + " " + df["CRASH TIME"]
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ----- Drop unnecessary columns -----
columns_to_drop = [
    "NUMBER OF PEDESTRIANS INJURED",
    "NUMBER OF PEDESTRIANS KILLED",
    "NUMBER OF CYCLIST INJURED",
    "NUMBER OF CYCLIST KILLED",
    "NUMBER OF MOTORIST INJURED",
    "NUMBER OF MOTORIST KILLED",
    "BOROUGH",
    "LATITUDE",
    "LONGITUDE",
    "CRASH DATE",
    "CRASH TIME",
]
df = df.drop(columns=columns_to_drop)

# ----- Remove rows with missing location -----
df = df.dropna(subset=["LOCATION"])

# ----- Rename and reorder columns -----
df = df.rename(
    columns={
        "COLLISION_ID": "collision_id",
        "LOCATION": "location",
        "NUMBER OF PERSONS INJURED": "persons_injured",
        "NUMBER OF PERSONS KILLED": "persons_killed",
    }
)
df = df[["collision_id", "timestamp", "location", "persons_injured", "persons_killed"]]

# ----- Convert injury and fatality columns to integers -----
df["persons_injured"] = pd.to_numeric(df["persons_injured"], errors="coerce").fillna(0).astype("int64")
df["persons_killed"] = pd.to_numeric(df["persons_killed"], errors="coerce").fillna(0).astype("int64")

# ----- Export cleaned dataset -----
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
