# Forecasting NYC Motor Vehicle Collisions


## 1. Software and Platform
- **Platform Used:** Mac was used to write and run scripts. The scripts can be run across platforms.
- **Software Used**: Python 3
 
- **Add-on Packages:**  
  - `pandas` - data loading and manipulation
  - `seaborn` – visualizing confusion matrices


## 2. Documentation Map
The hierarchy of folders and files contained in this project are as follows:

```text
DS-4002-Project2
├── DATA
│   ├── collisions_cleaned.csv.zip
│   └── collisions_raw.csv.zip
├── OUTPUT
│   ├── daily_baseline_model_figure.png
│   ├── daily_prophet_model.png
│   ├── hourly_baseline_model.png
│   └── hourly_prophet_model.png
├── SCRIPTS
│   ├── 1_BASELINE_daily.py
│   ├── 1_BASELINE_hourly.py
│   ├── 1_SARIMA_daily.py
|   ├── 2_SARIMA_hourly.py
|   ├── 3_Prophet_daily.py
|   ├── 3_Prophet_hourly.py
│   └── preprocessing.py
├── LICENSE.md
├── README.md
└── collisions_filtered.csv

```

## 3. Reproducing Our Results
  1. **Set up Python and install required add-on packages**
     - Clone this repository: https://github.com/evelynmaxwell/DS-4002-Project2/
     - Ensure you have Python 3 installed on your system.
     - See section 1 for packages needed.
  2. **Prepare the data**
     - If you wish to preprocess the raw data yourself, navigate to the `SCRIPTS` folder and run the `preprocessing.py` file, which will save the preprocessed data in a new `` file within the `DATA` folder.
     - Otherwise, download the `collisions_cleaned.csv.zip` file and unzip it. Then, manually add the `collisions_cleaned.csv` file to the DATA folder in your IDE.
  4. **Run model scripts**
     - Navigate to the `SCRIPTS` folder.
     - Each script corresponds to a model (
  5. **Download and view outputs** 
     -

     
