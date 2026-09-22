# Corporación Favorita Grocery Sales Forecasting

Kaggle Competition: [favorita-grocery-sales-forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

Predict unit sales for thousands of items sold across Corporación Favorita store locations in Ecuador using gradient-boosted decision trees (LightGBM) with lag and rolling window features.

---

## Goal & Evaluation Metric

* **Goal**: Forecast daily unit sales across thousands of item-store combinations for a 16-day future test horizon (August 16 – August 31, 2017).
* **Metric**: **NWRMSLE** (Normalized Weighted Root Mean Squared Logarithmic Error). Perishable items are weighted **1.25×** while non-perishable items are weighted **1.0×**.

---

## Project Structure

```text
favorita-sales-forecast/
├── data/
│   ├── raw/            # Raw Kaggle CSVs (uncompressed)
│   └── processed/      # Engineered parquet feature sets (auto-generated)
├── notebooks/
│   └── 01_eda.ipynb    # Exploratory Data Analysis
├── src/
│   ├── config.py       # Configuration paths & LightGBM hyperparameters
│   ├── features.py     # Vectorized feature generation (lags, rolling stats)
│   ├── metrics.py      # Custom NWRMSLE scoring function & sample weights
│   ├── predict.py      # Inference script to produce submission.csv
│   ├── train.py        # Model training loop with early stopping & hold-out CV
│   └── visualize.py    # Generates diagnostic plots (actual vs predicted, errors)
├── models/             # Trained model artifacts (.pkl)
├── reports/            # Validation metrics (cv_results.json) and diagnostic plots
└── submissions/        # Generated submission files
```

---

## Quickstart

### 1. Environment Setup

```bash
git clone [https://github.com/your-username/favorita-sales-forecast.git](https://github.com/your-username/favorita-sales-forecast.git)
cd favorita-sales-forecast
pip install -r requirements.txt
```

*(Key libraries: `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `pyarrow`, `matplotlib`, `seaborn`)*

### 2. Data Preparation

```bash
# 1. Download dataset via Kaggle CLI
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/

# 2. Extract 7z archives directly into data/raw/
7z e data/raw/*.7z -odata/raw/
```

### 3. Training & Validation

```bash
# Run feature engineering and model training
python src/train.py
# Or on Windows:
py .\src\train.py
```

*Note: Training automatically filters raw data starting from `2017-01-01` via chunked ingestion to keep RAM consumption manageable.*

### 4. Diagnostics & Visualization

```bash
python src/visualize.py
```
Outputs validation trend graphs, residual spreads, and top feature splits to `reports/model_diagnostics.png`.

### 5. Generate Submission

```bash
# Generate submission.csv
python src/predict.py

# Submit to Kaggle
kaggle competitions submit -c favorita-grocery-sales-forecasting -f submissions/submission.csv -m "LightGBM baseline with lag and rolling features"
```

---

## Approach

| Step | Detail |
|---|---|
| **Target Transformation** | `log1p(unit_sales.clip(lower=0))` to align training directly with the RMSLE loss objective and clip negative return units to zero. |
| **Feature Engineering** | <ul><li>**Lags**: 1, 7, 14, and 28-day lags built with vectorized unstacked pivots.</li><li>**Rolling Windows**: 7, 14, and 28-day moving means and standard deviations.</li><li>**Calendar Signals**: Day of week, day of month, day of year, ISO week, month, year, weekend flag.</li><li>**External Indicators**: Forward-filled crude oil prices, daily store transactions, and regional/national holiday flags.</li></ul> |
| **Model** | `LGBMRegressor` with early stopping on root mean squared error. |
| **Validation** | Time-series walk-forward hold-out using the last 16 days of available training data. |

---

## Results

| Evaluation Split | Metric (NWRMSLE) | Notes |
|---|---|---|
| **Local Validation (Last 16 Days)** | **0.48412** | Converged at round 838 via early stopping. |
| **Kaggle Public Leaderboard** | *Pending submission* | Baseline submission ready in `submissions/submission.csv`. |
