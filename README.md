# Corporación Favorita Grocery Sales Forecasting

Kaggle competition: [favorita-grocery-sales-forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

## Goal
Predict unit sales for thousands of products across Favorita grocery stores in Ecuador.

## Metric
**NWRMSLE** — Normalized Weighted Root Mean Squared Logarithmic Error (perishable items weighted 1.25×, non-perishable 1.0×).

## Project Structure

```
favorita-sales-forecast/
├── data/
│   ├── raw/          ← place Kaggle CSVs here
│   └── processed/    ← engineered features (auto-generated)
├── notebooks/
│   └── 01_eda.ipynb  ← exploratory data analysis
├── src/
│   ├── config.py     ← paths & hyperparameters
│   ├── metrics.py    ← NWRMSLE implementation
│   ├── features.py   ← all feature engineering
│   ├── train.py      ← model training + CV
│   └── predict.py    ← generate submission
├── models/           ← saved model artifacts
├── submissions/      ← submission CSVs
└── reports/          ← CV scores, charts
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Download data from Kaggle and put CSVs in data/raw/
kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw/
unzip data/raw/favorita-grocery-sales-forecasting.zip -d data/raw/

# 2. Run feature engineering + training
python src/train.py

# 3. Generate submission
python src/predict.py
```

## Approach

| Step | Detail |
|---|---|
| **Features** | Lag sales (1/7/14/30d), rolling means/stds, day-of-week, month, year, day-of-year, `onpromotion`, oil price, store metadata (type, cluster, city), item metadata (family, perishable), holiday flags, transaction lags |
| **Model** | LightGBM with time-series CV (walk-forward validation) |
| **Target** | `log1p(unit_sales.clip(lower=0))` — clips negative returns to 0, log-transforms to match RMSLE loss |
| **Validation** | Last 16 days of train as held-out set (mirrors test window) |

## Results

| Split | NWRMSLE |
|---|---|
| Local CV (last 16d) | see `reports/cv_results.json` |
| Kaggle Public LB | submit to find out |
