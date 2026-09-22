"""
train.py — trains a LightGBM model on engineered features with walk-forward CV.

Usage:
    python src/train.py

Outputs:
    models/lgb_model.pkl        — trained model
    reports/cv_results.json     — validation NWRMSLE score
    data/processed/X_train.parquet
    data/processed/X_test.parquet
    data/processed/y_train.parquet
"""
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TRAIN_CSV, TEST_CSV, LGB_PARAMS, EARLY_STOPPING_ROUNDS,
    VALIDATION_DAYS, DATA_PROC, MODELS_DIR, REPORTS_DIR,
)
from features import build_features
from metrics import nwrmsle, make_weights


# ─────────────────────────────────────────────────────────────────────────────

def load_train_recent(path: Path, dtypes: dict, start_date: str = "2017-01-01") -> pd.DataFrame:
    """
    Loads training data filtering for dates >= start_date.
    Reads in chunks to ensure low peak memory footprint.
    """
    print(f"  Reading {path.name} (filtering >= {start_date}) …")
    chunks = []
    chunk_size = 10_000_000
    
    for chunk in pd.read_csv(path, dtype=dtypes, parse_dates=["date"], chunksize=chunk_size, low_memory=False):
        chunk = chunk[chunk["date"] >= start_date]
        if not chunk.empty:
            chunks.append(chunk)
            
    df = pd.concat(chunks, ignore_index=True)
    return df


def load_raw(path: Path, **kwargs) -> pd.DataFrame:
    print(f"  Reading {path.name} …")
    return pd.read_csv(path, parse_dates=["date"], **kwargs)


def save_processed(X_train, y_train, X_test):
    print("Saving processed features to parquet …")
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(DATA_PROC / "X_train.parquet", index=False)
    y_train.to_frame("log_sales").to_parquet(DATA_PROC / "y_train.parquet", index=False)
    X_test.to_parquet(DATA_PROC / "X_test.parquet", index=False)


def train_model(X_train, y_train, X_val, y_val, weights_val):
    """Fit LightGBM with early stopping on the validation split."""
    params = dict(LGB_PARAMS)
    n_estimators = params.pop("n_estimators")

    model = lgb.LGBMRegressor(**params, n_estimators=n_estimators)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    return model


def main():
    t0 = time.time()
    print("=" * 60)
    print("Favorita Sales Forecasting — Training")
    print("=" * 60)

    # ── 1. Load raw data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading raw CSV files …")
    dtypes = {
        "store_nbr":    "int8",
        "item_nbr":     "int32",
        "unit_sales":   "float32",
        "onpromotion":  "object",
    }
    # Load 2017+ data to prevent out-of-memory errors
    train_df = load_train_recent(TRAIN_CSV, dtypes=dtypes, start_date="2017-01-01")
    test_df  = load_raw(TEST_CSV)

    print(f"  Filtered Train shape: {train_df.shape}  |  Test shape: {test_df.shape}")

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/4] Building features …")
    # Preserve date column before feature construction to ensure split consistency
    dates_train = train_df["date"].copy().reset_index(drop=True)
    items_train = train_df["item_nbr"].copy().reset_index(drop=True)

    X_train, y_train, X_test, feature_cols, perishable_map = build_features(
        train_df, test_df
    )
    save_processed(X_train, y_train, X_test)

    # ── 3. Time-series split: last VALIDATION_DAYS rows as hold-out ────────────
    print(f"\n[3/4] Splitting train / validation (last {VALIDATION_DAYS} days) …")

    split_date = dates_train.max() - pd.Timedelta(days=VALIDATION_DAYS)

    is_val = (dates_train > split_date).values
    is_tr  = ~is_val

    X_tr, y_tr = X_train.iloc[is_tr], y_train.iloc[is_tr]
    X_val, y_val = X_train.iloc[is_val], y_train.iloc[is_val]

    # Build per-sample weights for validation scoring
    item_col_val = items_train.iloc[is_val].reset_index(drop=True)
    w_val = make_weights(item_col_val, perishable_map)

    print(f"  Train rows : {len(X_tr):,}")
    print(f"  Val rows   : {len(X_val):,}")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("\n[4/4] Training LightGBM …")
    model = train_model(X_tr, y_tr, X_val, y_val, w_val)

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    val_pred_log = model.predict(X_val)
    val_pred     = np.expm1(np.maximum(val_pred_log, 0.0))
    val_true     = np.expm1(y_val.values)

    score = nwrmsle(val_true, val_pred, w_val)
    print(f"\n  ✓ Validation NWRMSLE: {score:.5f}")

    # Feature importance (top 20)
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  Top 20 features:")
    print(fi.head(20).to_string())

    # ── 6. Save artifacts ──────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n  Model saved → {model_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "val_nwrmsle": round(score, 6),
        "best_iteration": int(model.best_iteration_),
        "feature_importances": fi.head(30).to_dict(),
    }
    report_path = REPORTS_DIR / "cv_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {report_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()