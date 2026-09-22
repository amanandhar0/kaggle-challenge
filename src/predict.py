"""
predict.py — loads the trained model, makes predictions on X_test,
and creates the Kaggle submission CSV.

Usage:
    python src/predict.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PROC, MODELS_DIR, SUBS_DIR, TEST_CSV


def main():
    print("Loading test features and IDs …")
    X_test  = pd.read_parquet(DATA_PROC / "X_test.parquet")
    test_df = pd.read_csv(TEST_CSV, usecols=["id"])

    model_path = MODELS_DIR / "lgb_model.pkl"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Run train.py first.")
        sys.exit(1)

    print("Loading model …")
    model = joblib.load(model_path)

    print("Predicting …")
    pred_log = model.predict(X_test)

    # Convert from log1p space back to normal space: expm1(y) = e^y - 1
    # also max(0) because sales cannot be negative
    pred_unit = np.expm1(np.maximum(pred_log, 0.0))

    test_df["unit_sales"] = pred_unit

    sub_path = SUBS_DIR / "submission.csv"
    print(f"Saving submission to {sub_path} …")
    test_df.to_csv(sub_path, index=False, float_format="%.4f")

    print(test_df.head(10))
    print("\nDone. Ready for kaggle submission!")


if __name__ == "__main__":
    main()
