"""
config.py — central place for paths and hyperparameters.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
SUBS_DIR    = ROOT / "submissions"
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_PROC, MODELS_DIR, SUBS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw file paths ─────────────────────────────────────────────────────────────
TRAIN_CSV         = DATA_RAW / "train.csv"
TEST_CSV          = DATA_RAW / "test.csv"
STORES_CSV        = DATA_RAW / "stores.csv"
ITEMS_CSV         = DATA_RAW / "items.csv"
TRANSACTIONS_CSV  = DATA_RAW / "transactions.csv"
OIL_CSV           = DATA_RAW / "oil.csv"
HOLIDAYS_CSV      = DATA_RAW / "holidays_events.csv"
SAMPLE_SUB_CSV    = DATA_RAW / "sample_submission.csv"

# ── Feature engineering knobs ──────────────────────────────────────────────────
LAG_DAYS        = [1, 7, 14, 28]          # sales lag windows
ROLL_WINDOWS    = [7, 14, 28]             # rolling mean/std windows
VALIDATION_DAYS = 16                      # size of the held-out local CV window

# Competition test period: 2017-08-16 … 2017-08-31 (16 days)
TEST_START = "2017-08-16"
TEST_END   = "2017-08-31"

# ── LightGBM hyperparameters ───────────────────────────────────────────────────
LGB_PARAMS = {
    "objective":        "regression_l2",
    "metric":           "rmse",
    "learning_rate":    0.05,
    "num_leaves":       511,
    "min_child_samples":20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "lambda_l2":        1.0,
    "n_estimators":     2000,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

EARLY_STOPPING_ROUNDS = 50
