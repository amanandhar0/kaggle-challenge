"""
features.py — full feature engineering pipeline for Favorita.

Call `build_features(train_df, test_df)` which returns
(X_train, y_train, X_test, feature_cols, perishable_map).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STORES_CSV, ITEMS_CSV, TRANSACTIONS_CSV,
    OIL_CSV, HOLIDAYS_CSV, LAG_DAYS, ROLL_WINDOWS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_side_tables():
    """Load all auxiliary CSVs and return as a dict."""
    stores  = pd.read_csv(STORES_CSV)
    items   = pd.read_csv(ITEMS_CSV)
    txn     = pd.read_csv(TRANSACTIONS_CSV, parse_dates=["date"])
    oil     = pd.read_csv(OIL_CSV,          parse_dates=["date"])
    hol     = pd.read_csv(HOLIDAYS_CSV,     parse_dates=["date"])
    return dict(stores=stores, items=items, txn=txn, oil=oil, hol=hol)


def _process_holidays(hol: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with boolean flags:
      holiday_national, holiday_regional, holiday_local, holiday_transferred,
      holiday_bridge, holiday_work_day
    """
    # Only keep rows that are actual holidays / events
    hol = hol[hol["transferred"] == False].copy()   # noqa: E712

    national  = hol[hol["locale"] == "National"][["date"]].drop_duplicates()
    regional  = hol[hol["locale"] == "Regional"][["date"]].drop_duplicates()
    local_hol = hol[hol["locale"] == "Local"][["date"]].drop_duplicates()
    bridge    = hol[hol["type"] == "Bridge"][["date"]].drop_duplicates()
    work_day  = hol[hol["type"] == "Work Day"][["date"]].drop_duplicates()
    transfer  = hol[hol["type"] == "Transfer"][["date"]].drop_duplicates()

    national["holiday_national"]    = True
    regional["holiday_regional"]    = True
    local_hol["holiday_local"]      = True
    bridge["holiday_bridge"]        = True
    work_day["holiday_work_day"]    = True
    transfer["holiday_transferred"] = True

    result = national.merge(regional,  on="date", how="outer") \
                     .merge(local_hol, on="date", how="outer") \
                     .merge(bridge,    on="date", how="outer") \
                     .merge(work_day,  on="date", how="outer") \
                     .merge(transfer,  on="date", how="outer")

    bool_cols = [c for c in result.columns if c != "date"]
    result[bool_cols] = result[bool_cols].fillna(False)
    result = result.set_index("date")
    return result


def _fill_oil(oil: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill oil prices for every date in range."""
    oil = oil.set_index("date").reindex(all_dates)["dcoilwtico"]
    return oil.ffill().bfill()


# ── Main feature builder ───────────────────────────────────────────────────────

def build_features(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
):
    """
    Parameters
    ----------
    train_df : raw train.csv loaded as DataFrame (with 'date' as datetime)
    test_df  : raw test.csv loaded as DataFrame  (with 'date' as datetime)

    Returns
    -------
    X_train        : pd.DataFrame  — features for training rows
    y_train        : pd.Series     — log1p-clipped target
    X_test         : pd.DataFrame  — features for test rows
    feature_cols   : list[str]     — ordered feature column names
    perishable_map : dict          — {item_nbr: is_perishable (0/1)}
    """
    print("Loading side tables …")
    side = _load_side_tables()

    # ── 1. Clip negative sales; log1p transform target ─────────────────────────
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df["unit_sales"] = train_df["unit_sales"].clip(lower=0)
    train_df["log_sales"]  = np.log1p(train_df["unit_sales"])

    promo_map = {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}

    if "onpromotion" in train_df.columns:
        train_df["onpromotion"] = train_df["onpromotion"].map(promo_map).fillna(0).astype("float32")

    if "onpromotion" in test_df.columns:
        test_df["onpromotion"] = test_df["onpromotion"].map(promo_map).fillna(0).astype("float32")

    # ── 2. Merge stores & items metadata ──────────────────────────────────────
    train_df = train_df.merge(side["stores"], on="store_nbr", how="left")
    train_df = train_df.merge(side["items"],  on="item_nbr",  how="left")
    test_df  = test_df.merge(side["stores"],  on="store_nbr", how="left")
    test_df  = test_df.merge(side["items"],   on="item_nbr",  how="left")

    # Perishable map before we go further
    perishable_map = (
        side["items"].set_index("item_nbr")["perishable"].to_dict()
    )

    # ── 3. Calendar features ───────────────────────────────────────────────────
    for df in (train_df, test_df):
        df["dayofweek"]  = df["date"].dt.dayofweek
        df["dayofmonth"] = df["date"].dt.day
        df["dayofyear"]  = df["date"].dt.dayofyear
        df["week"]       = df["date"].dt.isocalendar().week.astype(int)
        df["month"]      = df["date"].dt.month
        df["year"]       = df["date"].dt.year
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # ── 4. Oil price ───────────────────────────────────────────────────────────
    all_dates = pd.date_range(
        start=train_df["date"].min(),
        end=test_df["date"].max(),
        freq="D",
    )
    oil_series = _fill_oil(side["oil"], all_dates)
    oil_map    = oil_series.to_dict()

    for df in (train_df, test_df):
        df["oil_price"] = df["date"].map(oil_map)

    # ── 5. Transactions (daily txn count per store) ────────────────────────────
    txn_map = (
        side["txn"].set_index(["date", "store_nbr"])["transactions"]
        .to_dict()
    )
    for df in (train_df, test_df):
        df["transactions"] = [
            txn_map.get((d, s), np.nan)
            for d, s in zip(df["date"], df["store_nbr"])
        ]

    # ── 6. Holiday flags ───────────────────────────────────────────────────────
    hol_flags = _process_holidays(side["hol"])
    for df in (train_df, test_df):
        for col in hol_flags.columns:
            df[col] = df["date"].map(hol_flags[col]).fillna(False).astype(int)

    # ── 7. Lag & rolling features ──────────────────────────────────────────────
    print("Computing lag / rolling features (this takes a minute) …")

    lookback = max(max(LAG_DAYS), max(ROLL_WINDOWS))

    pivot = (
        train_df[["date", "store_nbr", "item_nbr", "unit_sales"]]
        .set_index(["date", "store_nbr", "item_nbr"])["unit_sales"]
        .unstack(["store_nbr", "item_nbr"])
        .sort_index()
    )

    lag_frames  = {}
    roll_frames = {}

    for lag in tqdm(LAG_DAYS, desc="Lags"):
        lag_frames[f"lag_{lag}"] = pivot.shift(lag)

    for win in tqdm(ROLL_WINDOWS, desc="Rolling"):
        shifted = pivot.shift(1)  # avoid leakage: don't include current day
        roll_frames[f"roll_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
        roll_frames[f"roll_std_{win}"]  = shifted.rolling(win, min_periods=1).std().fillna(0)

    def _attach_lags(df: pd.DataFrame) -> pd.DataFrame:
        """Attach precomputed lag/roll columns to a df by (date, store_nbr, item_nbr)."""
        df = df.copy()
        idx = list(zip(df["date"], df["store_nbr"], df["item_nbr"]))

        for name, frame in {**lag_frames, **roll_frames}.items():
            vals = []
            for d, s, i in idx:
                try:
                    vals.append(frame.at[d, (s, i)])
                except KeyError:
                    vals.append(np.nan)
            df[name] = vals
        return df

    print("Attaching lag features to train …")
    train_df = _attach_lags(train_df)
    print("Attaching lag features to test …")
    test_df  = _attach_lags(test_df)

    # ── 8. Encode categoricals ─────────────────────────────────────────────────
    cat_cols = ["city", "state", "type", "family"]

    for col in cat_cols:
        if col in train_df.columns:
            codes, _ = pd.factorize(
                pd.concat([train_df[col], test_df[col]], ignore_index=True)
            )
            n_train = len(train_df)
            train_df[col] = codes[:n_train]
            test_df[col]  = codes[n_train:]

    # ── 9. Assemble feature matrix ─────────────────────────────────────────────
    feature_cols = (
        # identity / meta
        ["store_nbr", "item_nbr", "onpromotion", "perishable",
         "cluster", "city", "state", "type", "family",
        # calendar
         "dayofweek", "dayofmonth", "dayofyear", "week",
         "month", "year", "is_weekend",
        # external
         "oil_price", "transactions",
        # holiday flags
        ] +
        [c for c in train_df.columns if c.startswith("holiday_")] +
        # lag / rolling
        [f"lag_{l}" for l in LAG_DAYS] +
        [f"roll_mean_{w}" for w in ROLL_WINDOWS] +
        [f"roll_std_{w}"  for w in ROLL_WINDOWS]
    )

    # Keep only cols that actually exist
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df["log_sales"]
    X_test  = test_df[feature_cols]

    print(f"Feature matrix: {X_train.shape[1]} features, "
          f"{len(X_train):,} train rows, {len(X_test):,} test rows.")

    return X_train, y_train, X_test, feature_cols, perishable_map