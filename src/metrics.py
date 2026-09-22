"""
metrics.py — NWRMSLE (Normalized Weighted Root Mean Squared Logarithmic Error).

Perishable items (items.perishable == 1) carry weight 1.25;
non-perishable items carry weight 1.00.

NWRMSLE = sqrt( sum(w_i * (log1p(ŷ_i) - log1p(y_i))²) / sum(w_i) )
"""
import numpy as np
import pandas as pd


def nwrmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Compute NWRMSLE.

    Parameters
    ----------
    y_true   : actual unit sales (clipped to ≥ 0)
    y_pred   : predicted unit sales (clipped to ≥ 0)
    weights  : per-sample weights (1.0 or 1.25)
    """
    y_true = np.maximum(y_true, 0.0)
    y_pred = np.maximum(y_pred, 0.0)

    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    score = np.sqrt(np.sum(weights * log_diff ** 2) / np.sum(weights))
    return float(score)


def make_weights(item_nbr: pd.Series, perishable_map: dict) -> np.ndarray:
    """
    Build a weight array from item numbers using a {item_nbr: is_perishable} dict.
    """
    perishable = item_nbr.map(perishable_map).fillna(0).astype(int)
    return np.where(perishable == 1, 1.25, 1.0)
