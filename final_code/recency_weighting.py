"""
recency_weighting.py — exponential sample weights for the training set.

Addresses the IC decay you measured (Spearman IC 0.076 in the 1980s falling to
0.036 in the 2020s). The factors are being arbitraged away, so the most recent
cross-sections are more representative of what you'll trade next month than
1980s data. Exponential recency weighting tells XGBoost to fit the recent
regime more closely without discarding old data entirely.

HOW IT WORKS
------------
Each training row gets weight  w_i = halflife_decay ** (age_in_months_i),
normalised to mean 1.0 (so regularisation strength is unchanged). With a
half-life H (months), a row H months older than the newest training month gets
half the weight; 2H older gets a quarter; etc.

Pass the resulting array as sample_weight to XGBRegressor.fit(...).

CAUTION (read this)
-------------------
Recency weighting is a regime-adaptation tool, not a free Sharpe boost. It can:
  + help if the return-generating process drifts smoothly (likely here), but
  - HURT if recent data is just noisier, by overfitting the latest regime and
    raising turnover. It also shortens your effective training sample, which
    matters most in the early years when history is short.
Treat the half-life as a hyperparameter to validate, NOT a knob to crank.
Very short half-lives (<24m) usually overfit. Start at 120m (10y); compare
against no weighting (H = infinity) out-of-sample before adopting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exponential_sample_weights(
    eom_per_row: np.ndarray,
    half_life_months: float = 120.0,
) -> np.ndarray:
    """
    Compute exponential recency weights aligned to a stacked training array.

    Parameters
    ----------
    eom_per_row : array of datetime64 (or pandas Timestamps), one entry per
                  training row, giving the month each row belongs to. MUST be
                  in the same row order as X_train / y_train.
    half_life_months : months for the weight to halve. Larger = flatter (closer
                  to uniform). Use np.inf to disable (returns all ones).

    Returns
    -------
    sample_weight : float32 array, mean-normalised to 1.0.
    """
    if not np.isfinite(half_life_months) or half_life_months <= 0:
        return np.ones(len(eom_per_row), dtype=np.float32)

    eom = pd.to_datetime(pd.Series(eom_per_row))
    newest = eom.max()
    # age in (approximate) months: difference in days / 30.44
    age_months = (newest - eom).dt.days.values / 30.44

    decay = 0.5 ** (1.0 / half_life_months)      # per-month multiplier
    w = decay ** age_months
    w = w / w.mean()                              # mean-normalise to 1.0
    return w.astype(np.float32)


def build_eom_per_row(
    slices: dict,
    months: list,
    char_cols: list,
    start_year: int,
    end_year: int,
    y_col: str = "ret_exc_lead1m",
    ret_cap: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct the per-row eom labels for the SAME filtering used in
    _build_train_arrays_range, so weights line up exactly with X/y rows.

    This must mirror the dropna(subset=[y_col]) + abs(y) <= ret_cap filters
    in your train-array builder, in the same month order.
    """
    eom_list = []
    for m in months:
        if m.year < start_year:
            continue
        if m.year > end_year:
            break
        if m not in slices:
            continue
        frame = slices[m].dropna(subset=[y_col])
        frame = frame[frame[y_col].abs() <= ret_cap]
        if len(frame) == 0:
            continue
        eom_list.append(np.full(len(frame), np.datetime64(m), dtype="datetime64[ns]"))
    if not eom_list:
        return np.empty(0, dtype="datetime64[ns]")
    return np.concatenate(eom_list)
