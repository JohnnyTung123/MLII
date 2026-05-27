"""
subperiod_eval.py — honest sub-period performance reporting.

A single full-sample Sharpe hides the IC decay you measured (0.076 → 0.036).
This module breaks any monthly-return series into decades and trailing windows
so you can see what the strategy earns RECENTLY, which is the honest proxy for
what it would earn going forward.

Usage:
    from subperiod_eval import subperiod_table
    print(subperiod_table(returns_df, col="net_qspread").to_string())
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 6 or s.std() == 0:
        return np.nan
    return float(s.mean() / s.std() * np.sqrt(12))


def _ann_ret(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() * 12) if len(s) else np.nan


def _max_dd(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    cum = (1 + s).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def subperiod_table(
    returns: pd.DataFrame,
    col: str = "net_qspread",
    eom_col: str = "eom",
) -> pd.DataFrame:
    """
    Return a table of Sharpe / ann.return / maxDD / win-rate / mean-IC by:
      - each decade present in the data
      - trailing 60 and 120 months
      - full sample
    """
    df = returns.copy()
    df[eom_col] = pd.to_datetime(df[eom_col])
    df = df.sort_values(eom_col)
    s = df.set_index(eom_col)[col]

    rows = []

    def _row(label, sub, ic_sub=None):
        rows.append({
            "period":    label,
            "n_months":  int(sub.notna().sum()),
            "sharpe":    round(_sharpe(sub), 3),
            "ann_ret":   round(_ann_ret(sub), 4),
            "max_dd":    round(_max_dd(sub), 4),
            "win_rate":  round(float((sub.dropna() > 0).mean()), 4) if sub.notna().any() else np.nan,
            "mean_ic":   round(float(ic_sub.mean()), 4) if ic_sub is not None and len(ic_sub) else np.nan,
        })

    has_ic = "ic_spearman" in df.columns
    ic_series = df.set_index(eom_col)["ic_spearman"] if has_ic else None

    # by decade
    for dec in sorted({(y // 10) * 10 for y in s.index.year}):
        mask = (s.index.year >= dec) & (s.index.year < dec + 10)
        _row(f"{dec}s", s[mask], ic_series[mask] if has_ic else None)

    # trailing windows
    for w in (60, 120):
        _row(f"last_{w}m", s.iloc[-w:], ic_series.iloc[-w:] if has_ic else None)

    # full sample
    _row("FULL", s, ic_series if has_ic else None)

    return pd.DataFrame(rows).set_index("period")
