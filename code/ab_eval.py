"""
ab_eval.py — evaluate and plot A/B/C results from ab_harness.py.

Reads the three monthly_returns_*.csv files written by ab_harness.py and
produces:
  1. Four-panel comparison chart  (cumulative return, rolling Sharpe, IC, turnover)
  2. Sub-period Sharpe table      (by decade + last 60/120 months)
  3. Full performance summary     (gross + net, all metrics side-by-side)

Usage:
    python code/ab_eval.py
    python code/ab_eval.py --col net_qspread   # default
    python code/ab_eval.py --col qspread       # gross
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subperiod_eval import subperiod_table

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).parent.parent
OUT_DIR  = _ROOT / "data" / "backtest" / "xgboost_ab"

CFGS = {
    "A_decile_ew":          ("A — Decile EW (baseline)",  "steelblue",   "-"),
    "B_rankw_vw":           ("B — Rank-weight VW",         "darkorange",  "--"),
    "C_rankw_vw_balanced":  ("C — Rank-weight VW + balance", "seagreen", "-."),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> dict[str, pd.DataFrame]:
    dfs = {}
    for cfg in CFGS:
        path = OUT_DIR / f"monthly_returns_{cfg}.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping {cfg}")
            continue
        df = pd.read_csv(path)
        df["eom"] = pd.to_datetime(df["eom"])
        df = df.sort_values("eom").reset_index(drop=True)
        dfs[cfg] = df
    if not dfs:
        raise FileNotFoundError(
            f"No monthly_returns_*.csv files found in {OUT_DIR}. "
            "Run ab_harness.py first."
        )
    return dfs


# ══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE STATS
# ══════════════════════════════════════════════════════════════════════════════

def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 6:
        return {}
    ann_ret = float(s.mean() * 12)
    ann_vol = float(s.std() * np.sqrt(12))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + s).cumprod()
    max_dd  = float(((cum - cum.cummax()) / cum.cummax()).min())
    return {
        "ann_ret":  round(ann_ret, 4),
        "ann_vol":  round(ann_vol, 4),
        "sharpe":   round(sharpe,  3),
        "max_dd":   round(max_dd,  4),
        "win_rate": round(float((s > 0).mean()), 4),
        "n_months": len(s),
    }


def performance_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cfg, df in dfs.items():
        label = CFGS[cfg][0]
        g = _stats(df["qspread"])
        n = _stats(df["net_qspread"])
        ic = df["ic_spearman"].dropna()
        icir = float(ic.mean() / ic.std() * np.sqrt(12)) if ic.std() > 0 else np.nan
        rows.append({
            "config":          label,
            "gross_sharpe":    g.get("sharpe"),
            "gross_ann_ret":   g.get("ann_ret"),
            "gross_ann_vol":   g.get("ann_vol"),
            "gross_max_dd":    g.get("max_dd"),
            "gross_win_rate":  g.get("win_rate"),
            "net_sharpe":      n.get("sharpe"),
            "net_ann_ret":     n.get("ann_ret"),
            "net_ann_vol":     n.get("ann_vol"),
            "net_max_dd":      n.get("max_dd"),
            "net_win_rate":    n.get("win_rate"),
            "mean_ic":         round(float(ic.mean()), 4),
            "icir":            round(icir, 3),
            "n_months":        g.get("n_months"),
        })
    return pd.DataFrame(rows).set_index("config")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FOUR-PANEL PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(dfs: dict[str, pd.DataFrame], col: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        f"A/B/C Portfolio Construction Comparison  ({'net' if 'net' in col else 'gross'} Q-spread)",
        fontsize=13, y=0.99,
    )

    # Panel 1 — cumulative return
    ax = axes[0]
    for cfg, df in dfs.items():
        label, color, ls = CFGS[cfg]
        cum = (1 + df[col]).cumprod()
        ax.plot(df["eom"], cum, label=label, color=color, ls=ls, lw=1.6)
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Q-Spread Return")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 2 — rolling 12-month Sharpe
    ax = axes[1]
    for cfg, df in dfs.items():
        label, color, ls = CFGS[cfg]
        roll     = df[col].rolling(12)
        roll_std = roll.std()
        rs       = np.where(roll_std > 0, roll.mean() / roll_std * np.sqrt(12), np.nan)
        ax.plot(df["eom"], rs, label=label, color=color, ls=ls, lw=1.4)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 3 — rolling IC (Spearman) — shared signal so identical, shown once
    ax = axes[2]
    first_cfg = next(iter(dfs))
    df0 = dfs[first_cfg]
    if "ic_spearman" in df0.columns:
        ic = df0["ic_spearman"]
        colors_ic = ["steelblue" if v >= 0 else "tomato" for v in ic.fillna(0)]
        ax.bar(df0["eom"], ic, width=25, color=colors_ic, alpha=0.45, label="Monthly IC")
        ax.plot(df0["eom"], ic.rolling(12).mean(), color="navy", lw=1.5,
                label="12m Rolling Mean IC")
        ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("IC (Spearman)")
    ax.set_title("Information Coefficient — same signal, same model for all configs")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Panel 4 — turnover
    ax = axes[3]
    for cfg, df in dfs.items():
        if "turnover" not in df.columns:
            continue
        label, color, ls = CFGS[cfg]
        ax.plot(df["eom"], df["turnover"].rolling(12).mean(),
                label=label, color=color, ls=ls, lw=1.4)
    ax.set_ylabel("Turnover (12m avg)")
    ax.set_title("Portfolio Turnover")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = out_dir / "ab_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PRINT SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════

def print_performance(dfs: dict[str, pd.DataFrame]) -> None:
    tbl = performance_table(dfs)
    sep = "=" * 90

    print(f"\n{sep}")
    print("  A/B/C FULL-SAMPLE PERFORMANCE SUMMARY")
    print(sep)
    print(f"  {'Metric':<22}", end="")
    for cfg in tbl.index:
        print(f"  {cfg[:28]:<28}", end="")
    print()
    print("-" * 90)

    metrics = [
        ("gross_sharpe",   "Gross Sharpe",    "{:>8.3f}"),
        ("gross_ann_ret",  "Gross Ann. Return","{:>8.2%}"),
        ("gross_ann_vol",  "Gross Ann. Vol",   "{:>8.2%}"),
        ("gross_max_dd",   "Gross Max DD",     "{:>8.2%}"),
        ("gross_win_rate", "Gross Win Rate",   "{:>8.2%}"),
        (None, "", ""),
        ("net_sharpe",     "Net Sharpe",       "{:>8.3f}"),
        ("net_ann_ret",    "Net Ann. Return",  "{:>8.2%}"),
        ("net_ann_vol",    "Net Ann. Vol",     "{:>8.2%}"),
        ("net_max_dd",     "Net Max DD",       "{:>8.2%}"),
        ("net_win_rate",   "Net Win Rate",     "{:>8.2%}"),
        (None, "", ""),
        ("mean_ic",        "Mean IC",          "{:>8.4f}"),
        ("icir",           "ICIR",             "{:>8.3f}"),
        ("n_months",       "N Months",         "{:>8d}"),
    ]
    for col_key, label, fmt in metrics:
        if col_key is None:
            print()
            continue
        print(f"  {label:<22}", end="")
        for cfg in tbl.index:
            val = tbl.loc[cfg, col_key]
            try:
                print(f"  {fmt.format(val):<28}", end="")
            except (TypeError, ValueError):
                print(f"  {'N/A':<28}", end="")
        print()
    print(sep)


def print_subperiod(dfs: dict[str, pd.DataFrame], col: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SUB-PERIOD NET SHARPE  (col={col})")
    print(sep)
    for cfg, df in dfs.items():
        label = CFGS[cfg][0]
        print(f"\n── {label} ──")
        print(subperiod_table(df, col=col).to_string())
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(col: str = "net_qspread") -> None:
    print(f"Loading results from {OUT_DIR} …")
    dfs = load_results()
    print(f"  Loaded {len(dfs)} configs: {list(dfs.keys())}")

    print_performance(dfs)
    print_subperiod(dfs, col=col)
    plot_comparison(dfs, col=col, out_dir=OUT_DIR)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and plot A/B/C results from ab_harness.py."
    )
    parser.add_argument(
        "--col", default="net_qspread",
        choices=["net_qspread", "qspread"],
        help="Return column to use for cumulative return and Sharpe plots (default: net_qspread).",
    )
    args = parser.parse_args()
    run_eval(col=args.col)