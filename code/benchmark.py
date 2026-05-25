"""
USA market benchmark — JKP 2023 market_returns.csv.

Reads mkt_ew_lcl and mkt_vw_lcl for USA and computes the same performance
statistics produced by lasso.py and xg_boost_expanding_2.0.py.

Usage:
    python code/benchmark.py              # default: start=1986
    python code/benchmark.py --start 1990
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT    = Path(__file__).parent.parent
RAW      = _ROOT / "data" / "raw"
BACKTEST = _ROOT / "data" / "backtest" / "benchmark"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(start_year: int) -> pd.DataFrame:
    df = pd.read_csv(RAW / "market_returns.csv")
    df["eom"] = pd.to_datetime(df["eom"], format="%Y%m%d")
    df = (
        df[df["excntry"] == "USA"][["eom", "mkt_ew_lcl", "mkt_vw_lcl"]]
        .copy()
        .sort_values("eom")
        .reset_index(drop=True)
    )
    df = df[df["eom"].dt.year >= start_year].reset_index(drop=True)
    df = df.dropna(subset=["mkt_ew_lcl", "mkt_vw_lcl"]).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_performance(returns: pd.DataFrame) -> dict:
    def _stats(series: pd.Series, label: str) -> dict:
        s       = series.dropna()
        ann_ret = s.mean() * 12
        ann_vol = s.std()  * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum     = (1 + s).cumprod()
        max_dd  = float(((cum - cum.cummax()) / cum.cummax()).min())
        win_rt  = float((s > 0).mean())
        return {
            f"{label}_ann_ret":  round(ann_ret, 4),
            f"{label}_ann_vol":  round(ann_vol, 4),
            f"{label}_sharpe":   round(sharpe,  4),
            f"{label}_max_dd":   round(max_dd,  4),
            f"{label}_win_rate": round(win_rt,  4),
        }

    return {
        **_stats(returns["mkt_vw_lcl"], "vw"),
        **_stats(returns["mkt_ew_lcl"], "ew"),
        "n_months": int(returns["mkt_vw_lcl"].notna().sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path) -> None:
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("USA Market Benchmark — EW & VW Local Returns", fontsize=13, y=0.98)

    ax = axes[0]
    vw_cum = (1 + returns["mkt_vw_lcl"]).cumprod()
    ew_cum = (1 + returns["mkt_ew_lcl"]).cumprod()
    ax.plot(dates, vw_cum, label="Value-Weighted (VW)", lw=1.6, color="steelblue")
    ax.plot(dates, ew_cum, label="Equal-Weighted (EW)", lw=1.6, color="darkorange", ls="--")
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.fill_between(dates, vw_cum, 1, where=(vw_cum < 1), alpha=0.15, color="red")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Market Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    for col, color, ls, label in [
        ("mkt_vw_lcl", "steelblue",  "-",  "VW"),
        ("mkt_ew_lcl", "darkorange", "--", "EW"),
    ]:
        roll     = returns[col].rolling(12)
        roll_std = roll.std()
        roll_sharpe = np.where(roll_std > 0, roll.mean() / roll_std * np.sqrt(12), np.nan)
        ax.plot(dates, roll_sharpe, color=color, lw=1.4, ls=ls, label=label)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = out_dir / "cumulative_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(start_year: int = 1986) -> None:
    BACKTEST.mkdir(parents=True, exist_ok=True)

    print("Loading market_returns.csv …")
    returns = load_data(start_year)
    print(
        f"  {len(returns)} months  |  "
        f"{returns['eom'].min().date()} → {returns['eom'].max().date()}"
    )

    returns.to_csv(BACKTEST / "monthly_returns.csv", index=False)
    print(f"  → {BACKTEST}/monthly_returns.csv")

    perf = evaluate_performance(returns)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    print(f"  → {BACKTEST}/performance.csv")

    sep = "=" * 62
    print(sep)
    print("  USA MARKET BENCHMARK — PERFORMANCE SUMMARY")
    print(sep)
    print(f"  Period  : {returns['eom'].min().date()} → {returns['eom'].max().date()}")
    print(f"  Months  : {perf['n_months']}")
    print("  ── Value-Weighted (mkt_vw_lcl) ─────────────────────")
    print(f"  Ann. Return    : {perf['vw_ann_ret']:>8.2%}")
    print(f"  Ann. Volatility: {perf['vw_ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {perf['vw_sharpe']:>8.2f}")
    print(f"  Max Drawdown   : {perf['vw_max_dd']:>8.2%}")
    print(f"  Win Rate       : {perf['vw_win_rate']:>8.2%}")
    print("  ── Equal-Weighted (mkt_ew_lcl) ──────────────────────")
    print(f"  Ann. Return    : {perf['ew_ann_ret']:>8.2%}")
    print(f"  Ann. Volatility: {perf['ew_ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {perf['ew_sharpe']:>8.2f}")
    print(f"  Max Drawdown   : {perf['ew_max_dd']:>8.2%}")
    print(f"  Win Rate       : {perf['ew_win_rate']:>8.2%}")
    print(sep)

    plot_results(returns, BACKTEST)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USA market benchmark from JKP market_returns.csv."
    )
    parser.add_argument(
        "--start", type=int, default=1986,
        help="First year to include (default: 1986).",
    )
    args = parser.parse_args()
    run_benchmark(start_year=args.start)