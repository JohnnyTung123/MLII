"""
Single-factor factor backtest — JKP 2023 USA data.

For each characteristic, runs a standalone long-short decile backtest
using the same timing and universe as lasso.py, then computes:
  - Annualised Sharpe ratio
  - Annualised return and volatility
  - Cumulative return
  - t-statistic (mean monthly return vs zero)
  - Win rate

Results are saved to data/backtest/factor/ and a summary plot is produced.

Usage:
    python code/factor.py                        # full run
    python code/factor.py --start 1995 --tc 5   # custom start year / TC
    python code/factor.py --top 30               # show top 30 factors
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

warnings.filterwarnings("ignore")

# ── Paths (mirror lasso.py layout) ────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BENCH_DIR = _ROOT / "data" / "backtest" / "factor"

# ── Hyperparameters (keep consistent with lasso.py) ───────────────────────────
DECILE       = 0.10    # long/short fraction
MIN_STOCKS   = 100     # skip month if fewer stocks available
MAX_NAN_FRAC = 0.30    # drop characteristic if > this fraction NaN globally
RET_CAP      = 1.0     # winsorise |realized return| > 100% (data errors)

# Identifier / flag columns — never used as signals
_META = {
    "id", "date", "eom", "source_crsp", "size_grp",
    "obs_main", "exch_main", "primary_sec", "gvkey", "iid",
    "permno", "permco", "excntry", "curcd", "fx", "common",
    "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "adjfct", "shares", "me_lag1", "gics", "sic", "naics", "ff49",
}
_LOOKAHEAD = {"ret_exc_lead1m"}

Y_COL = "ret_exc_lead1m"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (identical to lasso.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[dict[pd.Timestamp, pd.DataFrame], list[str]]:
    """
    Load the cleaned parquet and split into per-month cross-sections.

    Returns
    -------
    slices    : {eom → DataFrame indexed by stock id}
    char_cols : list of characteristic predictor column names
    """
    parquet = PROCESSED / "data.parquet"
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found. Run the data-cleaning notebook first."
        )

    print("Loading data.parquet …")
    df = pd.read_parquet(parquet)

    char_cols = [
        c for c in df.columns
        if c not in _META
        and c not in _LOOKAHEAD
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    nan_frac  = df[char_cols].isna().mean()
    char_cols = [c for c in char_cols if nan_frac[c] <= MAX_NAN_FRAC]

    df = df[["id", "eom", Y_COL] + char_cols].copy()
    print(f"  {len(df):,} rows  |  {len(char_cols)} characteristics  "
          f"|  {df['eom'].nunique()} months")

    slices: dict[pd.Timestamp, pd.DataFrame] = {}
    for eom, grp in df.groupby("eom"):
        slices[eom] = grp.drop(columns="eom").set_index("id")

    return slices, char_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE-FACTOR MONTHLY RETURN
# ══════════════════════════════════════════════════════════════════════════════

def _factor_qspread(
    signal_slice: pd.DataFrame,
    return_slice: pd.DataFrame,
    col: str,
    tc: float,
    prev_long_ids:  set | None,
    prev_short_ids: set | None,
) -> tuple[float, float, set, set]:
    """
    Compute gross and net Q-spread for a single factor in one month.

    Timing (no look-ahead):
        signal  = characteristic at t-1   (signal_slice)
        return  = ret_exc_lead1m at t     (return_slice, realized at t+1)

    Returns
    -------
    gross_qs     : long minus short average realized return (gross)
    net_qs       : gross minus round-trip transaction cost on turnover
    long_ids     : stock ids in the long leg this month
    short_ids    : stock ids in the short leg this month
    """
    sig = signal_slice[col].dropna()
    ret = return_slice[Y_COL].reindex(sig.index).dropna()
    sig = sig.reindex(ret.index)

    # Remove extreme returns (data errors)
    mask = ret.abs() <= RET_CAP
    ret  = ret[mask]
    sig  = sig.reindex(ret.index)

    n        = len(sig)
    n_decile = max(1, int(n * DECILE))

    if n < 2 * n_decile or n < MIN_STOCKS:
        return np.nan, np.nan, set(), set()

    ranked    = sig.sort_values()
    short_ids = set(ranked.index[:n_decile])
    long_ids  = set(ranked.index[-n_decile:])

    long_ret  = ret.reindex(list(long_ids)).mean()
    short_ret = ret.reindex(list(short_ids)).mean()
    gross_qs  = long_ret - short_ret

    # Turnover vs previous period
    if prev_long_ids and prev_short_ids:
        long_to  = 1 - len(long_ids  & prev_long_ids)  / len(long_ids)
        short_to = 1 - len(short_ids & prev_short_ids) / len(short_ids)
        turnover = (long_to + short_to) / 2
    else:
        turnover = 1.0

    net_qs = gross_qs - 2 * tc * turnover

    return gross_qs, net_qs, long_ids, short_ids


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERFORMANCE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def _perf_stats(monthly: pd.Series, label: str = "") -> dict:
    """
    Compute annualised performance statistics for a monthly return series.

    Metrics
    -------
    ann_ret  : annualised mean return
    ann_vol  : annualised volatility
    sharpe   : ann_ret / ann_vol
    max_dd   : maximum drawdown
    win_rate : fraction of months with positive return
    t_stat   : two-sided t-test of mean != 0
    p_value  : corresponding p-value
    """
    s = monthly.dropna()
    if len(s) < 12:
        return {}

    ann_ret = s.mean() * 12
    ann_vol = s.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum    = (1 + s).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

    win_rate = float((s > 0).mean())
    cum_ret  = float(cum.iloc[-1] - 1)

    tstat, pval = ttest_1samp(s, 0)

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}ann_ret":  round(ann_ret,  4),
        f"{prefix}ann_vol":  round(ann_vol,  4),
        f"{prefix}sharpe":   round(sharpe,   3),
        f"{prefix}max_dd":   round(max_dd,   4),
        f"{prefix}win_rate": round(win_rate,  4),
        f"{prefix}cum_ret":  round(cum_ret,   4),
        f"{prefix}t_stat":   round(float(tstat), 3),
        f"{prefix}p_value":  round(float(pval),  4),
        f"{prefix}n_months": int(len(s)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN BENCHMARK LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_factor_benchmarks(
    start_year: int = 1981,
    tc_bps:     float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a long-short decile backtest for every characteristic independently.

    Parameters
    ----------
    start_year : first calendar year for portfolio returns
    tc_bps     : one-way transaction cost in basis points

    Returns
    -------
    summary_df      : one row per factor, sorted by gross Sharpe descending
    monthly_ret_df  : wide DataFrame of monthly gross Q-spreads (factors as columns)
    """
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    slices, char_cols = load_data()
    months   = sorted(slices.keys())
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    valid    = [(i, t) for i, t in enumerate(months) if i >= 1 and t >= start_ts]

    tc = tc_bps / 10_000

    print(f"\nBenchmark: {valid[0][1].date()} → {valid[-1][1].date()}"
          f"  |  {len(valid)} months  |  {len(char_cols)} factors"
          f"  |  TC = {tc_bps} bps one-way\n")

    # monthly_gross[factor][month_index] = gross Q-spread
    monthly_gross: dict[str, list] = {c: [] for c in char_cols}
    monthly_net:   dict[str, list] = {c: [] for c in char_cols}
    month_dates:   list            = []

    # Track per-factor portfolio holdings for turnover calculation
    prev_longs:  dict[str, set | None] = {c: None for c in char_cols}
    prev_shorts: dict[str, set | None] = {c: None for c in char_cols}

    for step, (i, t) in enumerate(valid):
        t_prev = months[i - 1]
        if t_prev not in slices or t not in slices:
            continue

        signal_slice = slices[t_prev]
        return_slice = slices[t]
        month_dates.append(t)

        for col in char_cols:
            gross, net, long_ids, short_ids = _factor_qspread(
                signal_slice, return_slice, col, tc,
                prev_longs[col], prev_shorts[col],
            )
            monthly_gross[col].append(gross)
            monthly_net[col].append(net)

            if long_ids:   # update holdings only if portfolio was formed
                prev_longs[col]  = long_ids
                prev_shorts[col] = short_ids

        # Progress every 12 months
        if (step + 1) % 12 == 0 or step == 0:
            print(f"  Processed through {t.date()}  "
                  f"({step + 1}/{len(valid)} months)")

    # ── Build monthly return DataFrames ───────────────────────────────────────
    gross_df = pd.DataFrame(monthly_gross, index=month_dates)
    net_df   = pd.DataFrame(monthly_net,   index=month_dates)
    gross_df.index.name = "eom"
    net_df.index.name   = "eom"

    # ── Compute summary statistics per factor ─────────────────────────────────
    print("\nComputing summary statistics …")
    rows = []
    for col in char_cols:
        g_stats = _perf_stats(gross_df[col], label="gross")
        n_stats = _perf_stats(net_df[col],   label="net")
        if not g_stats:
            continue
        rows.append({"factor": col, **g_stats, **n_stats})

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("gross_sharpe", ascending=False)
        .reset_index(drop=True)
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary_df.to_csv(BENCH_DIR / "factor_summary.csv",    index=False)
    gross_df.to_csv(  BENCH_DIR / "factor_monthly_gross.csv")
    net_df.to_csv(    BENCH_DIR / "factor_monthly_net.csv")

    print(f"\n  → {BENCH_DIR}/factor_summary.csv    ({len(summary_df)} factors)")
    print(f"  → {BENCH_DIR}/factor_monthly_gross.csv")
    print(f"  → {BENCH_DIR}/factor_monthly_net.csv")

    return summary_df, gross_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_factor_summary(
    summary_df: pd.DataFrame,
    gross_df:   pd.DataFrame,
    top_n:      int = 20,
) -> None:
    """
    Four-panel figure saved to BENCH_DIR/factor_benchmarks.png:
      (a) Top-N factors by gross Sharpe ratio  (colour = significance)
      (b) Top-N factors by t-statistic
      (c) Cumulative return of top 5 individual factors
      (d) Distribution of Sharpe ratios across all factors
    """
    top      = summary_df.head(top_n).copy()
    sig_mask = top["gross_t_stat"].abs() > 1.96
    colors   = ["steelblue" if s else "salmon" for s in sig_mask]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Single-Factor Benchmark  |  Top {top_n} Factors by Sharpe Ratio",
        fontsize=14, y=0.99,
    )

    # ── (a) Sharpe ratio bar chart ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.barh(top["factor"][::-1], top["gross_sharpe"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Annualised Sharpe Ratio (gross)")
    ax.set_title("Sharpe Ratio  [blue = |t| > 1.96,  red = insignificant]")
    ax.grid(True, alpha=0.2, axis="x")

    # ── (b) t-statistic bar chart ─────────────────────────────────────────────
    ax = axes[0, 1]
    ax.barh(top["factor"][::-1], top["gross_t_stat"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.axvline( 1.96, color="darkorange", lw=1.2, ls="--", label="5%  (1.96)")
    ax.axvline( 2.58, color="red",        lw=1.2, ls="--", label="1%  (2.58)")
    ax.axvline(0,     color="black",      lw=0.8)
    ax.set_xlabel("t-statistic")
    ax.set_title("Statistical Significance of Mean Monthly Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="x")

    # ── (c) Cumulative returns — top 5 factors ────────────────────────────────
    ax    = axes[1, 0]
    top5  = summary_df.head(5)["factor"].tolist()
    dates = pd.to_datetime(gross_df.index)
    cmap  = plt.cm.tab10

    for k, col in enumerate(top5):
        s   = gross_df[col].dropna()
        cum = (1 + s).cumprod()
        ax.plot(dates[:len(cum)], cum.values,
                label=col, lw=1.5, color=cmap(k))

    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Gross Return — Top 5 Factors")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.autofmt_xdate()

    # ── (d) Sharpe distribution across all factors ────────────────────────────
    ax = axes[1, 1]
    all_sharpes = summary_df["gross_sharpe"].dropna()
    ax.hist(all_sharpes, bins=40, color="steelblue", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.axvline(all_sharpes.median(), color="darkorange", lw=1.5,
               ls="--", label=f"Median = {all_sharpes.median():.2f}")
    ax.axvline(all_sharpes.max(),    color="red",        lw=1.5,
               ls="--", label=f"Max = {all_sharpes.max():.2f}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Gross Sharpe Ratio")
    ax.set_ylabel("Number of Factors")
    ax.set_title("Distribution of Sharpe Ratios Across All Factors")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = BENCH_DIR / "factor_benchmarks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(summary_df: pd.DataFrame, top_n: int = 20) -> None:
    """Print a formatted leaderboard of the top factors."""
    top = summary_df.head(top_n)

    print()
    print("=" * 90)
    print("  SINGLE-FACTOR BENCHMARK — TOP FACTORS BY GROSS SHARPE")
    print("=" * 90)
    print(f"  {'Rank':<5} {'Factor':<25} {'Sharpe':>7} {'Ann Ret':>8} "
          f"{'Ann Vol':>8} {'t-stat':>7} {'p-val':>7} {'Win%':>6} {'Months':>7}")
    print("-" * 90)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        sig = "**" if abs(row["gross_t_stat"]) > 2.58 else \
              "*"  if abs(row["gross_t_stat"]) > 1.96 else "  "
        print(
            f"  {rank:<5} {row['factor']:<25} "
            f"{row['gross_sharpe']:>7.3f} "
            f"{row['gross_ann_ret']:>7.2%} "
            f"{row['gross_ann_vol']:>7.2%} "
            f"{row['gross_t_stat']:>7.2f}{sig} "
            f"{row['gross_p_value']:>7.4f} "
            f"{row['gross_win_rate']:>5.1%} "
            f"{row['gross_n_months']:>7}"
        )

    print("=" * 90)
    print("  * p < 0.05   ** p < 0.01")
    print(f"\n  Total factors evaluated : {len(summary_df)}")
    print(f"  Significant (p < 0.05)  : {(summary_df['gross_p_value'] < 0.05).sum()}")
    print(f"  Significant (p < 0.01)  : {(summary_df['gross_p_value'] < 0.01).sum()}")
    print(f"  Best Sharpe             : {summary_df['gross_sharpe'].max():.3f}"
          f"  ({summary_df.iloc[0]['factor']})")
    print(f"  Median Sharpe           : {summary_df['gross_sharpe'].median():.3f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-factor factor backtest on JKP USA data."
    )
    parser.add_argument(
        "--start", type=int, default=1981,
        help="First year for portfolio returns (default: 1981).",
    )
    parser.add_argument(
        "--tc", type=float, default=10.0,
        help="One-way transaction cost in basis points (default: 10).",
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top factors to display and plot (default: 20).",
    )
    args = parser.parse_args()

    summary_df, gross_df = run_factor_benchmarks(
        start_year=args.start,
        tc_bps=args.tc,
    )

    print_summary(summary_df, top_n=args.top)
    plot_factor_summary(summary_df, gross_df, top_n=args.top)

    print("\nDone. Use these results to set your LASSO factor target:")
    print(f"  → Your LASSO gross Sharpe must exceed "
          f"{summary_df['gross_sharpe'].max():.3f} "
          f"({summary_df.iloc[0]['factor']}) to justify ML complexity.\n")