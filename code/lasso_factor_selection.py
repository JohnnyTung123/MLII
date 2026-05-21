"""
Cross-sectional LASSO equity backtest — JKP 2023 USA data.

Timing (no look-ahead bias):
  Training at month t:
      y  = ret_exc_lead1m[t-1]   →  realized excess return at month t
      X  = characteristics[t-1]  →  all signals known at end of month t-1
  Signal at month t:
      X  = characteristics[t]    →  signals known at end of month t
      ŷ  = predicted return at month t+1
  Portfolio held during t+1:
      realized = ret_exc_lead1m[t]

Feature filtering (no look-ahead bias):
  Before each month's training, each candidate factor is evaluated over the
  preceding FILTER_LOOKBACK months using a simple long-short decile sort.
  Only factors with Sharpe >= FILTER_MIN_SHARPE AND |t-stat| >= FILTER_MIN_TSTAT
  are passed to LASSO. All evaluation uses only data strictly before month t.

Extreme return removal:
  Stocks whose realized |ret_exc_lead1m| exceeds RET_CAP are dropped from both
  training data and portfolio evaluation each month.

Checkpointing:
  Results are flushed to CSV after every CHECKPOINT_EVERY months so a crash
  never loses more than one checkpoint window of work.
  A checkpoint.pkl records the last completed month and portfolio state so the
  run can be resumed with --resume.

Usage:
    python code/lasso.py                       # fresh run
    python code/lasso.py --resume              # continue from last checkpoint
    python code/lasso.py --start 1995 --tc 5
    python code/lasso.py --no-filter           # skip feature filtering
"""

import argparse
import pickle
import warnings
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "lasso_factor_selection"

# ── Hyperparameters ────────────────────────────────────────────────────────────
DECILE           = 0.10    # long/short fraction
MIN_STOCKS       = 100     # skip month if fewer stocks survive cleaning
MAX_NAN_FRAC     = 0.30    # drop characteristic column if > this fraction NaN globally
CV_FOLDS         = 5
N_ALPHAS         = 50      # coarser alpha grid for speed
MAX_ITER         = 10_000
CHECKPOINT_EVERY = 12      # flush to disk and save checkpoint every N months

# ── Training window ───────────────────────────────────────────────────────────
TRAIN_LOOKBACK   = 60      # rolling training window in months (5 years)

# ── Extreme return filter ──────────────────────────────────────────────────────
RET_CAP          = 1.0     # drop stocks with |realized return| > 100% in a month

# ── Feature filtering ──────────────────────────────────────────────────────────
FILTER_LOOKBACK   = 60     # months of history used to evaluate each factor
FILTER_MIN_SHARPE = 0.8    # minimum annualised Sharpe for a factor to survive
FILTER_MIN_TSTAT  = 2.0    # minimum |t-stat| for a factor to survive
FILTER_MIN_ACTIVE = 5      # fallback: if fewer factors survive, use all char_cols

# Identifier / flag columns — never used as predictors
_META = {
    "id", "date", "eom", "source_crsp", "size_grp",
    "obs_main", "exch_main", "primary_sec", "gvkey", "iid",
    "permno", "permco", "excntry", "curcd", "fx", "common",
    "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "adjfct", "shares", "me_lag1", "gics", "sic", "naics", "ff49",
}
_LOOKAHEAD = {"ret_exc_lead1m"}   # forward-looking — never use as predictor

Y_COL = "ret_exc_lead1m"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
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
    print(f"  {len(df):,} rows  |  {len(char_cols)} predictors  "
          f"|  {df['eom'].nunique()} months")

    slices: dict[pd.Timestamp, pd.DataFrame] = {}
    for eom, grp in df.groupby("eom"):
        slices[eom] = grp.drop(columns="eom").set_index("id")

    return slices, char_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_factors(
    slices: dict[pd.Timestamp, pd.DataFrame],
    months: list[pd.Timestamp],
    t_idx: int,
    char_cols: list[str],
) -> list[str]:
    """
    Select factors with sufficient single-factor predictive power, evaluated
    on a rolling window of historical months strictly before month t.

    For each candidate factor, a simple equal-weighted long-short decile sort
    is run over the preceding FILTER_LOOKBACK months. The factor survives if:
        annualised Sharpe >= FILTER_MIN_SHARPE
        AND |t-stat|      >= FILTER_MIN_TSTAT

    All data used is strictly historical — no look-ahead bias.

    Parameters
    ----------
    slices    : per-month cross-section dict (from load_data)
    months    : sorted list of all available timestamps
    t_idx     : index of the current month in `months`
    char_cols : full list of candidate predictor columns

    Returns
    -------
    List of factor names that pass both thresholds. Falls back to char_cols
    if fewer than FILTER_MIN_ACTIVE factors survive.
    """
    # Only look at months strictly before t
    history_idx = range(max(1, t_idx - FILTER_LOOKBACK), t_idx)

    # Accumulate monthly Q-spread for each factor over the lookback window
    factor_monthly: dict[str, list[float]] = {c: [] for c in char_cols}

    for i in history_idx:
        t_h      = months[i]
        t_h_prev = months[i - 1]

        if t_h_prev not in slices or t_h not in slices:
            continue

        signal_slice = slices[t_h_prev]   # signal known at end of t_h-1
        ret          = slices[t_h][Y_COL] # realized return at t_h

        # Remove extreme returns within the historical window too
        ret = ret[ret.abs() <= RET_CAP]

        for col in char_cols:
            sig = signal_slice[col].reindex(ret.index).dropna()
            r   = ret.reindex(sig.index).dropna()
            sig = sig.reindex(r.index)

            n        = len(sig)
            n_decile = max(1, int(n * DECILE))
            if n < 2 * n_decile:
                continue

            ranked    = sig.sort_values()
            short_ids = ranked.index[:n_decile]
            long_ids  = ranked.index[-n_decile:]

            qspread = r.reindex(long_ids).mean() - r.reindex(short_ids).mean()
            factor_monthly[col].append(float(qspread))

    # Score each factor and apply thresholds
    selected = []
    for col, monthly in factor_monthly.items():
        s = pd.Series(monthly).dropna()
        if len(s) < 12:          # need at least 12 months of history
            continue

        ann_ret = s.mean() * 12
        ann_vol = s.std()  * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        tstat   = s.mean() / (s.std() / np.sqrt(len(s)))

        if sharpe >= FILTER_MIN_SHARPE and abs(tstat) >= FILTER_MIN_TSTAT:
            selected.append(col)

    # Fallback: if too few factors survive, use all char_cols
    if len(selected) < FILTER_MIN_ACTIVE:
        return char_cols

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    slices: dict[pd.Timestamp, pd.DataFrame],
    t_prev: pd.Timestamp,
    t: pd.Timestamp,
    char_cols: list[str],
    months: list[pd.Timestamp] | None = None,
    t_idx: int | None = None,
    use_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, SimpleImputer, list[str]] | tuple[None, ...]:
    """
    Build training arrays (from month t-1) and signal DataFrame (from month t).

    Changes vs original:
      - Extreme realized returns (|ret| > RET_CAP) are dropped from training.
      - If use_filter=True, only factors passing the rolling Sharpe/t-stat
        filter are used as predictors (evaluated on data strictly before t).
      - Returns active_cols as a fifth element so downstream functions know
        which columns were actually used.

    Training:
        y  = Y_COL at t-1     — excess return realized at month t
        X  = active_cols[t-1] — characteristics known at end of month t-1
        Missing feature values are imputed with the cross-sectional mean.

    Signal:
        DataFrame at month t; includes active_cols and Y_COL (for evaluation).

    Returns (None, None, None, None, None) if too few training stocks.
    """
    if t not in slices:
        return None, None, None, None, None

    # ── Build rolling training window (up to TRAIN_LOOKBACK months before t) ──
    if months is not None and t_idx is not None:
        window_start = max(0, t_idx - TRAIN_LOOKBACK)
        train_months = [months[j] for j in range(window_start, t_idx)]
    else:
        train_months = [t_prev]

    train_frames = []
    for m in train_months:
        if m not in slices:
            continue
        frame = slices[m].dropna(subset=[Y_COL])
        frame = frame[frame[Y_COL].abs() <= RET_CAP]
        if len(frame) > 0:
            train_frames.append(frame)

    if not train_frames:
        return None, None, None, None, None

    train = pd.concat(train_frames)

    if len(train) < MIN_STOCKS:
        return None, None, None, None, None

    # ── Feature filtering ─────────────────────────────────────────────────────
    if use_filter and months is not None and t_idx is not None:
        active_cols = filter_factors(slices, months, t_idx, char_cols)
    else:
        active_cols = char_cols

    y_train = train[Y_COL].values
    X_raw   = train[active_cols].values

    # Drop columns that are entirely NaN in this training cross-section.
    # sklearn >= 1.2 SimpleImputer removes such columns by default, which
    # would make model.coef_ shorter than active_cols and cause a length
    # mismatch when building the coef Series. Early-period data (e.g. 1981)
    # can have many characteristics not yet available.
    has_data    = ~np.isnan(X_raw).all(axis=0)
    active_cols = [c for c, v in zip(active_cols, has_data) if v]
    X_train     = X_raw[:, has_data]

    imputer     = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)

    signal_df = slices[t][active_cols + [Y_COL]]
    return X_train_imp, y_train, signal_df, imputer, active_cols


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_lasso(
    X_train_imp: np.ndarray,
    y_train: np.ndarray,
    active_cols: list[str],
) -> tuple[LassoCV, StandardScaler, pd.Series]:
    """
    Standardize features and fit LassoCV (alpha selected by k-fold CV).

    Returns model, scaler (reuse for signal transform), and coefficient Series.
    """
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_train_imp)

    model = LassoCV(
        cv=CV_FOLDS,
        n_alphas=N_ALPHAS,
        max_iter=MAX_ITER,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_std, y_train)

    coefs = pd.Series(model.coef_, index=active_cols)
    return model, scaler, coefs


# ══════════════════════════════════════════════════════════════════════════════
# 5. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    model: LassoCV,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    signal_df: pd.DataFrame,
    active_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """
    Apply fitted model to characteristics at month t → predicted return at t+1.

    Returns (predicted, realized) Series indexed by stock id,
    or (None, None) if signal DataFrame is empty.
    """
    if signal_df.empty:
        return None, None

    X_raw = signal_df[active_cols].values
    X_std = scaler.transform(imputer.transform(X_raw))
    pred  = model.predict(X_std)

    predicted = pd.Series(pred,       index=signal_df.index, name="predicted_ret")
    realized  = signal_df[Y_COL].rename("realized_ret")
    return predicted, realized


# ══════════════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def construct_portfolio(
    predicted: pd.Series,
    realized: pd.Series,
    prev_long_ids: set | None,
    prev_short_ids: set | None,
    tc_bps: float,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Form equal-weighted long-short decile portfolio and compute returns.

    Uses only stocks with valid predicted AND valid realized returns.
    Extreme realized returns (|ret| > RET_CAP) are dropped before portfolio
    construction to prevent a single bad data point from distorting results.

    Returns (None, None) when there are not enough eligible stocks to fill
    both legs (e.g. the last month of the dataset has no forward return).

    Parameters
    ----------
    predicted      : predicted return signal for each stock
    realized       : actual ret_exc_lead1m (= portfolio return at t+1)
    prev_long_ids  : long-leg stock ids from the previous period (for turnover)
    prev_short_ids : short-leg stock ids from the previous period
    tc_bps         : one-way transaction cost in basis points
    """
    tc = tc_bps / 10_000

    # ── Remove extreme realized returns ───────────────────────────────────────
    realized = realized[realized.abs() <= RET_CAP]

    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index)

    n        = len(pred_c)
    n_decile = max(1, int(n * DECILE))

    # Need enough stocks to populate both legs without overlap
    if n < 2 * n_decile:
        return None, None

    ranked    = pred_c.sort_values()
    short_ids = set(ranked.index[:n_decile])
    long_ids  = set(ranked.index[-n_decile:])

    long_ret  = real_c.reindex(list(long_ids)).mean()
    short_ret = real_c.reindex(list(short_ids)).mean()
    qspread   = long_ret - short_ret

    # Turnover: fraction of each leg replaced vs previous period
    if prev_long_ids and prev_short_ids:
        long_to  = 1 - len(long_ids  & prev_long_ids)  / len(long_ids)
        short_to = 1 - len(short_ids & prev_short_ids) / len(short_ids)
        turnover = (long_to + short_to) / 2
    else:
        turnover = 1.0  # first period

    net_qspread = qspread - 2 * tc * turnover   # round-trip cost on turned-over fraction

    all_ids = list(long_ids) + list(short_ids)
    port_df = pd.DataFrame({
        "id":            all_ids,
        "leg":           ["long"]  * len(long_ids) + ["short"] * len(short_ids),
        "weight":        [1 / len(long_ids)]  * len(long_ids)
                       + [-1 / len(short_ids)] * len(short_ids),
        "predicted_ret": pred_c.reindex(all_ids).values,
        "realized_ret":  real_c.reindex(all_ids).values,
    })

    summary = {
        "long_ret":    long_ret,
        "short_ret":   short_ret,
        "qspread":     qspread,
        "net_qspread": net_qspread,
        "turnover":    turnover,
        "n_long":      len(long_ids),
        "n_short":     len(short_ids),
        "n_universe":  n,
    }
    return port_df, summary


# ══════════════════════════════════════════════════════════════════════════════
# 7. PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ic(predicted: pd.Series, realized: pd.Series) -> tuple[float, float]:
    """Cross-sectional Pearson and Spearman IC (predicted vs realized returns)."""
    aligned = pd.concat([predicted, realized], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    p = aligned.iloc[:, 0].values
    r = aligned.iloc[:, 1].values
    return float(np.corrcoef(p, r)[0, 1]), float(spearmanr(p, r)[0])


def evaluate_performance(returns: pd.DataFrame, tc_bps: float) -> dict:
    """
    Compute aggregate backtest statistics (annualised return, vol, Sharpe,
    max drawdown, win rate, mean IC, ICIR) for gross and net Q-spread.
    """
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

    ic_p = returns["ic_pearson"].dropna()
    ic_s = returns["ic_spearman"].dropna()
    icir_p = ic_p.mean() / ic_p.std() * np.sqrt(12) if ic_p.std() > 0 else np.nan
    icir_s = ic_s.mean() / ic_s.std() * np.sqrt(12) if ic_s.std() > 0 else np.nan

    return {
        **_stats(returns["qspread"],     "gross"),
        **_stats(returns["net_qspread"], "net"),
        "ic_mean_pearson":  round(float(ic_p.mean()), 4),
        "ic_mean_spearman": round(float(ic_s.mean()), 4),
        "icir_pearson":     round(float(icir_p), 4),
        "icir_spearman":    round(float(icir_s), 4),
        "n_months":         int(returns["qspread"].notna().sum()),
        "tc_bps":           tc_bps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path) -> None:
    """
    Three-panel figure saved to out_dir/cumulative_returns.png:
      (a) Cumulative gross and net Q-spread return
      (b) Rolling 12-month annualised Sharpe ratio
      (c) Monthly Spearman IC with 12-month rolling mean
    """
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("LASSO Long-Short Backtest (Top/Bottom Decile)", fontsize=13, y=0.98)

    # (a) Cumulative return
    ax = axes[0]
    gross_cum = (1 + returns["qspread"]).cumprod()
    net_cum   = (1 + returns["net_qspread"]).cumprod()
    ax.plot(dates, gross_cum, label="Gross Q-spread", lw=1.6, color="steelblue")
    ax.plot(dates, net_cum,   label="Net Q-spread",   lw=1.6, color="darkorange", ls="--")
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.fill_between(dates, gross_cum, 1,
                    where=(gross_cum < 1), alpha=0.15, color="red")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Q-Spread Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # (b) Rolling 12-month Sharpe
    ax = axes[1]
    roll     = returns["qspread"].rolling(12)
    roll_std = roll.std()
    roll_sharpe = np.where(roll_std > 0, roll.mean() / roll_std * np.sqrt(12), np.nan)
    ax.plot(dates, roll_sharpe, color="seagreen", lw=1.4)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.grid(True, alpha=0.25)

    # (c) IC
    ax = axes[2]
    ic     = returns["ic_spearman"]
    colors = ["steelblue" if v >= 0 else "tomato" for v in ic.fillna(0)]
    ax.bar(dates, ic, width=25, color=colors, alpha=0.55, label="Monthly IC")
    ax.plot(dates, ic.rolling(12).mean(), color="navy", lw=1.5, label="12m Rolling Mean IC")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("IC (Spearman)")
    ax.set_title("Information Coefficient — Predicted vs Realised Returns")
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


def plot_filter_stats(returns: pd.DataFrame, out_dir: Path) -> None:
    """
    Two-panel figure showing how feature filtering evolves over time:
      (a) Number of active factors selected each month
      (b) Fraction of total factors selected (active / total)
    Saved to out_dir/filter_stats.png.
    Only produced when feature filtering was enabled (n_active column present).
    """
    if "n_active" not in returns.columns:
        return

    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle("Feature Filtering — Active Factors Over Time", fontsize=13, y=0.98)

    ax = axes[0]
    ax.plot(dates, returns["n_active"], color="steelblue", lw=1.4)
    ax.set_ylabel("# Active Factors")
    ax.set_title(f"Factors Selected per Month  "
                 f"(Sharpe ≥ {FILTER_MIN_SHARPE}, |t| ≥ {FILTER_MIN_TSTAT}, "
                 f"lookback = {FILTER_LOOKBACK}m)")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    frac = returns["n_active"] / returns["n_total_cols"]
    ax.plot(dates, frac, color="darkorange", lw=1.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Fraction Selected")
    ax.set_title("Active Factors as % of Total Candidate Factors")
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = out_dir / "filter_stats.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"

def _save_checkpoint(step: int, t: pd.Timestamp,
                     prev_long_ids: set, prev_short_ids: set) -> None:
    """Persist the loop state so the run can be resumed after a crash."""
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "step":           step,
            "eom":            t,
            "prev_long_ids":  prev_long_ids  or set(),
            "prev_short_ids": prev_short_ids or set(),
        }, f)


def _load_checkpoint() -> dict | None:
    """Return saved checkpoint dict, or None if none exists."""
    if _CKPT_PATH.exists():
        with open(_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    return None


def _flush(
    rows_buf: list[dict],
    port_buf: list[pd.DataFrame],
    coef_buf: list[pd.Series],
    first_flush: bool,
) -> None:
    """
    Append buffered results to the three CSV output files.

    On the very first flush of a fresh run, headers are written.
    All subsequent flushes (and all flushes when resuming) append without headers.
    """
    if not rows_buf:
        return

    mode   = "w" if first_flush else "a"
    header = first_flush

    pd.DataFrame(rows_buf).to_csv(
        BACKTEST / "monthly_returns.csv", mode=mode, header=header, index=False
    )
    pd.concat(port_buf, ignore_index=True).to_csv(
        BACKTEST / "portfolios.csv", mode=mode, header=header, index=False
    )
    if coef_buf:
        pd.concat(coef_buf, axis=1).T.to_csv(
            BACKTEST / "coefs.csv", mode=mode, header=header
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int = 1981,
    tc_bps: float = 10.0,
    resume: bool = False,
    use_filter: bool = True,
) -> pd.DataFrame:
    """
    Execute the full LASSO cross-sectional backtest.

    Results are flushed to CSV every CHECKPOINT_EVERY months and a
    checkpoint.pkl is written so the run can be resumed with resume=True.

    Parameters
    ----------
    start_year : first calendar year for portfolio returns
    tc_bps     : one-way transaction cost in basis points
    resume     : if True, skip months already completed in a previous run
    use_filter : if True, apply rolling Sharpe/t-stat factor filter each month
    """
    BACKTEST.mkdir(parents=True, exist_ok=True)

    slices, char_cols = load_data()
    months   = sorted(slices.keys())
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    valid    = [(i, t) for i, t in enumerate(months) if i >= 1 and t >= start_ts]

    filter_label = (
        f"Sharpe≥{FILTER_MIN_SHARPE}, |t|≥{FILTER_MIN_TSTAT}, "
        f"lookback={FILTER_LOOKBACK}m"
        if use_filter else "disabled"
    )
    print(f"  Feature filtering : {filter_label}")

    # ── Resume logic ──────────────────────────────────────────────────────────
    prev_long_ids:  set | None = None
    prev_short_ids: set | None = None
    resume_after:   pd.Timestamp | None = None
    first_flush = True

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            print("No checkpoint found — starting fresh.")
        else:
            resume_after   = ckpt["eom"]
            prev_long_ids  = ckpt["prev_long_ids"]
            prev_short_ids = ckpt["prev_short_ids"]
            first_flush    = False   # existing CSV files already have headers
            print(f"Resuming from checkpoint: last completed month = {resume_after.date()}")

    print(f"\nBacktest: {valid[0][1].date()} → {valid[-1][1].date()}"
          f"  |  {len(valid)} months  |  TC = {tc_bps} bps one-way\n")

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    coef_buf: list[pd.Series]    = []

    for step, (i, t) in enumerate(valid):
        # Skip months already saved when resuming
        if resume_after is not None and t <= resume_after:
            continue

        t_prev = months[i - 1]

        # ── Preprocess (includes extreme return filter + feature filtering) ────
        result = preprocess(
            slices, t_prev, t, char_cols,
            months=months, t_idx=i,
            use_filter=use_filter,
        )
        if result[0] is None:
            continue
        X_train, y_train, signal_df, imputer, active_cols = result

        # ── Train ─────────────────────────────────────────────────────────────
        model, scaler, coefs = train_lasso(X_train, y_train, active_cols)

        # ── Signal generation ─────────────────────────────────────────────────
        predicted, realized = generate_signals(model, scaler, imputer, signal_df, active_cols)
        if predicted is None:
            continue

        # ── IC ────────────────────────────────────────────────────────────────
        ic_p, ic_s = compute_ic(predicted, realized)

        # ── Portfolio ─────────────────────────────────────────────────────────
        port_df, summary = construct_portfolio(
            predicted, realized, prev_long_ids, prev_short_ids, tc_bps
        )
        if port_df is None:
            print(f"  {t.date()}  skipped — insufficient stocks with forward returns")
            continue

        prev_long_ids  = set(port_df.loc[port_df["leg"] == "long",  "id"])
        prev_short_ids = set(port_df.loc[port_df["leg"] == "short", "id"])

        # ── Buffer results ────────────────────────────────────────────────────
        row = {
            "eom":          t,
            "ic_pearson":   ic_p,
            "ic_spearman":  ic_s,
            "alpha":        model.alpha_,
            "n_nonzero":    int((coefs != 0).sum()),
            "n_active":     len(active_cols),       # factors surviving filter
            "n_total_cols": len(char_cols),         # total candidate factors
            "n_train":      len(y_train),
            **summary,
        }
        rows_buf.append(row)

        port_df["eom"] = t
        port_buf.append(port_df)

        # coefs Series may have different indices each month (only active_cols);
        # reindex to full char_cols so coefs.csv always has the same columns,
        # with NaN for factors that were filtered out this month.
        coef_s      = coefs.reindex(char_cols)
        coef_s.name = t
        coef_buf.append(coef_s)

        # ── Progress ──────────────────────────────────────────────────────────
        if len(rows_buf) % 12 == 1 or len(rows_buf) == 1:
            print(
                f"  {t.date()}  |  "
                f"qspread={summary['qspread']:+.4f}  "
                f"IC={ic_s:+.3f}  "
                f"alpha={model.alpha_:.2e}  "
                f"nonzero={int((coefs != 0).sum()):>3d}  "
                f"active={len(active_cols):>3d}/{len(char_cols)}  "
                f"n_train={len(y_train):,}"
            )

        # ── Checkpoint flush ──────────────────────────────────────────────────
        if len(rows_buf) % CHECKPOINT_EVERY == 0:
            _flush(rows_buf, port_buf, coef_buf, first_flush)
            _save_checkpoint(step, t, prev_long_ids, prev_short_ids)
            print(f"  ✓ checkpoint saved  ({t.date()})")
            rows_buf, port_buf, coef_buf = [], [], []
            first_flush = False

    # ── Final flush (remaining buffer) ────────────────────────────────────────
    if rows_buf:
        _flush(rows_buf, port_buf, coef_buf, first_flush)

    # ── Aggregate summary files ───────────────────────────────────────────────
    print("\nFinalising outputs …")
    returns_df = pd.read_csv(BACKTEST / "monthly_returns.csv", parse_dates=["eom"])
    print(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")
    print(f"  → {BACKTEST}/portfolios.csv")
    print(f"  → {BACKTEST}/coefs.csv")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    print(f"  → {BACKTEST}/performance.csv")

    # ── Print summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  LASSO BACKTEST — PERFORMANCE SUMMARY")
    print("=" * 62)
    print(f"  Period         : {returns_df['eom'].min().date()} "
          f"→ {returns_df['eom'].max().date()}")
    print(f"  Months         : {perf['n_months']}")
    print(f"  TC             : {tc_bps} bps one-way")
    print(f"  Feature filter : {filter_label}")
    if use_filter and "n_active" in returns_df.columns:
        print(f"  Avg active fac : {returns_df['n_active'].mean():.1f} "
              f"/ {returns_df['n_total_cols'].iloc[0]}")
    print()
    print("  ── Gross Q-spread ──────────────────────────────")
    print(f"  Ann. Return    : {perf['gross_ann_ret']:>8.2%}")
    print(f"  Ann. Volatility: {perf['gross_ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {perf['gross_sharpe']:>8.2f}")
    print(f"  Max Drawdown   : {perf['gross_max_dd']:>8.2%}")
    print(f"  Win Rate       : {perf['gross_win_rate']:>8.2%}")
    print(f"\n  ── Net Q-spread (after {tc_bps} bps TC) ─────────────")
    print(f"  Ann. Return    : {perf['net_ann_ret']:>8.2%}")
    print(f"  Ann. Volatility: {perf['net_ann_vol']:>8.2%}")
    print(f"  Sharpe Ratio   : {perf['net_sharpe']:>8.2f}")
    print(f"  Max Drawdown   : {perf['net_max_dd']:>8.2%}")
    print(f"  Win Rate       : {perf['net_win_rate']:>8.2%}")
    print(f"\n  ── Information Coefficient ─────────────────────")
    print(f"  Mean IC (Pearson)  : {perf['ic_mean_pearson']:>8.4f}")
    print(f"  Mean IC (Spearman) : {perf['ic_mean_spearman']:>8.4f}")
    print(f"  ICIR (Spearman)    : {perf['icir_spearman']:>8.2f}")
    print("=" * 62)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_results(returns_df, BACKTEST)
    plot_filter_stats(returns_df, BACKTEST)   # no-op if filtering was disabled

    # Clean up checkpoint once run completes successfully
    if _CKPT_PATH.exists():
        _CKPT_PATH.unlink()
        print("  checkpoint.pkl removed (run complete)")

    return returns_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-sectional LASSO equity backtest on JKP USA data."
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
        "--resume", action="store_true",
        help="Resume from the last saved checkpoint instead of starting fresh.",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable rolling Sharpe/t-stat feature filtering (use all factors).",
    )
    args = parser.parse_args()
    run_backtest(
        start_year=args.start,
        tc_bps=args.tc,
        resume=args.resume,
        use_filter=not args.no_filter,
    )