"""
Cross-sectional XGBoost equity backtest — JKP 2023 USA data.

Timing is identical to lasso.py (no look-ahead bias):
  Training at month t:
      y  = ret_exc_lead1m[t-1]   — realized excess return at month t
      X  = characteristics[t-1]  — all signals known at end of month t-1
  Signal at month t:
      X  = characteristics[t]    — signals known at end of month t
      ŷ  = predicted return at month t+1
  Portfolio held during t+1:
      realized = ret_exc_lead1m[t]

Key differences vs lasso.py:
  • XGBoost (gradient-boosted trees) needs no feature scaling or imputation.
    Missing feature values are handled natively: XGBoost learns the optimal
    split direction for NaN entries during training.
  • Alpha selection (LassoCV) is replaced by early stopping on a held-out
    validation split (VAL_FRAC of the training cross-section).
  • Coefficients (coefs.csv) are replaced by feature_importance.csv.
  • All console output is also written to data/backtest/xgboost/run.log.

Usage:
    python code/xg_boost.py                       # fresh run
    python code/xg_boost.py --resume              # continue from checkpoint
    python code/xg_boost.py --start 1995 --tc 5
"""

import argparse
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "xgboost"

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators":       500,    # upper bound; early stopping will reduce this
    "learning_rate":      0.05,
    "max_depth":          4,      # shallow trees reduce overfitting on noisy finance data
    "subsample":          0.8,    # row subsampling per tree
    "colsample_bytree":   0.5,    # feature subsampling per tree (important with 300+ features)
    "min_child_weight":   20,     # minimum leaf sample weight; prevents micro-splits
    "reg_alpha":          0.1,    # L1 regularization
    "reg_lambda":         1.0,    # L2 regularization
    "tree_method":        "hist", # fast histogram-based algorithm
    "n_jobs":             -1,
    "random_state":       42,
    "verbosity":          0,      # suppress XGBoost's own output
}
EARLY_STOPPING_ROUNDS = 30   # stop if validation RMSE doesn't improve for N rounds
VAL_FRAC              = 0.20 # fraction of training cross-section held out for early stopping

# ── Backtest hyperparameters ───────────────────────────────────────────────────
DECILE           = 0.10
MIN_STOCKS       = 100
MAX_NAN_FRAC     = 0.30
CHECKPOINT_EVERY = 12
RET_CAP          = 1.0    # ← add this

# Identifier / flag columns — never used as predictors
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
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    """
    Create a logger that writes to both the console and run.log.

    Each run appends to the same log file so historical runs are preserved.
    A separator line marks each new run start.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("xgboost_backtest")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()   # avoid duplicate handlers on re-imports

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — append mode so we keep a full audit trail across runs
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Mark start of a new run in the log file
    logger.info("=" * 70)
    logger.info(f"NEW RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    return logger


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(
    logger: logging.Logger,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[str]]:
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

    logger.info("Loading data.parquet …")
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
    logger.info(
        f"  {len(df):,} rows  |  {len(char_cols)} predictors  "
        f"|  {df['eom'].nunique()} months"
    )

    slices: dict[pd.Timestamp, pd.DataFrame] = {}
    for eom, grp in df.groupby("eom"):
        slices[eom] = grp.drop(columns="eom").set_index("id")

    return slices, char_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    slices: dict[pd.Timestamp, pd.DataFrame],
    t_prev: pd.Timestamp,
    t: pd.Timestamp,
    char_cols: list[str],
    ret_cap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | tuple[None, None, None]:
    """
    Build training arrays (from month t-1) and signal DataFrame (from month t).

    Unlike lasso.py, no imputation is applied: XGBoost handles NaN in X natively.
    Only rows where the target y (ret_exc_lead1m) is NaN are dropped.

    Returns (None, None, None) if too few training stocks.
    """
    if t_prev not in slices or t not in slices:
        return None, None, None

    train = slices[t_prev].dropna(subset=[Y_COL])

    # ── Remove extreme return observations ───────────────────────────
    train = train[train[Y_COL].abs() <= ret_cap]
    # ─────────────────────────────────────────────────────────────────

    if len(train) < MIN_STOCKS:
        return None, None, None

    y_train = train[Y_COL].values.astype(np.float32)
    X_train = train[char_cols].values.astype(np.float32)

    signal_df = slices[t][char_cols + [Y_COL]]
    return X_train, y_train, signal_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    char_cols: list[str],
    logger: logging.Logger,
) -> tuple[XGBRegressor, pd.Series]:
    """
    Fit XGBRegressor with early stopping on a held-out validation split.

    The validation set (VAL_FRAC of the training cross-section) is used only
    for early stopping — it does not introduce look-ahead bias because all data
    is from the same historical cross-section at month t-1.

    Returns
    -------
    model     : fitted XGBRegressor
    feat_imp  : pd.Series of feature importances (gain) indexed by char name
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_FRAC, random_state=42
    )

    model = XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    feat_imp = pd.Series(
        model.feature_importances_, index=char_cols, name="importance"
    )
    logger.debug(
        f"    XGBoost: best_iteration={model.best_iteration}  "
        f"top_feature={feat_imp.idxmax()}"
    )
    return model, feat_imp


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    model: XGBRegressor,
    signal_df: pd.DataFrame,
    char_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """
    Apply fitted model to characteristics at month t → predicted return at t+1.

    NaN feature values are passed directly to XGBoost (no imputation needed).

    Returns (predicted, realized) Series indexed by stock id,
    or (None, None) if signal DataFrame is empty.
    """
    if signal_df.empty:
        return None, None

    X_raw     = signal_df[char_cols].values.astype(np.float32)
    pred      = model.predict(X_raw)
    predicted = pd.Series(pred,               index=signal_df.index, name="predicted_ret")
    realized  = signal_df[Y_COL].rename("realized_ret")
    return predicted, realized


# ══════════════════════════════════════════════════════════════════════════════
# 5. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def construct_portfolio(
    predicted: pd.Series,
    realized: pd.Series,
    prev_long_ids: set | None,
    prev_short_ids: set | None,
    tc_bps: float,
    ret_cap: float = 1.0,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Form equal-weighted long-short decile portfolio and compute returns.

    Returns (None, None) when there are not enough eligible stocks (e.g. the
    final months of the dataset where ret_exc_lead1m is all NaN).
    """
    tc = tc_bps / 10_000

    # ── Remove extreme realized returns ──────────────────────────────
    realized = realized[realized.abs() <= ret_cap]
    # ─────────────────────────────────────────────────────────────────

    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index)

    n        = len(pred_c)
    n_decile = max(1, int(n * DECILE))

    if n < 2 * n_decile:   # guard against empty legs → ZeroDivisionError
        return None, None

    ranked    = pred_c.sort_values()
    short_ids = set(ranked.index[:n_decile])
    long_ids  = set(ranked.index[-n_decile:])

    long_ret  = real_c.reindex(list(long_ids)).mean()
    short_ret = real_c.reindex(list(short_ids)).mean()
    qspread   = long_ret - short_ret

    if prev_long_ids and prev_short_ids:
        long_to  = 1 - len(long_ids  & prev_long_ids)  / len(long_ids)
        short_to = 1 - len(short_ids & prev_short_ids) / len(short_ids)
        turnover = (long_to + short_to) / 2
    else:
        turnover = 1.0

    net_qspread = qspread - 2 * tc * turnover

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
# 6. PERFORMANCE EVALUATION
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
    Aggregate backtest statistics: annualised return, vol, Sharpe, max drawdown,
    win rate (gross and net), plus mean IC and ICIR.
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
# 7. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    """
    Three-panel figure saved to out_dir/cumulative_returns.png:
      (a) Cumulative gross and net Q-spread return
      (b) Rolling 12-month annualised Sharpe ratio
      (c) Monthly Spearman IC with 12-month rolling mean
    """
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("XGBoost Long-Short Backtest (Top/Bottom Decile)", fontsize=13, y=0.98)

    ax = axes[0]
    gross_cum = (1 + returns["qspread"]).cumprod()
    net_cum   = (1 + returns["net_qspread"]).cumprod()
    ax.plot(dates, gross_cum, label="Gross Q-spread", lw=1.6, color="steelblue")
    ax.plot(dates, net_cum,   label="Net Q-spread",   lw=1.6, color="darkorange", ls="--")
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.fill_between(dates, gross_cum, 1, where=(gross_cum < 1), alpha=0.15, color="red")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Q-Spread Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    roll     = returns["qspread"].rolling(12)
    roll_std = roll.std()
    roll_sharpe = np.where(roll_std > 0, roll.mean() / roll_std * np.sqrt(12), np.nan)
    ax.plot(dates, roll_sharpe, color="seagreen", lw=1.4)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.grid(True, alpha=0.25)

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
    logger.info(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"


def _save_checkpoint(
    step: int,
    t: pd.Timestamp,
    prev_long_ids: set,
    prev_short_ids: set,
) -> None:
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "step":           step,
            "eom":            t,
            "prev_long_ids":  prev_long_ids  or set(),
            "prev_short_ids": prev_short_ids or set(),
        }, f)


def _load_checkpoint() -> dict | None:
    if _CKPT_PATH.exists():
        with open(_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    return None


def _flush(
    rows_buf: list[dict],
    port_buf: list[pd.DataFrame],
    imp_buf: list[pd.Series],
    first_flush: bool,
) -> None:
    """Append buffered results to the three CSV output files."""
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
    if imp_buf:
        pd.concat(imp_buf, axis=1).T.to_csv(
            BACKTEST / "feature_importance.csv", mode=mode, header=header
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int = 1981,
    tc_bps: float = 10.0,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Execute the full XGBoost cross-sectional backtest.

    Results are flushed to CSV every CHECKPOINT_EVERY months.
    All output is also written to data/backtest/xgboost/run.log.
    """
    BACKTEST.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(BACKTEST)
    logger.info(f"Config: start_year={start_year}  tc={tc_bps} bps  "
                f"max_depth={XGB_PARAMS['max_depth']}  "
                f"lr={XGB_PARAMS['learning_rate']}  "
                f"n_estimators={XGB_PARAMS['n_estimators']}  "
                f"early_stopping={EARLY_STOPPING_ROUNDS}  "
                f"val_frac={VAL_FRAC}")

    slices, char_cols = load_data(logger)
    months   = sorted(slices.keys())
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    valid    = [(i, t) for i, t in enumerate(months) if i >= 1 and t >= start_ts]

    # ── Resume logic ──────────────────────────────────────────────────────────
    prev_long_ids:  set | None = None
    prev_short_ids: set | None = None
    resume_after:   pd.Timestamp | None = None
    first_flush = True

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found — starting fresh.")
        else:
            resume_after   = ckpt["eom"]
            prev_long_ids  = ckpt["prev_long_ids"]
            prev_short_ids = ckpt["prev_short_ids"]
            first_flush    = False
            logger.info(f"Resuming from checkpoint: last completed = {resume_after.date()}")

    logger.info(
        f"Backtest: {valid[0][1].date()} → {valid[-1][1].date()}  "
        f"|  {len(valid)} months  |  TC = {tc_bps} bps one-way"
    )

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    imp_buf:  list[pd.Series]    = []

    t_start = datetime.now()

    for step, (i, t) in enumerate(valid):
        if resume_after is not None and t <= resume_after:
            continue

        t_prev = months[i - 1]

        # ── Preprocess ────────────────────────────────────────────────────────
        X_train, y_train, signal_df = preprocess(slices, t_prev, t, char_cols)
        if X_train is None:
            logger.warning(f"{t.date()}  skipped — fewer than {MIN_STOCKS} training stocks")
            continue

        # ── Train ─────────────────────────────────────────────────────────────
        try:
            model, feat_imp = train_model(X_train, y_train, char_cols, logger)
        except Exception as exc:
            logger.error(f"{t.date()}  model training failed: {exc}")
            continue

        # ── Signal generation ─────────────────────────────────────────────────
        predicted, realized = generate_signals(model, signal_df, char_cols)
        if predicted is None:
            logger.warning(f"{t.date()}  skipped — empty signal DataFrame")
            continue

        # ── IC ────────────────────────────────────────────────────────────────
        ic_p, ic_s = compute_ic(predicted, realized)

        # ── Portfolio ─────────────────────────────────────────────────────────
        port_df, summary = construct_portfolio(
            predicted, realized, prev_long_ids, prev_short_ids, tc_bps
        )
        if port_df is None:
            logger.warning(
                f"{t.date()}  skipped — insufficient stocks with forward returns"
            )
            continue

        prev_long_ids  = set(port_df.loc[port_df["leg"] == "long",  "id"])
        prev_short_ids = set(port_df.loc[port_df["leg"] == "short", "id"])

        # ── Buffer ────────────────────────────────────────────────────────────
        row = {
            "eom":            t,
            "ic_pearson":     ic_p,
            "ic_spearman":    ic_s,
            "best_iteration": model.best_iteration,
            "n_train":        len(y_train),
            **summary,
        }
        rows_buf.append(row)

        port_df["eom"] = t
        port_buf.append(port_df)

        imp_row      = feat_imp.copy()
        imp_row.name = t
        imp_buf.append(imp_row)

        # ── Log progress ──────────────────────────────────────────────────────
        logger.info(
            f"{t.date()}  |  "
            f"qspread={summary['qspread']:+.4f}  "
            f"IC={ic_s:+.3f}  "
            f"trees={model.best_iteration:>3d}  "
            f"n_train={len(y_train):,}  "
            f"n_universe={summary['n_universe']:,}"
        )

        # ── Checkpoint flush ──────────────────────────────────────────────────
        if len(rows_buf) % CHECKPOINT_EVERY == 0:
            _flush(rows_buf, port_buf, imp_buf, first_flush)
            _save_checkpoint(step, t, prev_long_ids, prev_short_ids)
            elapsed = (datetime.now() - t_start).seconds // 60
            logger.info(f"  ✓ checkpoint saved  ({t.date()})  elapsed={elapsed}m")
            rows_buf, port_buf, imp_buf = [], [], []
            first_flush = False

    # ── Final flush ────────────────────────────────────────────────────────────
    if rows_buf:
        _flush(rows_buf, port_buf, imp_buf, first_flush)

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    logger.info("Finalising outputs …")
    returns_df = pd.read_csv(
        BACKTEST / "monthly_returns.csv",
        parse_dates=["eom"],
    )
    logger.info(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")
    logger.info(f"  → {BACKTEST}/portfolios.csv")
    logger.info(f"  → {BACKTEST}/feature_importance.csv")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    logger.info(f"  → {BACKTEST}/performance.csv")

    # ── Summary ────────────────────────────────────────────────────────────────
    sep = "=" * 58
    logger.info(sep)
    logger.info("  XGBOOST BACKTEST — PERFORMANCE SUMMARY")
    logger.info(sep)
    logger.info(
        f"  Period         : {returns_df['eom'].min().date()} "
        f"→ {returns_df['eom'].max().date()}"
    )
    logger.info(f"  Months         : {perf['n_months']}")
    logger.info(f"  TC             : {tc_bps} bps one-way")
    logger.info("  ── Gross Q-spread ──────────────────────────────")
    logger.info(f"  Ann. Return    : {perf['gross_ann_ret']:>8.2%}")
    logger.info(f"  Ann. Volatility: {perf['gross_ann_vol']:>8.2%}")
    logger.info(f"  Sharpe Ratio   : {perf['gross_sharpe']:>8.2f}")
    logger.info(f"  Max Drawdown   : {perf['gross_max_dd']:>8.2%}")
    logger.info(f"  Win Rate       : {perf['gross_win_rate']:>8.2%}")
    logger.info(f"  ── Net Q-spread (after {tc_bps} bps TC) ─────────────")
    logger.info(f"  Ann. Return    : {perf['net_ann_ret']:>8.2%}")
    logger.info(f"  Ann. Volatility: {perf['net_ann_vol']:>8.2%}")
    logger.info(f"  Sharpe Ratio   : {perf['net_sharpe']:>8.2f}")
    logger.info(f"  Max Drawdown   : {perf['net_max_dd']:>8.2%}")
    logger.info(f"  Win Rate       : {perf['net_win_rate']:>8.2%}")
    logger.info("  ── Information Coefficient ─────────────────────")
    logger.info(f"  Mean IC (Pearson)  : {perf['ic_mean_pearson']:>8.4f}")
    logger.info(f"  Mean IC (Spearman) : {perf['ic_mean_spearman']:>8.4f}")
    logger.info(f"  ICIR (Spearman)    : {perf['icir_spearman']:>8.2f}")
    logger.info(sep)

    total_min = (datetime.now() - t_start).seconds // 60
    logger.info(f"Total elapsed: {total_min} min")

    plot_results(returns_df, BACKTEST, logger)

    if _CKPT_PATH.exists():
        _CKPT_PATH.unlink()
        logger.info("checkpoint.pkl removed (run complete)")

    return returns_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-sectional XGBoost equity backtest on JKP USA data."
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
    args = parser.parse_args()
    run_backtest(start_year=args.start, tc_bps=args.tc, resume=args.resume)