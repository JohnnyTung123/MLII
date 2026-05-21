"""
Cross-sectional XGBoost equity backtest with expanding training window — JKP 2023 USA data.

Timing (no look-ahead bias):
  Training at month t (retrained every January):
      y  = ret_exc_lead1m[s]   for ALL s in [start, t-1]  — expanding window
      X  = characteristics[s]  for ALL s in [start, t-1]
      Early-stopping val set = LAST VAL_FRAC of the time-ordered training
      rows (temporal split, not random shuffle — avoids tuning the model
      to recent patterns before signal generation).
  Signal at month t:
      X  = characteristics[t]  — signals known at end of month t
      ŷ  = predicted return at month t+1
  Portfolio held during t+1:
      realized = ret_exc_lead1m[t]

Difference vs xg_boost.py (rolling window):
  Rolling:   train on (X[t-1], y[t-1]) only — one cross-section per month.
  Expanding: train on (X[s], y[s]) for all s from the first available month
             up to t-1.  The training set grows monotonically, giving the
             model more observations as time progresses and making early-period
             predictions more stable.

Everything else (features, portfolio construction, checkpointing, output format)
is identical to xg_boost.py so results are directly comparable.

Usage:
    python code/xg_boost_expanding.py                       # fresh run
    python code/xg_boost_expanding.py --resume              # continue from checkpoint
    python code/xg_boost_expanding.py --start 1995 --tc 5
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
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "xgboost_expanding"

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators":       500,
    "learning_rate":      0.05,
    "max_depth":          4,
    "subsample":          0.8,
    "colsample_bytree":   0.5,
    "min_child_weight":   20,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "tree_method":        "hist",
    "n_jobs":             -1,
    "random_state":       42,
    "verbosity":          0,
}
EARLY_STOPPING_ROUNDS = 30
VAL_FRAC              = 0.20

# ── Backtest hyperparameters ───────────────────────────────────────────────────
DECILE           = 0.10
MIN_STOCKS       = 100
MAX_NAN_FRAC     = 0.30
CHECKPOINT_EVERY = 12
RET_CAP          = 1.0

# Identifier / flag columns — never used as predictors
_META = {
    "id", "date", "eom", "source_crsp", "size_grp",
    "obs_main", "exch_main", "primary_sec", "gvkey", "iid",
    "permno", "permco", "excntry", "curcd", "fx", "common",
    "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "adjfct", "shares", "me_lag1", "gics", "sic", "naics", "ff49",
}
# ret_1_0 is byte-for-byte identical to ret; drop the duplicate so the
# current-month return doesn't receive double feature weight.
_LOOKAHEAD = {"ret_exc_lead1m", "ret_1_0", "ret"}

Y_COL = "ret_exc_lead1m"

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("xgboost_expanding_backtest")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
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
    # NaN filtering is NOT done here — computing nan_frac over the full sample
    # (including future dates) would introduce look-ahead bias.  Feature
    # selection is deferred to the annual retraining step, where it uses only
    # the expanding training data available at that point in time.

    df = df[["id", "eom", Y_COL] + char_cols].copy()
    logger.info(
        f"  {len(df):,} rows  |  {len(char_cols)} candidate predictors  "
        f"|  {df['eom'].nunique()} months"
    )

    slices: dict[pd.Timestamp, pd.DataFrame] = {}
    for eom, grp in df.groupby("eom"):
        slices[eom] = grp.drop(columns="eom").set_index("id")

    return slices, char_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    char_cols: list[str],
    logger: logging.Logger,
) -> tuple[XGBRegressor, pd.Series]:
    """
    Fit XGBRegressor with early stopping on a temporal validation split.

    We hold out the LAST VAL_FRAC of the expanding window as validation so
    that early stopping is calibrated on the most recent historical period —
    not on a random mix of past and recent months (which would let the model
    tune itself to recent data patterns before prediction, causing inflated
    win rates).  XGBoost handles NaN natively so no imputation is needed.
    """
    n = len(y_train)
    split = max(MIN_STOCKS, int(n * (1 - VAL_FRAC)))
    split = min(split, n - 1)           # need at least 1 row in val
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

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
# 3. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    model: XGBRegressor,
    signal_df: pd.DataFrame,
    char_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    if signal_df.empty:
        return None, None

    X_raw     = signal_df[char_cols].values.astype(np.float32)
    pred      = model.predict(X_raw)
    predicted = pd.Series(pred,               index=signal_df.index, name="predicted_ret")
    realized  = signal_df[Y_COL].rename("realized_ret")
    return predicted, realized


# ══════════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def construct_portfolio(
    predicted: pd.Series,
    realized: pd.Series,
    prev_long_ids: set | None,
    prev_short_ids: set | None,
    tc_bps: float,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    tc = tc_bps / 10_000

    realized = realized[realized.abs() <= RET_CAP]

    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index)

    n        = len(pred_c)
    n_decile = max(1, int(n * DECILE))

    if n < 2 * n_decile:
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
# 5. PERFORMANCE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ic(predicted: pd.Series, realized: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([predicted, realized], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    p = aligned.iloc[:, 0].values
    r = aligned.iloc[:, 1].values
    return float(np.corrcoef(p, r)[0, 1]), float(spearmanr(p, r)[0])


def evaluate_performance(returns: pd.DataFrame, tc_bps: float) -> dict:
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
# 6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        "XGBoost (Expanding Window) Long-Short Backtest (Top/Bottom Decile)",
        fontsize=13, y=0.98,
    )

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
# 7. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"


def _save_checkpoint(
    step: int,
    t: pd.Timestamp,
    prev_long_ids: set,
    prev_short_ids: set,
    n_train_rows: int,
    nan_sum: np.ndarray,
    obs_count: np.ndarray,
    current_indices: np.ndarray | None,
    current_cols: list[str] | None,
) -> None:
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "step":            step,
            "eom":             t,
            "prev_long_ids":   prev_long_ids  or set(),
            "prev_short_ids":  prev_short_ids or set(),
            "n_train_rows":    n_train_rows,
            "nan_sum":         nan_sum,
            "obs_count":       obs_count,
            "current_indices": current_indices,
            "current_cols":    current_cols,
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
# 8. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int = 1981,
    tc_bps: float = 10.0,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Execute the full expanding-window XGBoost cross-sectional backtest.

    At each month t the model is retrained on every cross-section from the
    start of the sample through month t-1.  The training set grows each month,
    so early-period models benefit from a small but growing dataset and
    later-period models leverage the full history.

    Results are flushed to CSV every CHECKPOINT_EVERY months.
    All output is also written to data/backtest/xgboost_expanding/run.log.
    """
    BACKTEST.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(BACKTEST)
    logger.info(
        f"Config: start_year={start_year}  tc={tc_bps} bps  "
        f"max_depth={XGB_PARAMS['max_depth']}  "
        f"lr={XGB_PARAMS['learning_rate']}  "
        f"n_estimators={XGB_PARAMS['n_estimators']}  "
        f"early_stopping={EARLY_STOPPING_ROUNDS}  "
        f"val_frac={VAL_FRAC}  window=expanding"
    )

    slices, char_cols = load_data(logger)
    months   = sorted(slices.keys())
    start_ts = pd.Timestamp(f"{start_year}-01-01")
    valid    = [(i, t) for i, t in enumerate(months) if i >= 1 and t >= start_ts]

    # ── Resume logic ──────────────────────────────────────────────────────────
    prev_long_ids:   set | None          = None
    prev_short_ids:  set | None          = None
    resume_after:    pd.Timestamp | None = None
    first_flush    = True
    n_train_rows   = 0

    # NaN accumulators for bias-free feature selection: we track cumulative
    # NaN counts and total observations over the expanding training window so
    # that at each annual retraining we can select features based solely on
    # data available up to that point — no full-sample look-ahead.
    nan_sum         = np.zeros(len(char_cols), dtype=np.float64)
    obs_count       = np.zeros(len(char_cols), dtype=np.float64)
    current_indices: np.ndarray | None = None   # selected col indices into char_cols
    current_cols:    list[str]  | None = None   # names of currently active features

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found — starting fresh.")
        else:
            resume_after    = ckpt["eom"]
            prev_long_ids   = ckpt["prev_long_ids"]
            prev_short_ids  = ckpt["prev_short_ids"]
            n_train_rows    = ckpt.get("n_train_rows", 0)
            nan_sum         = ckpt.get("nan_sum",         nan_sum)
            obs_count       = ckpt.get("obs_count",       obs_count)
            current_indices = ckpt.get("current_indices", None)
            current_cols    = ckpt.get("current_cols",    None)
            first_flush     = False
            logger.info(
                f"Resuming from checkpoint: last completed = {resume_after.date()}  "
                f"n_train_rows={n_train_rows:,}  "
                f"n_active_features={len(current_cols) if current_cols else 0}"
            )

    logger.info(
        f"Backtest: {valid[0][1].date()} → {valid[-1][1].date()}  "
        f"|  {len(valid)} months  |  TC = {tc_bps} bps one-way"
    )

    # ── Training data accumulator ─────────────────────────────────────────────
    # Each entry is one month's cleaned (X, y) arrays.  We stack lazily at
    # training time so numpy works on a contiguous block without repeated copies.
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    # When resuming, fast-forward the accumulator to the checkpoint month
    # without fitting any models.
    if resume_after is not None:
        logger.info("Rebuilding expanding training data for resume …")
        for k, m in enumerate(months):
            if k == 0:
                continue
            m_prev = months[k - 1]
            if m_prev in slices:
                frame = slices[m_prev].dropna(subset=[Y_COL])
                frame = frame[frame[Y_COL].abs() <= RET_CAP]
                if len(frame) > 0:
                    X_raw = frame[char_cols].values.astype(np.float32)
                    nan_sum   += np.isnan(X_raw).sum(axis=0)
                    obs_count += X_raw.shape[0]
                    X_list.append(X_raw)
                    y_list.append(frame[Y_COL].values.astype(np.float32))
            if m == resume_after:
                break
        logger.info(
            f"  Resume data rebuilt: {sum(len(y) for y in y_list):,} rows"
        )

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    imp_buf:  list[pd.Series]    = []
    t_start = datetime.now()

    model:        XGBRegressor | None = None
    feat_imp:     pd.Series    | None = None
    n_train_last: int                 = 0

    for step, (i, t) in enumerate(valid):
        t_prev = months[i - 1]

        # Skip months already handled by the pre-fill resume loop.
        # The pre-fill already built X_list / y_list through resume_after,
        # so re-appending t_prev here would double-count those months.
        if resume_after is not None and t <= resume_after:
            continue

        # ── Append t_prev's cross-section to the expanding training set ───────
        if t_prev in slices:
            frame = slices[t_prev].dropna(subset=[Y_COL])
            frame = frame[frame[Y_COL].abs() <= RET_CAP]
            if len(frame) > 0:
                X_raw = frame[char_cols].values.astype(np.float32)
                nan_sum   += np.isnan(X_raw).sum(axis=0)
                obs_count += X_raw.shape[0]
                X_list.append(X_raw)
                y_list.append(frame[Y_COL].values.astype(np.float32))

        if not X_list:
            continue

        # ── Retrain once per calendar year (or when no model exists yet) ───────
        if model is None or t.month == 1:
            X_train_all = np.vstack(X_list)
            y_train     = np.concatenate(y_list)

            if len(y_train) >= MIN_STOCKS:
                try:
                    # Feature selection: use only columns whose NaN fraction
                    # in the expanding training data is within the threshold.
                    # This avoids look-ahead bias from the full-sample filter.
                    nan_frac_tr     = np.where(obs_count > 0, nan_sum / obs_count, 1.0)
                    sel_mask        = nan_frac_tr <= MAX_NAN_FRAC
                    current_indices = np.where(sel_mask)[0]
                    current_cols    = [char_cols[i] for i in current_indices]
                    X_train         = X_train_all[:, current_indices]

                    model, feat_imp = train_model(X_train, y_train, current_cols, logger)
                    n_train_last = len(y_train)
                    logger.info(
                        f"  [retrain]  {t.date()}  n_train={n_train_last:,}  "
                        f"n_features={len(current_cols)}"
                    )
                except Exception as exc:
                    logger.error(f"{t.date()}  model training failed: {exc}")
                    if model is None:
                        continue   # no fallback model yet
            else:
                logger.warning(
                    f"{t.date()}  not enough training rows ({len(y_train)}), "
                    f"{'skipping' if model is None else 'keeping last model'}"
                )
                if model is None:
                    continue

        if model is None or current_cols is None:
            continue

        # ── Signal DataFrame ──────────────────────────────────────────────────
        if t not in slices:
            continue
        signal_df = slices[t][current_cols + [Y_COL]]

        # ── Signal generation ─────────────────────────────────────────────────
        predicted, realized = generate_signals(model, signal_df, current_cols)
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
            "n_train":        n_train_last,
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
            f"n_train={n_train_last:,}  "
            f"n_universe={summary['n_universe']:,}"
        )

        # ── Checkpoint flush ──────────────────────────────────────────────────
        if len(rows_buf) % CHECKPOINT_EVERY == 0:
            _flush(rows_buf, port_buf, imp_buf, first_flush)
            _save_checkpoint(
                step, t, prev_long_ids, prev_short_ids, n_train_last,
                nan_sum, obs_count, current_indices, current_cols,
            )
            elapsed = (datetime.now() - t_start).seconds // 60
            logger.info(f"  ✓ checkpoint saved  ({t.date()})  elapsed={elapsed}m")
            rows_buf, port_buf, imp_buf = (
                [], [], [])
            first_flush = False

    # ── Final flush ────────────────────────────────────────────────────────────
    if rows_buf:
        _flush(rows_buf, port_buf, imp_buf, first_flush)

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    logger.info("Finalising outputs …")
    returns_df = pd.read_csv(BACKTEST / "monthly_returns.csv", parse_dates=["eom"])
    logger.info(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")
    logger.info(f"  → {BACKTEST}/portfolios.csv")
    logger.info(f"  → {BACKTEST}/feature_importance.csv")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    logger.info(f"  → {BACKTEST}/performance.csv")

    sep = "=" * 62
    logger.info(sep)
    logger.info("  XGBOOST (EXPANDING WINDOW) — PERFORMANCE SUMMARY")
    logger.info(sep)
    logger.info(
        f"  Period         : {returns_df['eom'].min().date()} "
        f"→ {returns_df['eom'].max().date()}"
    )
    logger.info(f"  Months         : {perf['n_months']}")
    logger.info(f"  TC             : {tc_bps} bps one-way")
    logger.info(f"  Window         : expanding (all history)")
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
        description="Cross-sectional XGBoost (expanding window) equity backtest on JKP USA data."
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