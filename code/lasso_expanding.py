"""
Cross-sectional LASSO equity backtest with expanding training window — JKP 2023 USA data.

Timing (matches xg_boost_expanding.py exactly — no look-ahead bias):

  Burn-in: data from [data_start .. start_year-1] is used ONLY for training.
           No portfolio returns are generated during this period.

  For each scoring year Y (starting at start_year):
    1. TRAIN:  fit LassoCV on ALL cross-sections from data_start through Dec(Y-1).
               Each month's cross-section is imputed independently (cross-sectional
               mean) before stacking, then a single StandardScaler is fit on the
               full stacked training matrix.  LassoCV selects alpha via 5-fold CV.
               Training window expands by one full year each iteration.
    2. SCORE:  for each month t in Jan(Y) .. Dec(Y):
                   X  = characteristics[t]        (known at end of month t)
                   ŷ  = model.predict(X[t])
                   realized = ret_exc_lead1m[t]    (return earned in month t+1)

  Example (start_year=1986, burn_in_years=5 so data_start=1981):
    Train Model 1 on [1981-01 .. 1985-12]  →  score 1986-01 .. 1986-12
    Train Model 2 on [1981-01 .. 1986-12]  →  score 1987-01 .. 1987-12
    ...

Usage:
    python code/lasso_expanding.py                        # fresh run
    python code/lasso_expanding.py --resume               # continue from checkpoint
    python code/lasso_expanding.py --start 1986 --tc 5 --burn-in 5
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "lasso_expanding"

# ── LASSO hyperparameters ──────────────────────────────────────────────────────
CV_FOLDS  = 5
N_ALPHAS  = 50
MAX_ITER  = 10_000

# ── Backtest hyperparameters ───────────────────────────────────────────────────
DECILE           = 0.10
MIN_STOCKS       = 100
MAX_NAN_FRAC     = 0.30
CHECKPOINT_EVERY = 12
RET_CAP          = 4.0

# Identifier / flag columns — never used as predictors
_META = {
    "id", "date", "eom", "source_crsp", "size_grp",
    "obs_main", "exch_main", "primary_sec", "gvkey", "iid",
    "permno", "permco", "excntry", "curcd", "fx", "common",
    "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "adjfct", "shares", "me_lag1", "gics", "sic", "naics", "ff49",
}
# ret_1_0 is byte-for-byte identical to ret; drop it to avoid double feature weight.
_LOOKAHEAD = {"ret_exc_lead1m", "ret_1_0", "ret"}

Y_COL = "ret_exc_lead1m"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("lasso_expanding_backtest")
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
    # NaN filtering is deferred to each annual retraining step so that
    # feature selection uses only in-sample data (no look-ahead bias).

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
# 2. TRAINING ARRAY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_train_arrays(
    slices: dict[pd.Timestamp, pd.DataFrame],
    months: list[pd.Timestamp],
    char_cols: list[str],
    end_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack (X, y) for every month whose eom falls in any year <= end_year.

    Each month's cross-section is imputed independently with its own
    cross-sectional column means before stacking, so no future month's data
    influences the imputed values of any past month.

    Also returns per-column nan_sum and obs_count accumulators (computed on
    the raw, pre-imputation data) so that feature selection at training time
    uses only in-sample NaN rates.
    """
    X_list:    list[np.ndarray] = []
    y_list:    list[np.ndarray] = []
    nan_sum   = np.zeros(len(char_cols), dtype=np.float64)
    obs_count = np.zeros(len(char_cols), dtype=np.float64)

    for m in months:
        if m.year > end_year:
            break
        if m not in slices:
            continue
        frame = slices[m].dropna(subset=[Y_COL])
        frame = frame[frame[Y_COL].abs() <= RET_CAP]
        if len(frame) == 0:
            continue

        X_raw = frame[char_cols].values.astype(np.float64)
        nan_sum   += np.isnan(X_raw).sum(axis=0)
        obs_count += X_raw.shape[0]

        # Cross-sectional imputation: replace NaN with this month's column mean.
        col_means = np.nanmean(X_raw, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_mask  = np.isnan(X_raw)
        X_imp     = X_raw.copy()
        X_imp[nan_mask] = np.broadcast_to(col_means, X_raw.shape)[nan_mask]

        X_list.append(X_imp)
        y_list.append(frame[Y_COL].values.astype(np.float64))

    if not X_list:
        return (
            np.empty((0, len(char_cols)), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            nan_sum,
            obs_count,
        )

    return np.vstack(X_list), np.concatenate(y_list), nan_sum, obs_count


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    char_cols: list[str],
    logger: logging.Logger,
) -> tuple[LassoCV, StandardScaler, pd.Series]:
    """
    Standardize the (already-imputed) training matrix and fit LassoCV.

    StandardScaler is fit on all training rows so that the scale is set by the
    full in-sample distribution.  LassoCV selects alpha via CV_FOLDS-fold CV.
    """
    scaler  = StandardScaler()
    X_std   = scaler.fit_transform(X_train)

    model = LassoCV(
        cv=CV_FOLDS,
        n_alphas=N_ALPHAS,
        max_iter=MAX_ITER,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_std, y_train)

    coefs = pd.Series(model.coef_, index=char_cols, name="coef")
    n_nonzero = int((coefs != 0).sum())
    logger.debug(
        f"    LASSO: alpha={model.alpha_:.4e}  n_nonzero={n_nonzero}"
    )
    return model, scaler, coefs


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    model: LassoCV,
    scaler: StandardScaler,
    signal_df: pd.DataFrame,
    char_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    """
    Apply the fitted model to characteristics at month t → predicted return t+1.

    Missing feature values are imputed with the scoring month's own cross-sectional
    column means (no information from future months).  The fitted scaler (from the
    training data) is then applied before prediction.
    """
    if signal_df.empty:
        return None, None

    X_raw = signal_df[char_cols].values.astype(np.float64)

    col_means = np.nanmean(X_raw, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask  = np.isnan(X_raw)
    X_imp     = X_raw.copy()
    X_imp[nan_mask] = np.broadcast_to(col_means, X_raw.shape)[nan_mask]

    X_std     = scaler.transform(X_imp)
    pred      = model.predict(X_std)
    predicted = pd.Series(pred.astype(float), index=signal_df.index, name="predicted_ret")
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
# 6. PERFORMANCE EVALUATION
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
# 7. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        "LASSO (Expanding Window) Long-Short Backtest (Top/Bottom Decile)",
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
# 8. CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"


def _save_checkpoint(
    scoring_year: int,
    prev_long_ids: set,
    prev_short_ids: set,
    n_train_rows: int,
) -> None:
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "scoring_year":   scoring_year,
            "prev_long_ids":  prev_long_ids  or set(),
            "prev_short_ids": prev_short_ids or set(),
            "n_train_rows":   n_train_rows,
        }, f)


def _load_checkpoint() -> dict | None:
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
# 9. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int    = 1986,
    burn_in_years: int = 5,
    tc_bps: float      = 10.0,
    resume: bool       = False,
) -> pd.DataFrame:
    """
    Expanding-window LASSO backtest.

    Structure:
      data_start = start_year - burn_in_years

      For scoring_year in [start_year, start_year+1, ...]:
          train_end_year = scoring_year - 1
          Train on ALL months in [data_start .. train_end_year]
          Score every month t in scoring_year:
              predict using characteristics[t]
              realized = ret_exc_lead1m[t]  (return in t+1)

    The training window grows by 12 months each year; no data from the
    scoring year (or beyond) ever enters the training set.
    """
    BACKTEST.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(BACKTEST)

    data_start_year = start_year - burn_in_years
    logger.info(
        f"Config: data_start={data_start_year}  start_year={start_year}  "
        f"burn_in={burn_in_years}yr  tc={tc_bps}bps  "
        f"cv_folds={CV_FOLDS}  n_alphas={N_ALPHAS}  "
        f"max_iter={MAX_ITER}  window=expanding"
    )

    slices, char_cols = load_data(logger)
    months      = sorted(slices.keys())
    last_data_year  = max(m.year for m in months)
    scoring_years   = [y for y in range(start_year, last_data_year + 1)]

    logger.info(
        f"Scoring years: {scoring_years[0]} → {scoring_years[-1]}  "
        f"({len(scoring_years)} annual retrains)"
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    resume_from_year: int | None = None
    prev_long_ids:   set | None  = None
    prev_short_ids:  set | None  = None
    first_flush = True

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found — starting fresh.")
        else:
            resume_from_year = ckpt["scoring_year"] + 1
            prev_long_ids    = ckpt["prev_long_ids"]
            prev_short_ids   = ckpt["prev_short_ids"]
            first_flush      = False
            logger.info(
                f"Resuming from checkpoint: next scoring_year={resume_from_year}  "
                f"n_train_rows={ckpt['n_train_rows']:,}"
            )

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    coef_buf: list[pd.Series]    = []
    t_start = datetime.now()
    scored_months = 0

    # ── Main loop: one iteration = one scoring year ───────────────────────────
    for scoring_year in scoring_years:

        if resume_from_year is not None and scoring_year < resume_from_year:
            continue

        train_end_year = scoring_year - 1

        if train_end_year < data_start_year:
            logger.warning(
                f"scoring_year={scoring_year}: train_end_year={train_end_year} "
                f"< data_start_year={data_start_year}, skipping."
            )
            continue

        # ── Build full expanding training set ─────────────────────────────────
        logger.info(
            f"[{scoring_year}]  Training on [{data_start_year}-01 .. "
            f"{train_end_year}-12] …"
        )
        X_all, y_all, nan_sum, obs_count = _build_train_arrays(
            slices, months, char_cols, end_year=train_end_year
        )

        if len(y_all) < MIN_STOCKS:
            logger.warning(
                f"[{scoring_year}]  Only {len(y_all)} training rows — skipping."
            )
            continue

        # ── Feature selection: NaN fraction computed over training data only ──
        nan_frac_tr     = np.where(obs_count > 0, nan_sum / obs_count, 1.0)
        sel_mask        = nan_frac_tr <= MAX_NAN_FRAC
        current_indices = np.where(sel_mask)[0]
        current_cols    = [char_cols[j] for j in current_indices]
        X_train         = X_all[:, current_indices]

        # ── Train LASSO for this scoring year ─────────────────────────────────
        try:
            model, scaler, coefs = train_model(X_train, y_all, current_cols, logger)
            n_train_rows = len(y_all)
            n_nonzero    = int((coefs != 0).sum())
            logger.info(
                f"[{scoring_year}]  Model trained  "
                f"n_train={n_train_rows:,}  "
                f"n_features={len(current_cols)}  "
                f"alpha={model.alpha_:.4e}  "
                f"n_nonzero={n_nonzero}"
            )
        except Exception as exc:
            logger.error(f"[{scoring_year}]  Training failed: {exc}")
            continue

        # ── Score every month in scoring_year ─────────────────────────────────
        score_months = [m for m in months if m.year == scoring_year]

        for t in score_months:
            if t not in slices:
                continue

            signal_df = slices[t][current_cols + [Y_COL]]

            # Drop penny stocks (price < $5). CRSP uses negative prc to record
            # bid-ask midpoints when a closing price is unavailable, so filter
            # on the absolute value. Apply only at portfolio formation — training
            # data keeps all stocks so the model learns the full return distribution.
            if "prc" in slices[t].columns:
                prc = slices[t]["prc"].reindex(signal_df.index)
                signal_df = signal_df[(prc.abs() >= 5.0) | prc.isna()]

            predicted, realized = generate_signals(model, scaler, signal_df, current_cols)
            if predicted is None:
                logger.warning(f"  {t.date()}  skipped — empty signal DataFrame")
                continue

            ic_p, ic_s = compute_ic(predicted, realized)

            port_df, summary = construct_portfolio(
                predicted, realized, prev_long_ids, prev_short_ids, tc_bps
            )
            if port_df is None:
                logger.warning(
                    f"  {t.date()}  skipped — insufficient stocks with forward returns"
                )
                continue

            prev_long_ids  = set(port_df.loc[port_df["leg"] == "long",  "id"])
            prev_short_ids = set(port_df.loc[port_df["leg"] == "short", "id"])

            row = {
                "eom":            t,
                "ic_pearson":     ic_p,
                "ic_spearman":    ic_s,
                "alpha":          model.alpha_,
                "n_nonzero":      n_nonzero,
                "n_train":        n_train_rows,
                "scoring_year":   scoring_year,
                "train_end_year": train_end_year,
                **summary,
            }
            rows_buf.append(row)
            port_df["eom"] = t
            port_buf.append(port_df)

            coef_row      = coefs.reindex(char_cols)  # NaN for cols absent this year
            coef_row.name = t
            coef_buf.append(coef_row)

            scored_months += 1
            logger.info(
                f"  {t.date()}  |  "
                f"qspread={summary['qspread']:+.4f}  "
                f"IC={ic_s:+.3f}  "
                f"n_universe={summary['n_universe']:,}"
            )

        # ── Checkpoint + flush at end of each scoring year ────────────────────
        if rows_buf:
            _flush(rows_buf, port_buf, coef_buf, first_flush)
            _save_checkpoint(scoring_year, prev_long_ids or set(), prev_short_ids or set(), n_train_rows)
            elapsed = (datetime.now() - t_start).seconds // 60
            logger.info(
                f"  ✓ checkpoint saved after scoring_year={scoring_year}  "
                f"scored_months={scored_months}  elapsed={elapsed}m"
            )
            rows_buf, port_buf, coef_buf = [], [], []
            first_flush = False

    # ── Final flush (any remaining buffer) ────────────────────────────────────
    if rows_buf:
        _flush(rows_buf, port_buf, coef_buf, first_flush)

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    logger.info("Finalising outputs …")
    returns_df = pd.read_csv(BACKTEST / "monthly_returns.csv", parse_dates=["eom"])
    logger.info(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")
    logger.info(f"  → {BACKTEST}/portfolios.csv")
    logger.info(f"  → {BACKTEST}/coefs.csv")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    logger.info(f"  → {BACKTEST}/performance.csv")

    sep = "=" * 62
    logger.info(sep)
    logger.info("  LASSO (EXPANDING WINDOW) — PERFORMANCE SUMMARY")
    logger.info(sep)
    logger.info(
        f"  Period         : {returns_df['eom'].min().date()} "
        f"→ {returns_df['eom'].max().date()}"
    )
    logger.info(f"  Months         : {perf['n_months']}")
    logger.info(f"  TC             : {tc_bps} bps one-way")
    logger.info(f"  Burn-in        : {burn_in_years} years ({data_start_year}-{start_year-1})")
    logger.info(f"  Window         : expanding (all history from {data_start_year})")
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
        description="Cross-sectional LASSO (expanding window) equity backtest on JKP USA data."
    )
    parser.add_argument(
        "--start", type=int, default=1986,
        help="First year for which portfolio returns are generated (default: 1986).",
    )
    parser.add_argument(
        "--burn-in", type=int, default=5,
        help="Number of years of burn-in before scoring begins (default: 5). "
             "Data start = start - burn_in.",
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
    run_backtest(
        start_year=args.start,
        burn_in_years=args.burn_in,
        tc_bps=args.tc,
        resume=args.resume,
    )