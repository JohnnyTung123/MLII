"""
Cross-sectional XGBoost equity backtest — enhanced pipeline — JKP 2023 USA data.

Four improvements stacked on top of xg_boost_expanding.py:

  1. Per-month cross-sectional rank normalization (X and Y)
       Each feature column and the return target are mapped to [-0.5, 0.5]
       within the monthly cross-section before entering training.  This
       neutralises outliers, makes each feature's distribution uniform
       across time, and aligns the RMSE objective with rank correlation (IC).

  2. IC-aligned objective (RMSE on rank-normalized targets)
       By rank-normalizing Y as well as X, minimising RMSE is equivalent to
       maximising cross-sectional rank correlation — no custom gradient needed.

  3. Ensemble across random seeds
       ENSEMBLE_SEEDS independent models are trained per year and their raw
       predictions are averaged before ranking into the portfolio.  Diversity
       across seeds (subsample / colsample) reduces prediction variance.

  4. Volatility-targeting the spread
       The Q-spread is scaled each month so the rolling realised vol matches
       VOL_TARGET.  Raw and vol-targeted returns are both recorded.

Usage:
    python code/xg_boost_enhanced.py
    python code/xg_boost_enhanced.py --start 1986 --tc 10
    python code/xg_boost_enhanced.py --resume
"""

import argparse
import logging
import pickle
import warnings
from collections import deque
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
from portfolio_fixed import construct_portfolio_vw

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "xgboost_enhanced"

# ── XGBoost hyperparameters ────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators":       300,
    "learning_rate":      0.05,
    "max_depth":          4,
    "subsample":          0.8,
    "colsample_bytree":   0.5,
    "min_child_weight":   20,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "tree_method":        "hist",
    "n_jobs":             -1,
    "verbosity":          0,
}
EARLY_STOPPING_ROUNDS = 30
VAL_FRAC              = 0.20

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 456]   # add seeds to grow ensemble; each adds one retrain/year

# ── Volatility targeting ───────────────────────────────────────────────────────
VOL_TARGET    = 0.10   # annualised target volatility for the Q-spread
VOL_LOOKBACK  = 36     # rolling window (months) for vol estimation
VOL_SCALE_CAP = 2.0    # cap on the leverage multiple (both up and down)

# ── Backtest hyperparameters ───────────────────────────────────────────────────
DECILE           = 0.10
MIN_STOCKS       = 100
MAX_NAN_FRAC     = 0.30
CHECKPOINT_EVERY = 12
RET_CAP          = 1.0

_META = {
    "id", "date", "eom", "source_crsp", "size_grp",
    "obs_main", "exch_main", "primary_sec", "gvkey", "iid",
    "permno", "permco", "excntry", "curcd", "fx", "common",
    "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd",
    "adjfct", "shares", "me_lag1", "gics", "sic", "naics", "ff49", "ret_lag_dif",
}
_LOOKAHEAD = {"ret_exc_lead1m"}

Y_COL = "ret_exc_lead1m"


# ══════════════════════════════════════════════════════════════════════════════
# RANK NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def rank_normalize(X: np.ndarray) -> np.ndarray:
    """
    Per-column cross-sectional rank normalization to (-0.5, 0.5].
    NaN entries are excluded from ranking and stay NaN in output.
    Pure numpy — avoids pandas DataFrame overhead on every monthly slice.
    """
    out = np.full(X.shape, np.nan, dtype=np.float32)
    for j in range(X.shape[1]):
        col  = X[:, j]
        mask = np.isfinite(col)
        nv   = int(mask.sum())
        if nv == 0:
            continue
        r = np.empty(nv, dtype=np.float32)
        r[np.argsort(col[mask], kind="quicksort")] = np.arange(nv, dtype=np.float32)
        out[mask, j] = r / nv - 0.5
    return out


def rank_normalize_1d(y: np.ndarray) -> np.ndarray:
    """Rank-normalize a 1-D target vector to (-0.5, 0.5]. No NaN expected (Y is pre-filtered)."""
    n = len(y)
    r = np.empty(n, dtype=np.float32)
    r[np.argsort(y, kind="quicksort")] = np.arange(n, dtype=np.float32)
    return r / n - 0.5


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY TARGETING
# ══════════════════════════════════════════════════════════════════════════════

def vol_scale(recent_spreads: deque, target_vol: float = VOL_TARGET) -> float:
    """
    Return the multiplier that scales this month's spread to target_vol.
    Requires at least 12 observations; capped at VOL_SCALE_CAP on both sides.
    """
    if len(recent_spreads) < 12:
        return 1.0
    realised = float(np.std(recent_spreads)) * np.sqrt(12)
    if realised <= 0:
        return 1.0
    raw = target_vol / realised
    return float(np.clip(raw, 1.0 / VOL_SCALE_CAP, VOL_SCALE_CAP))


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xgb_enhanced")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dir / "run.log", mode="a", encoding="utf-8")
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
# 2. MODEL ENSEMBLE TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    char_cols: list[str],
    logger: logging.Logger,
) -> tuple[list[XGBRegressor], pd.Series]:
    """
    Train one XGBRegressor per seed; return list of models and averaged
    feature importances.  The validation split is identical for all seeds
    so diversity comes from subsampling randomness only.
    """
    n     = len(y_train)
    split = max(MIN_STOCKS, int(n * (1 - VAL_FRAC)))
    split = min(split, n - 1)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    models:    list[XGBRegressor] = []
    feat_imps: list[pd.Series]   = []

    for seed in ENSEMBLE_SEEDS:
        m = XGBRegressor(
            **{**XGB_PARAMS, "random_state": seed},
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="rmse",
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models.append(m)
        feat_imps.append(pd.Series(m.feature_importances_, index=char_cols))
        logger.debug(f"    seed={seed}  best_iter={m.best_iteration}")

    avg_imp      = pd.concat(feat_imps, axis=1).mean(axis=1)
    avg_imp.name = "importance"
    return models, avg_imp


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL GENERATION  (ensemble predict on rank-normalized cross-section)
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    models: list[XGBRegressor],
    signal_df: pd.DataFrame,
    char_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    if signal_df.empty:
        return None, None

    X_raw  = signal_df[char_cols].values.astype(np.float32)
    X_norm = rank_normalize(X_raw)

    # Average raw predictions across ensemble, then use for ranking
    preds = np.stack([m.predict(X_norm) for m in models], axis=0)
    pred  = preds.mean(axis=0)

    predicted = pd.Series(pred, index=signal_df.index, name="predicted_ret")
    realized  = signal_df[Y_COL].rename("realized_ret")
    return predicted, realized


# ══════════════════════════════════════════════════════════════════════════════
# 4. IC + PERFORMANCE
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

    ic_p   = returns["ic_pearson"].dropna()
    ic_s   = returns["ic_spearman"].dropna()
    icir_p = ic_p.mean() / ic_p.std() * np.sqrt(12) if ic_p.std() > 0 else np.nan
    icir_s = ic_s.mean() / ic_s.std() * np.sqrt(12) if ic_s.std() > 0 else np.nan

    return {
        **_stats(returns["qspread"],        "gross"),
        **_stats(returns["net_qspread"],    "net"),
        **_stats(returns["qspread_vt"],     "gross_vt"),
        **_stats(returns["net_qspread_vt"], "net_vt"),
        "ic_mean_pearson":  round(float(ic_p.mean()),  4),
        "ic_mean_spearman": round(float(ic_s.mean()),  4),
        "icir_pearson":     round(float(icir_p),       4),
        "icir_spearman":    round(float(icir_s),       4),
        "n_months":         int(returns["qspread"].notna().sum()),
        "tc_bps":           tc_bps,
        "vol_target":       VOL_TARGET,
        "n_seeds":          len(ENSEMBLE_SEEDS),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(returns: pd.DataFrame, out_dir: Path, logger: logging.Logger) -> None:
    dates = pd.to_datetime(returns["eom"])

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"XGBoost Enhanced (rank-norm · IC obj · {len(ENSEMBLE_SEEDS)}-seed ensemble · "
        f"{VOL_TARGET:.0%} vol-target) — Long-Short Backtest",
        fontsize=11, y=0.98,
    )

    # Panel 1: cumulative returns — raw gross/net + vol-targeted net
    ax = axes[0]
    gross_cum    = (1 + returns["qspread"]).cumprod()
    net_cum      = (1 + returns["net_qspread"]).cumprod()
    net_vt_cum   = (1 + returns["net_qspread_vt"]).cumprod()
    ax.plot(dates, gross_cum,  label="Gross (raw)",      lw=1.4, color="steelblue")
    ax.plot(dates, net_cum,    label="Net (raw)",         lw=1.4, color="darkorange",  ls="--")
    ax.plot(dates, net_vt_cum, label=f"Net VT ({VOL_TARGET:.0%})", lw=1.6, color="seagreen")
    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.fill_between(dates, net_vt_cum, 1, where=(net_vt_cum < 1), alpha=0.12, color="red")
    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Q-Spread Return")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel 2: rolling Sharpe on vol-targeted gross
    ax = axes[1]
    for col, color, ls, lbl in [
        ("qspread_vt",  "seagreen",   "-",  f"Gross VT ({VOL_TARGET:.0%})"),
        ("qspread",     "steelblue",  "--", "Gross (raw)"),
    ]:
        roll     = returns[col].rolling(12)
        roll_std = roll.std()
        rs       = np.where(roll_std > 0, roll.mean() / roll_std * np.sqrt(12), np.nan)
        ax.plot(dates, rs, color=color, lw=1.4, ls=ls, label=lbl)
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Sharpe (annualised)")
    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel 3: IC
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
# 6. CHECKPOINT + FLUSH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"


def _save_checkpoint(
    scoring_year: int,
    prev_weights: dict,
    n_train_rows: int,
    recent_spreads: deque,
) -> None:
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "scoring_year":   scoring_year,
            "prev_weights":   prev_weights or {},
            "n_train_rows":   n_train_rows,
            "recent_spreads": list(recent_spreads),
        }, f)


def _load_checkpoint() -> dict | None:
    if _CKPT_PATH.exists():
        with open(_CKPT_PATH, "rb") as f:
            return pickle.load(f)
    return None


def _flush(
    rows_buf: list[dict],
    port_buf: list[pd.DataFrame],
    imp_buf:  list[pd.Series],
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
# 7. TRAINING ARRAY BUILDER  (rank-normalize X and Y per month)
# ══════════════════════════════════════════════════════════════════════════════

def _build_train_arrays_range(
    slices: dict[pd.Timestamp, pd.DataFrame],
    months: list[pd.Timestamp],
    char_cols: list[str],
    start_year: int,
    end_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack rank-normalized (X, y) for months in [start_year, end_year] only.

    Called incrementally: each scoring year appends ~12 new months instead of
    rebuilding the full history (O(N) work per year vs O(N²) total).
    NaN counts use raw X values so the NaN filter reflects true availability.
    """
    X_list:    list[np.ndarray] = []
    y_list:    list[np.ndarray] = []
    nan_sum   = np.zeros(len(char_cols), dtype=np.float64)
    obs_count = np.zeros(len(char_cols), dtype=np.float64)

    for m in months:
        if m.year < start_year:
            continue
        if m.year > end_year:
            break
        if m not in slices:
            continue
        frame = slices[m].dropna(subset=[Y_COL])
        frame = frame[frame[Y_COL].abs() <= RET_CAP]
        if len(frame) == 0:
            continue

        X_raw = frame[char_cols].values.astype(np.float32)
        y_raw = frame[Y_COL].values.astype(np.float32)

        nan_sum   += np.isnan(X_raw).sum(axis=0)
        obs_count += X_raw.shape[0]

        X_list.append(rank_normalize(X_raw))
        y_list.append(rank_normalize_1d(y_raw))

    if not X_list:
        return (
            np.empty((0, len(char_cols)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            nan_sum, obs_count,
        )
    return np.vstack(X_list), np.concatenate(y_list), nan_sum, obs_count


# ══════════════════════════════════════════════════════════════════════════════
# 8. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int    = 1986,
    burn_in_years: int = 5,
    tc_bps: float      = 10.0,
    resume: bool       = False,
) -> pd.DataFrame:
    BACKTEST.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(BACKTEST)

    data_start_year = start_year - burn_in_years
    logger.info(
        f"Config: data_start={data_start_year}  start_year={start_year}  "
        f"burn_in={burn_in_years}yr  tc={tc_bps}bps  "
        f"rank_normalize=X+Y  n_seeds={len(ENSEMBLE_SEEDS)}  "
        f"vol_target={VOL_TARGET:.0%}  vol_lookback={VOL_LOOKBACK}m  "
        f"max_depth={XGB_PARAMS['max_depth']}  "
        f"lr={XGB_PARAMS['learning_rate']}  "
        f"n_estimators={XGB_PARAMS['n_estimators']}  "
        f"early_stopping={EARLY_STOPPING_ROUNDS}  val_frac={VAL_FRAC}"
    )

    slices, char_cols = load_data(logger)
    months            = sorted(slices.keys())

    last_data_year = max(m.year for m in months)
    scoring_years  = list(range(start_year, last_data_year + 1))

    logger.info(
        f"Scoring years: {scoring_years[0]} → {scoring_years[-1]}  "
        f"({len(scoring_years)} annual retrains × {len(ENSEMBLE_SEEDS)} seeds)"
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    resume_from_year: int | None = None
    prev_weights: dict | None    = None
    recent_spreads               = deque(maxlen=VOL_LOOKBACK)
    first_flush                  = True

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found — starting fresh.")
        else:
            resume_from_year = ckpt["scoring_year"] + 1
            prev_weights     = ckpt.get("prev_weights")
            recent_spreads.extend(ckpt.get("recent_spreads", []))
            first_flush      = False
            logger.info(
                f"Resuming from checkpoint: next scoring_year={resume_from_year}  "
                f"n_train_rows={ckpt['n_train_rows']:,}  "
                f"spread_buffer={len(recent_spreads)}m"
            )

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    imp_buf:  list[pd.Series]    = []
    t_start       = datetime.now()
    scored_months = 0

    # ── Incremental training cache ─────────────────────────────────────────────
    # Build the cache once through the year before the first scoring year we
    # will actually process.  After that, each scoring year extends the cache
    # by exactly 12 months — O(N) total work instead of O(N²).
    first_score_year = resume_from_year if resume_from_year else start_year
    cache_through    = first_score_year - 1

    logger.info(f"Building initial training cache [{data_start_year} .. {cache_through}] …")
    X_cache, y_cache, nan_sum_cache, obs_count_cache = _build_train_arrays_range(
        slices, months, char_cols,
        start_year=data_start_year,
        end_year=cache_through,
    )
    cache_end_year = cache_through
    logger.info(f"  Cache ready: {len(y_cache):,} rows")

    # ── Main loop ─────────────────────────────────────────────────────────────
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

        # Extend cache by one year (the year just added to the training window)
        if train_end_year > cache_end_year:
            X_new, y_new, ns_new, oc_new = _build_train_arrays_range(
                slices, months, char_cols,
                start_year=cache_end_year + 1,
                end_year=train_end_year,
            )
            if len(y_new) > 0:
                X_cache         = np.vstack([X_cache, X_new])
                y_cache         = np.concatenate([y_cache, y_new])
                nan_sum_cache  += ns_new
                obs_count_cache += oc_new
            cache_end_year = train_end_year

        X_all     = X_cache
        y_all     = y_cache
        nan_sum   = nan_sum_cache
        obs_count = obs_count_cache

        logger.info(
            f"[{scoring_year}]  Training on [{data_start_year}-01 .. "
            f"{train_end_year}-12]  ({len(ENSEMBLE_SEEDS)} seeds) …"
        )

        if len(y_all) < MIN_STOCKS:
            logger.warning(
                f"[{scoring_year}]  Only {len(y_all)} training rows — skipping."
            )
            continue

        # ── NaN filter (in-sample; raw counts) ────────────────────────────────
        nan_frac_tr     = np.where(obs_count > 0, nan_sum / obs_count, 1.0)
        sel_mask        = nan_frac_tr <= MAX_NAN_FRAC
        current_indices = np.where(sel_mask)[0]
        current_cols    = [char_cols[j] for j in current_indices]
        X_train         = X_all[:, current_indices]

        # ── Train ensemble ─────────────────────────────────────────────────────
        try:
            models, feat_imp = train_ensemble(X_train, y_all, current_cols, logger)
            n_train_rows = len(y_all)
            best_iters   = [m.best_iteration for m in models]
            logger.info(
                f"[{scoring_year}]  Ensemble trained  "
                f"n_train={n_train_rows:,}  "
                f"n_features={len(current_cols)}  "
                f"best_iters={best_iters}"
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

            if "prc" in slices[t].columns:
                prc       = slices[t]["prc"].reindex(signal_df.index)
                signal_df = signal_df[(prc.abs() >= 5.0) | prc.isna()]

            predicted, realized = generate_signals(models, signal_df, current_cols)
            if predicted is None:
                logger.warning(f"  {t.date()}  skipped — empty signal DataFrame")
                continue

            ic_p, ic_s = compute_ic(predicted, realized)

            me_t = slices[t]["me_company"].reindex(signal_df.index)

            port_df, summary = construct_portfolio_vw(
                predicted, realized, me_t,
                prev_weights, tc_bps,
                cap_pctl=0.80,
                micro_pctl=0.20,
                ret_cap=RET_CAP,
                decile=DECILE,
            )
            if port_df is None:
                logger.warning(
                    f"  {t.date()}  skipped — insufficient stocks with forward returns"
                )
                continue

            prev_weights = dict(zip(port_df["id"], port_df["weight"]))

            # ── Volatility targeting ───────────────────────────────────────────
            # Scale is computed from returns BEFORE this month (no look-ahead).
            scale             = vol_scale(recent_spreads)
            qspread_vt        = summary["qspread"]     * scale
            net_qspread_vt    = summary["net_qspread"] * scale
            recent_spreads.append(summary["qspread"])  # update buffer after scaling

            row = {
                "eom":            t,
                "ic_pearson":     ic_p,
                "ic_spearman":    ic_s,
                "best_iter_mean": float(np.mean(best_iters)),
                "n_seeds":        len(ENSEMBLE_SEEDS),
                "n_train":        n_train_rows,
                "scoring_year":   scoring_year,
                "train_end_year": train_end_year,
                "vol_scale":      scale,
                "qspread_vt":     qspread_vt,
                "net_qspread_vt": net_qspread_vt,
                **summary,
            }
            rows_buf.append(row)
            port_df["eom"] = t
            port_buf.append(port_df)
            imp_row      = feat_imp.copy()
            imp_row.name = t
            imp_buf.append(imp_row)

            scored_months += 1
            logger.info(
                f"  {t.date()}  |  "
                f"qspread={summary['qspread']:+.4f}  "
                f"qspread_vt={qspread_vt:+.4f}  "
                f"vol_scale={scale:.2f}  "
                f"IC={ic_s:+.3f}  "
                f"n_universe={summary['n_universe']:,}"
            )

        # ── Checkpoint + flush ─────────────────────────────────────────────────
        if rows_buf:
            _flush(rows_buf, port_buf, imp_buf, first_flush)
            _save_checkpoint(scoring_year, prev_weights or {}, n_train_rows, recent_spreads)
            elapsed = (datetime.now() - t_start).seconds // 60
            logger.info(
                f"  ✓ checkpoint  scoring_year={scoring_year}  "
                f"scored_months={scored_months}  elapsed={elapsed}m"
            )
            rows_buf, port_buf, imp_buf = [], [], []
            first_flush = False

    if rows_buf:
        _flush(rows_buf, port_buf, imp_buf, first_flush)

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    logger.info("Finalising outputs …")
    returns_df = pd.read_csv(BACKTEST / "monthly_returns.csv", parse_dates=["eom"])
    logger.info(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    logger.info(f"  → {BACKTEST}/performance.csv")

    sep = "=" * 62
    logger.info(sep)
    logger.info("  XGBOOST ENHANCED — PERFORMANCE SUMMARY")
    logger.info(sep)
    logger.info(
        f"  Period         : {returns_df['eom'].min().date()} "
        f"→ {returns_df['eom'].max().date()}"
    )
    logger.info(f"  Months         : {perf['n_months']}")
    logger.info(f"  TC             : {tc_bps} bps one-way")
    logger.info(f"  Seeds          : {len(ENSEMBLE_SEEDS)}  ({ENSEMBLE_SEEDS})")
    logger.info(f"  Vol target     : {VOL_TARGET:.0%}  (lookback {VOL_LOOKBACK}m)")
    logger.info(f"  Burn-in        : {burn_in_years} years ({data_start_year}-{start_year-1})")
    logger.info("  ── Raw Q-spread ─────────────────────────────────")
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
    logger.info(f"  ── Vol-targeted net ({VOL_TARGET:.0%}) ──────────────────")
    logger.info(f"  Ann. Return    : {perf['net_vt_ann_ret']:>8.2%}")
    logger.info(f"  Ann. Volatility: {perf['net_vt_ann_vol']:>8.2%}")
    logger.info(f"  Sharpe Ratio   : {perf['net_vt_sharpe']:>8.2f}")
    logger.info(f"  Max Drawdown   : {perf['net_vt_max_dd']:>8.2%}")
    logger.info(f"  Win Rate       : {perf['net_vt_win_rate']:>8.2%}")
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
        description="XGBoost enhanced backtest: rank-norm + IC obj + ensemble + vol-target."
    )
    parser.add_argument(
        "--start", type=int, default=1986,
        help="First scoring year (default: 1986).",
    )
    parser.add_argument(
        "--burn-in", type=int, default=5,
        help="Burn-in years before scoring (default: 5).",
    )
    parser.add_argument(
        "--tc", type=float, default=10.0,
        help="One-way transaction cost in bps (default: 10).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last saved checkpoint.",
    )
    args = parser.parse_args()
    run_backtest(
        start_year=args.start,
        burn_in_years=args.burn_in,
        tc_bps=args.tc,
        resume=args.resume,
    )