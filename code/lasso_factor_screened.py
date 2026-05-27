"""
Cross-sectional LASSO equity backtest with expanding training window +
in-sample factor screening — JKP 2023 USA data.

Before each annual retrain, every candidate feature is evaluated as a
standalone equal-weighted long-short decile factor using only the months
in the training window (data_start .. scoring_year-1).  Features whose
annualised Sharpe ratio falls below SHARPE_THRESHOLD are dropped before
LASSO training.

No look-ahead bias: screening uses only data that is strictly in-sample
for the corresponding scoring year — the same expanding window the model
trains on.

Usage:
    python code/lasso_factor_screened.py
    python code/lasso_factor_screened.py --start 1986 --tc 10 --sharpe-threshold 0.8
    python code/lasso_factor_screened.py --resume
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from portfolio_fixed import construct_portfolio_vw

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent.parent
PROCESSED = _ROOT / "data" / "processed"
BACKTEST  = _ROOT / "data" / "backtest" / "lasso_factor_screened"

# ── LASSO hyperparameters ──────────────────────────────────────────────────────
CV_FOLDS = 5
N_ALPHAS = 50
MAX_ITER = 10_000

# ── Backtest hyperparameters ───────────────────────────────────────────────────
DECILE           = 0.10
MIN_STOCKS       = 100
MAX_NAN_FRAC     = 0.30
SHARPE_THRESHOLD = 0.8   # minimum in-sample factor Sharpe to enter the model
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
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lasso_factor_screened")
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
# 2. IN-SAMPLE FACTOR SCREENING
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_sharpes(
    slices: dict[pd.Timestamp, pd.DataFrame],
    months: list[pd.Timestamp],
    candidate_cols: list[str],
    data_start_year: int,
    end_year: int,
) -> pd.Series:
    """
    For each factor in candidate_cols, compute the annualised Sharpe ratio of
    its equal-weighted long-short decile portfolio using only training months
    in [data_start_year, end_year].  NaN Sharpe = fewer than 12 valid months.
    """
    train_months = [
        m for m in months
        if data_start_year <= m.year <= end_year and m in slices
    ]

    n_months  = len(train_months)
    n_factors = len(candidate_cols)
    ret_mat   = np.full((n_months, n_factors), np.nan, dtype=np.float64)

    for i, m in enumerate(train_months):
        frame = slices[m].dropna(subset=[Y_COL])
        frame = frame[frame[Y_COL].abs() <= RET_CAP]
        if len(frame) < MIN_STOCKS // 2:
            continue

        y = frame[Y_COL].values.astype(np.float64)
        X = frame[candidate_cols].values.astype(np.float64)

        for j in range(n_factors):
            x_j   = X[:, j]
            valid = np.isfinite(x_j)
            n_v   = int(valid.sum())
            n_d   = max(1, int(n_v * DECILE))
            if n_v < 2 * n_d:
                continue
            order         = np.argsort(x_j[valid])
            y_v           = y[valid]
            ret_mat[i, j] = y_v[order[-n_d:]].mean() - y_v[order[:n_d]].mean()

    sharpes = np.full(n_factors, np.nan, dtype=np.float64)
    for j in range(n_factors):
        s = ret_mat[:, j]
        s = s[np.isfinite(s)]
        if len(s) < 12:
            continue
        ann_vol = s.std() * np.sqrt(12)
        sharpes[j] = s.mean() * 12 / ann_vol if ann_vol > 0 else np.nan

    return pd.Series(sharpes, index=candidate_cols, name="sharpe")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    char_cols: list[str],
    logger: logging.Logger,
) -> tuple[Pipeline, pd.Series]:
    """
    Fit impute → scale → LassoCV on the stacked training matrix.
    The pipeline handles NaN imputation so raw X can be passed directly.
    Returns the fitted pipeline and absolute coefficients as feature importances.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("lasso",   LassoCV(
            cv=CV_FOLDS, n_alphas=N_ALPHAS, max_iter=MAX_ITER,
            n_jobs=-1, random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    lasso     = pipe.named_steps["lasso"]
    coefs     = np.abs(lasso.coef_)
    n_nonzero = int((coefs > 0).sum())
    feat_imp  = pd.Series(coefs, index=char_cols, name="importance")
    logger.debug(
        f"    LASSO: best_alpha={lasso.alpha_:.6f}  "
        f"n_nonzero={n_nonzero}/{len(char_cols)}  "
        f"top_feature={feat_imp.idxmax()}"
    )
    return pipe, feat_imp


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    pipe: Pipeline,
    signal_df: pd.DataFrame,
    char_cols: list[str],
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    if signal_df.empty:
        return None, None

    X_raw     = signal_df[char_cols].values.astype(np.float64)
    pred      = pipe.predict(X_raw)
    predicted = pd.Series(pred, index=signal_df.index, name="predicted_ret")
    realized  = signal_df[Y_COL].rename("realized_ret")
    return predicted, realized


# ══════════════════════════════════════════════════════════════════════════════
# 5. IC + PERFORMANCE
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
        **_stats(returns["qspread"],     "gross"),
        **_stats(returns["net_qspread"], "net"),
        "ic_mean_pearson":  round(float(ic_p.mean()),  4),
        "ic_mean_spearman": round(float(ic_s.mean()),  4),
        "icir_pearson":     round(float(icir_p),       4),
        "icir_spearman":    round(float(icir_s),       4),
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
        "LASSO + Factor Screening (Sharpe > 0.8) — Long-Short Backtest",
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
    roll        = returns["qspread"].rolling(12)
    roll_std    = roll.std()
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
# 7. CHECKPOINT + FLUSH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_CKPT_PATH = BACKTEST / "checkpoint.pkl"


def _save_checkpoint(scoring_year: int, prev_weights: dict, n_train_rows: int) -> None:
    with open(_CKPT_PATH, "wb") as f:
        pickle.dump({
            "scoring_year": scoring_year,
            "prev_weights": prev_weights or {},
            "n_train_rows": n_train_rows,
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
    sel_buf:  list[pd.DataFrame],
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
    if sel_buf:
        pd.concat(sel_buf, ignore_index=True).to_csv(
            BACKTEST / "factor_selection.csv", mode=mode, header=header, index=False
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING ARRAY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_train_arrays(
    slices: dict[pd.Timestamp, pd.DataFrame],
    months: list[pd.Timestamp],
    char_cols: list[str],
    end_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack raw (X, y) for all training months up to end_year.
    NaN values are preserved in X so the pipeline imputer can process them.
    NaN counts track raw missingness for the column NaN filter.
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
        X_raw      = frame[char_cols].values.astype(np.float64)
        nan_sum   += np.isnan(X_raw).sum(axis=0)
        obs_count += X_raw.shape[0]
        X_list.append(X_raw)
        y_list.append(frame[Y_COL].values.astype(np.float64))

    if not X_list:
        return (
            np.empty((0, len(char_cols)), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            nan_sum, obs_count,
        )
    return np.vstack(X_list), np.concatenate(y_list), nan_sum, obs_count


# ══════════════════════════════════════════════════════════════════════════════
# 9. BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    start_year: int         = 1986,
    burn_in_years: int      = 5,
    tc_bps: float           = 10.0,
    sharpe_threshold: float = SHARPE_THRESHOLD,
    resume: bool            = False,
) -> pd.DataFrame:
    BACKTEST.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(BACKTEST)

    data_start_year = start_year - burn_in_years
    logger.info(
        f"Config: data_start={data_start_year}  start_year={start_year}  "
        f"burn_in={burn_in_years}yr  tc={tc_bps}bps  "
        f"sharpe_threshold={sharpe_threshold}  "
        f"cv_folds={CV_FOLDS}  n_alphas={N_ALPHAS}  "
        f"max_iter={MAX_ITER}  window=expanding"
    )

    slices, char_cols = load_data(logger)
    months            = sorted(slices.keys())

    last_data_year = max(m.year for m in months)
    scoring_years  = list(range(start_year, last_data_year + 1))

    logger.info(
        f"Scoring years: {scoring_years[0]} → {scoring_years[-1]}  "
        f"({len(scoring_years)} annual retrains)"
    )

    # ── Resume logic ──────────────────────────────────────────────────────────
    resume_from_year: int | None = None
    prev_weights: dict | None    = None
    first_flush = True

    if resume:
        ckpt = _load_checkpoint()
        if ckpt is None:
            logger.info("No checkpoint found — starting fresh.")
        else:
            resume_from_year = ckpt["scoring_year"] + 1
            prev_weights     = ckpt.get("prev_weights")
            first_flush      = False
            logger.info(
                f"Resuming from checkpoint: next scoring_year={resume_from_year}  "
                f"n_train_rows={ckpt['n_train_rows']:,}"
            )

    rows_buf: list[dict]         = []
    port_buf: list[pd.DataFrame] = []
    imp_buf:  list[pd.Series]    = []
    sel_buf:  list[pd.DataFrame] = []
    t_start       = datetime.now()
    scored_months = 0

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

        # ── Step 1: NaN filter (in-sample only) ───────────────────────────────
        nan_frac_tr  = np.where(obs_count > 0, nan_sum / obs_count, 1.0)
        nan_pass_idx = np.where(nan_frac_tr <= MAX_NAN_FRAC)[0]
        nan_cols     = [char_cols[j] for j in nan_pass_idx]

        # ── Step 2: Factor screening (in-sample Sharpe) ───────────────────────
        logger.info(
            f"[{scoring_year}]  Screening {len(nan_cols)} factors for "
            f"Sharpe >= {sharpe_threshold} …"
        )
        factor_sharpes = compute_factor_sharpes(
            slices, months, nan_cols, data_start_year, train_end_year
        )
        selected_mask = factor_sharpes >= sharpe_threshold
        current_cols  = factor_sharpes[selected_mask].index.tolist()
        n_selected    = len(current_cols)

        logger.info(
            f"[{scoring_year}]  {n_selected}/{len(nan_cols)} factors passed "
            f"Sharpe filter  (NaN-filtered pool: {len(nan_cols)})"
        )

        sel_df = factor_sharpes.reset_index()
        sel_df.columns = ["factor", "sharpe"]
        sel_df["selected"]     = selected_mask.values
        sel_df["scoring_year"] = scoring_year
        sel_buf.append(sel_df[["scoring_year", "factor", "sharpe", "selected"]])

        if n_selected < 2:
            logger.warning(
                f"[{scoring_year}]  Fewer than 2 factors selected — skipping year."
            )
            continue

        # ── Build X_train from selected columns only ───────────────────────────
        col_to_global = {c: i for i, c in enumerate(char_cols)}
        selected_idx  = np.array([col_to_global[c] for c in current_cols])
        X_train       = X_all[:, selected_idx]

        # ── Train model ────────────────────────────────────────────────────────
        try:
            pipe, feat_imp   = train_model(X_train, y_all, current_cols, logger)
            lasso            = pipe.named_steps["lasso"]
            n_nonzero        = int((feat_imp.values > 0).sum())
            n_train_rows     = len(y_all)
            logger.info(
                f"[{scoring_year}]  Model trained  "
                f"n_train={n_train_rows:,}  "
                f"n_features={n_selected}  "
                f"best_alpha={lasso.alpha_:.6f}  "
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

            if "prc" in slices[t].columns:
                prc       = slices[t]["prc"].reindex(signal_df.index)
                signal_df = signal_df[(prc.abs() >= 5.0) | prc.isna()]

            predicted, realized = generate_signals(pipe, signal_df, current_cols)
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

            row = {
                "eom":            t,
                "ic_pearson":     ic_p,
                "ic_spearman":    ic_s,
                "best_alpha":     lasso.alpha_,
                "n_nonzero":      n_nonzero,
                "n_train":        n_train_rows,
                "n_features":     n_selected,
                "scoring_year":   scoring_year,
                "train_end_year": train_end_year,
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
                f"IC={ic_s:+.3f}  "
                f"n_features={n_selected}  "
                f"n_universe={summary['n_universe']:,}"
            )

        # ── Checkpoint + flush ─────────────────────────────────────────────────
        if rows_buf:
            _flush(rows_buf, port_buf, imp_buf, sel_buf, first_flush)
            _save_checkpoint(scoring_year, prev_weights or {}, n_train_rows)
            elapsed = (datetime.now() - t_start).seconds // 60
            logger.info(
                f"  ✓ checkpoint  scoring_year={scoring_year}  "
                f"scored_months={scored_months}  elapsed={elapsed}m"
            )
            rows_buf, port_buf, imp_buf, sel_buf = [], [], [], []
            first_flush = False

    if rows_buf:
        _flush(rows_buf, port_buf, imp_buf, sel_buf, first_flush)

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    logger.info("Finalising outputs …")
    returns_df = pd.read_csv(BACKTEST / "monthly_returns.csv", parse_dates=["eom"])
    logger.info(f"  → {BACKTEST}/monthly_returns.csv  ({len(returns_df)} rows)")

    perf = evaluate_performance(returns_df, tc_bps)
    pd.Series(perf).to_frame("value").to_csv(BACKTEST / "performance.csv")
    logger.info(f"  → {BACKTEST}/performance.csv")

    sep = "=" * 62
    logger.info(sep)
    logger.info("  LASSO + FACTOR SCREENING — PERFORMANCE SUMMARY")
    logger.info(sep)
    logger.info(
        f"  Period         : {returns_df['eom'].min().date()} "
        f"→ {returns_df['eom'].max().date()}"
    )
    logger.info(f"  Months         : {perf['n_months']}")
    logger.info(f"  TC             : {tc_bps} bps one-way")
    logger.info(f"  Sharpe thresh  : {sharpe_threshold}")
    logger.info(f"  Burn-in        : {burn_in_years} years ({data_start_year}-{start_year-1})")
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
        description="LASSO expanding-window backtest with in-sample factor screening."
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
        "--sharpe-threshold", type=float, default=SHARPE_THRESHOLD,
        help=f"Minimum in-sample factor Sharpe to enter the model (default: {SHARPE_THRESHOLD}).",
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
        sharpe_threshold=args.sharpe_threshold,
        resume=args.resume,
    )