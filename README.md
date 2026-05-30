# Factor Investing with Machine Learning — MLII Project

A quantitative equity factor-investing research pipeline built on the Jensen-Kelly-Pedersen (JKP) 2023 *Journal of Finance* dataset ("Is There a Replication Crisis in Finance?"). The project trains cross-sectional predictive models on ~900 stock characteristics to form monthly long-short decile portfolios and rigorously backtests their performance against a market benchmark.

---

## Research Question

Can machine learning models (LASSO, XGBoost, LightGBM) trained on cross-sectional stock characteristics generate statistically robust long-short equity premia out-of-sample, and how do portfolio construction choices affect realized Sharpe ratios?

---

## Dataset

All raw data lives in `data/raw/` (not committed — see below) and was sourced from the JKP 2023 replication package.

| File | Description |
|---|---|
| `usa characteristics.csv` | ~900 cross-sectional characteristics for US equities (monthly, end-of-month) |
| `usa.csv` | US stock daily returns with market equity (`me`), `ret`, `ret_exc` |
| `market_returns.csv` | Monthly US/world market returns (value/equal-weighted) |
| `market_returns_daily.csv` | Same as above, daily frequency |
| `Documentation_JKP_2023.pdf` | Official dataset documentation |

Key column conventions:
- `id` — stock identifier (permno-style)
- `eom` — end-of-month date (`YYYYMMDD`)
- `me_lag1` — lagged market equity (millions USD); use instead of `me`
- `ret_exc_lead1m` — 1-month forward excess return (prediction target; never a feature)
- `size_grp` — size bucket: `mega`, `large`, `small`, `micro`

Processed parquet (`data/processed/data.parquet`): 2.26M rows × 438 columns, 1981–2025, dates already parsed to `datetime64`.

---

## Project Structure

```
.
├── code/                        # All model and analysis scripts
│   ├── data_cleaner.ipynb       # Raw CSV → processed parquet
│   ├── researcher.ipynb         # Exploratory factor analysis
│   ├── benchmark.py             # USA market benchmark (EW / VW)
│   ├── lasso_factor_screened.py # LASSO + in-sample factor pre-screening
│   ├── xg_boost_enhanced.py     # XGBoost (rank-norm, ensemble, vol-target)
│   ├── xg_boost_expanding.py    # XGBoost with expanding training window
│   ├── xg_boost_factor_screened.py  # XGBoost + factor pre-screening
│   ├── lgbm_factor_screened.py  # LightGBM + factor pre-screening
│   ├── ab_harness.py            # A/B test: 3 portfolio constructions on same model
│   ├── portfolio_fixed.py       # Corrected VW portfolio (no membership leak)
│   ├── portfolio_v2.py          # Rank-weighted VW portfolio + leg-balance
│   ├── recency_weighting.py     # Exponential sample weights for XGBoost training
│   ├── subperiod_eval.py        # Sub-period Sharpe / IC table
│   └── factor.py                # Standalone single-factor utilities
├── final_code/                  # Clean final versions + presentation notebooks
│   ├── ab_harness.py
│   ├── strategy_analysis.ipynb
│   ├── feature_analysis.ipynb
│   └── presentation.ipynb
├── data/
│   ├── raw/                     # Source CSVs (not tracked in git)
│   ├── processed/               # Cleaned parquet (not tracked in git)
│   └── backtest/                # Per-strategy output directories
├── 1155143226_MLII.pptx         # Project presentation slides
└── Explanation.docx             # Methodology write-up
```

---

## Models

### 1. LASSO with Factor Pre-Screening (`lasso_factor_screened.py`)
Before each annual retrain, every candidate feature is evaluated as a standalone long-short decile factor using only in-sample months. Features whose annualised Sharpe falls below a threshold (default 0.8) are dropped before LASSO training. No look-ahead bias: screening uses only strictly historical data.

### 2. XGBoost — Enhanced (`xg_boost_enhanced.py`)
Stacks four improvements over the baseline expanding-window XGBoost:
1. **Per-month cross-sectional rank normalization** of X and Y (maps features and target to [−0.5, 0.5] within each cross-section)
2. **IC-aligned objective**: rank-normalizing Y makes RMSE equivalent to maximizing rank IC
3. **Ensemble across random seeds**: multiple models averaged to reduce prediction variance
4. **Volatility targeting**: Q-spread scaled monthly to a target annualised vol

### 3. XGBoost + Factor Pre-Screening (`xg_boost_factor_screened.py`)
Combines the enhanced XGBoost pipeline with the same in-sample factor Sharpe screen used by the LASSO variant.

### 4. LightGBM + Factor Pre-Screening (`lgbm_factor_screened.py`)
Drop-in LightGBM equivalent of the XGBoost factor-screened pipeline, useful for speed and as a cross-check.

### 5. A/B Harness (`ab_harness.py`)
Runs three portfolio construction methods on identical model predictions in a single backtest pass — a controlled comparison:
- **Config A**: Equal-weight decile (original baseline, leak intact for faithful reproduction)
- **Config B**: Rank-weighted value-weight (`portfolio_v2`)
- **Config C**: Rank-weighted VW + leg-balance (`portfolio_v2`, both upgrades)

Optionally adds **exponential recency weighting** (`--half-life`) to up-weight recent training months.

---

## Portfolio Construction

`portfolio_fixed.py` corrects four issues present in a naive equal-weight baseline:

| Issue | Fix |
|---|---|
| Membership leak: dropping names with extreme realized returns before forming legs | Winsorize (clip) returns instead of dropping — every selected name counts |
| Equal-weight microcap overstatement | Capped value-weighting (JKP-style), winsorized at `cap_pctl` |
| No microcap screen | Drop bottom `micro_pctl` of market equity before ranking |
| Turnover cost on name overlap rather than weight change | `cost_t = tc × Σ|w_{i,t} − w_{i,t−1}|` (true L1 weight-change cost) |

---

## Backtest Design

- **Timing**: train on `(X[t-1], y[t-1])` where `y = ret_exc_lead1m`; score on `X[t]`; realized return is `ret_exc_lead1m[t]` (next-month return)
- **Features**: all numeric columns except `_META` identifiers and `ret_exc_lead1m`; cross-sectional mean imputation → standardization
- **Expanding window**: models retrain annually, always using all available history up to that point
- **Universe**: non-micro US equities with sufficient data coverage

---

## Installation

```bash
pip install pandas numpy scipy matplotlib scikit-learn pyarrow xgboost lightgbm
```

---

## Usage

**Step 1 — Clean the raw data:**
```bash
jupyter nbconvert --to notebook --execute code/data_cleaner.ipynb
```

**Step 2 — Run the market benchmark:**
```bash
python code/benchmark.py --start 1986
```

**Step 3 — Run a model backtest:**
```bash
# LASSO with factor screening
python code/lasso_factor_screened.py --start 1986 --tc 10 --sharpe-threshold 0.8

# XGBoost enhanced
python code/xg_boost_enhanced.py --start 1986 --tc 10

# LightGBM with factor screening
python code/lgbm_factor_screened.py --start 1986 --tc 10

# A/B portfolio comparison (XGBoost, recency-weighted)
python final_code/ab_harness.py --start 1986 --tc 10 --half-life 120

# Disable recency weighting
python final_code/ab_harness.py --half-life inf
```

All scripts support `--resume` to restart from the last checkpoint.

---

## Outputs

Each strategy writes results to `data/backtest/<strategy-name>/`:

| File | Description |
|---|---|
| `monthly_returns.csv` | Per-month long / short / Q-spread returns, IC, turnover |
| `portfolios.csv` | Stock-level leg assignments for every period |
| `coefs.csv` | Coefficient / feature-importance matrix (months × characteristics) |
| `performance.csv` | Aggregate Sharpe, max drawdown, IC, ICIR |
| `cumulative_returns.png` | Three-panel performance chart |

---

## Key Design Choices & Lessons

- **No look-ahead bias**: factor screening, imputation, and scaling all use only strictly in-sample data relative to the scoring date.
- **Recency weighting** (half-life ~120 months) can help when IC decays over time (factors being arbitraged away), but is a hyperparameter to validate — short half-lives overfit recent noise.
- **Equal-weight vs. value-weight**: equal-weight portfolios overstate performance by overloading microcaps. Capped VW with a microcap screen gives a more realistic picture.
- **Membership leak** in return winsorization is a common but subtle source of backtest inflation — the corrected pipeline clips rather than drops.

---

## Reference

Jensen, T. I., Kelly, B. T., & Pedersen, L. H. (2023). Is There a Replication Crisis in Finance? *Journal of Finance*, 78(5), 2465–2518.