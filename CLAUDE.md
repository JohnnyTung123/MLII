# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative finance factor-investing research project (MLII course) built on the Jensen-Kelly-Pedersen (JKP) 2023 *Journal of Finance* dataset ("Is There a Replication Crisis in Finance?"). The pipeline is: raw data → cleaning/processing → factor research → backtesting.

## Data

All raw data lives in `data/raw/` and should never be modified in place. Processed outputs go to `data/processed/`; backtest results go to `data/backtest/`.

| File | Description |
|---|---|
| `usa characteristics.csv` | ~900 cross-sectional stock characteristics for US equities (monthly, end-of-month) |
| `usa.csv` | US stock daily returns with market equity (`me`), `ret`, `ret_exc` |
| `market_returns.csv` | Monthly market returns by country (value/equal-weighted, local/USD) |
| `market_returns_daily.csv` | Same as above but daily frequency |
| `world_ret_monthly.csv` | World-level monthly returns |
| `data/Documentation_JKP_2023.pdf` | Official dataset documentation — read this before using characteristics |

Key column conventions (from JKP):
- `id` — stock identifier (permno-style)
- `excntry` — exchange country (e.g. `USA`)
- `eom` / `date` — end-of-month date in `YYYYMMDD` format
- `me` — market equity in millions USD
- `ret_exc` — excess return (return minus risk-free rate)
- `size_grp` — size bucket (`mega`, `large`, `small`, `micro`)

## Code Structure

`code/data_cleaner.ipynb` — Jupyter notebook that loads and cleans the JKP raw CSVs. Writes the processed parquet to `data/processed/data.parquet` (2.26M rows × 438 cols, dates already parsed to `datetime64`).

`code/lasso.py` — Cross-sectional LASSO backtest. Reads `data/processed/data.parquet`, runs monthly LASSO regressions, forms a long-short decile portfolio, and writes results to `data/backtest/lasso/`.

## Running the Code

Dependencies:

```bash
pip install pandas numpy scipy matplotlib scikit-learn pyarrow
```

Run the LASSO backtest:

```bash
python code/lasso.py                      # default: start=1990, tc=10 bps
python code/lasso.py --start 1995 --tc 5  # custom start year and transaction cost
```

Outputs written to `data/backtest/lasso/`:
- `monthly_returns.csv` — per-month long/short/qspread returns, IC, alpha, turnover
- `portfolios.csv` — stock-level leg assignments for every period
- `coefs.csv` — LASSO coefficient matrix (months × characteristics)
- `performance.csv` — aggregate Sharpe, drawdown, IC, ICIR metrics
- `cumulative_returns.png` — three-panel performance chart

## Key Data Facts (processed parquet)

- `data/processed/data.parquet`: 2.26M rows, 438 columns, 1981-01 to 2025-12 (540 months)
- `eom` and `date` are already `datetime64[us]` — no re-parsing needed
- **`ret_exc_lead1m`** is the 1-month forward excess return — the prediction target (`Y_COL`); never use as a predictor feature
- **`ret_exc` is not present** in the parquet; use `ret` (total return) as a momentum signal feature
- **`me` is not present**; use `me_lag1` for size-related weighting
- ~219 characteristic columns survive a 20% global NaN filter; ~350+ survive a 30% filter

## Backtest Design Notes

- **Timing**: train on (X[t-1], y[t-1]) where y = `ret_exc_lead1m` (= return at t); signal from X[t]; realized return is `ret_exc_lead1m[t]` (= return at t+1)
- **Features**: all numeric columns excluding `_META` identifiers and `ret_exc_lead1m`; missing values imputed cross-sectionally with mean before standardisation
- **Model**: `LassoCV` with 5-fold CV and 50-point alpha grid per month
- `construct_portfolio()` returns both gross and TC-adjusted net Q-spread; turnover is fraction of portfolio replaced vs. previous period
- The JKP documentation PDF (`data/Documentation_JKP_2023.pdf`) defines all ~900 characteristic columns.