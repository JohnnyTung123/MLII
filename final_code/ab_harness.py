"""
ab_harness.py — run THREE portfolio constructions in one backtest pass.

This answers "run the enhanced model with the OLD equal-weight portfolio too"
by scoring all three configs on the SAME predictions every month, so the
comparison is perfectly controlled (identical model, seeds, universe, timing):

  A. decile equal-weight  — the ORIGINAL leaky baseline (RET_CAP drop included,
                            so it faithfully reproduces the 1.31-Sharpe regime)
  B. rank-weight VW       — portfolio_v2, rank-weighting only
  C. rank-weight VW + leg-balance — portfolio_v2, both upgrades

It imports the enhanced model machinery from xg_boost_enhanced.py unchanged
(rank-normalization, ensemble, IC objective) and adds OPTIONAL recency-weighted
training. One run → three monthly_returns CSVs + a sub-period comparison table.

Place this file next to:  xg_boost_enhanced.py, portfolio_v2.py,
recency_weighting.py, subperiod_eval.py   (all in the same directory).

Run:
    python ab_harness.py --start 1986 --tc 10 --half-life 120
    python ab_harness.py --half-life inf        # disable recency weighting
"""

from __future__ import annotations

import argparse
import logging
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the enhanced model pipeline verbatim
import xg_boost_enhanced as E
from portfolio_v2 import construct_portfolio_v2
from recency_weighting import exponential_sample_weights, build_eom_per_row
from subperiod_eval import subperiod_table


# ── config A: the original equal-weight portfolio (leaky baseline, faithful) ──
def construct_portfolio_ew(predicted, realized, prev_long_ids, prev_short_ids,
                           tc_bps, ret_cap=1.0, decile=0.10):
    """Faithful copy of the ORIGINAL construct_portfolio (equal-weight, RET_CAP
    drop leak intact) so config A reproduces the pre-fix behaviour exactly."""
    tc = tc_bps / 10_000
    realized = realized[realized.abs() <= ret_cap]          # the drop-leak, kept on purpose
    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index)
    n = len(pred_c); nd = max(1, int(n * decile))
    if n < 2 * nd:
        return None, None
    ranked = pred_c.sort_values()
    short_ids = set(ranked.index[:nd]); long_ids = set(ranked.index[-nd:])
    long_ret = real_c.reindex(list(long_ids)).mean()
    short_ret = real_c.reindex(list(short_ids)).mean()
    qspread = long_ret - short_ret
    if prev_long_ids and prev_short_ids:
        to = ((1 - len(long_ids & prev_long_ids) / len(long_ids)) +
              (1 - len(short_ids & prev_short_ids) / len(short_ids))) / 2
    else:
        to = 1.0
    net = qspread - 2 * tc * to
    summary = {"long_ret": long_ret, "short_ret": short_ret, "qspread": qspread,
               "net_qspread": net, "turnover": to, "n_long": len(long_ids),
               "n_short": len(short_ids), "n_universe": n}
    return long_ids, short_ids, summary


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt_path(out: Path) -> Path:
    return out / "checkpoint.pkl"


def _save_checkpoint(out, scoring_year, prevA_long, prevA_short, prevB, prevC, recL, recS):
    with open(_ckpt_path(out), "wb") as f:
        pickle.dump({
            "scoring_year": scoring_year,
            "prevA_long":   prevA_long,
            "prevA_short":  prevA_short,
            "prevB":        prevB or {},
            "prevC":        prevC or {},
            "recL":         list(recL),
            "recS":         list(recS),
        }, f)


def _load_checkpoint(out) -> dict | None:
    p = _ckpt_path(out)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _flush_rows(rows: list[dict], cfg: str, out: Path, first_flush: bool) -> None:
    if not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        out / f"monthly_returns_{cfg}.csv",
        mode="w" if first_flush else "a",
        header=first_flush,
        index=False,
    )


def run_ab(start_year=1986, burn_in_years=5, tc_bps=10.0, half_life=120.0, fresh=False,
           run_name=None):
    hl_tag = f"hl{int(half_life)}" if np.isfinite(half_life) else "hlinf"
    out = E.BACKTEST.parent / (run_name or f"xgboost_ab_{hl_tag}")
    out.mkdir(parents=True, exist_ok=True)
    logger = E.setup_logger(out)
    logger.info(f"A/B/C harness — recency half-life = {half_life} months")

    slices, char_cols = E.load_data(logger)
    months = sorted(slices.keys())
    data_start_year = start_year - burn_in_years
    last_year = max(m.year for m in months)
    scoring_years = list(range(start_year, last_year + 1))

    # per-config rolling state
    cfgs = ["A_decile_ew", "B_rankw_vw", "C_rankw_vw_balanced"]
    rows = {c: [] for c in cfgs}
    first_flush = {c: True for c in cfgs}
    prevA_long = prevA_short = None
    prevB = prevC = None
    recL = deque(maxlen=E.VOL_LOOKBACK)
    recS = deque(maxlen=E.VOL_LOOKBACK)
    resume_from_year: int | None = None

    ckpt = None if fresh else _load_checkpoint(out)
    if fresh:
        logger.info("--fresh flag set — ignoring any existing checkpoint.")
    elif ckpt is None:
        logger.info("No checkpoint found — starting fresh.")
    else:
        resume_from_year = ckpt["scoring_year"] + 1
        prevA_long  = ckpt["prevA_long"]
        prevA_short = ckpt["prevA_short"]
        prevB       = ckpt["prevB"]
        prevC       = ckpt["prevC"]
        recL.extend(ckpt["recL"])
        recS.extend(ckpt["recS"])
        first_flush = {c: False for c in cfgs}
        logger.info(f"Resuming from checkpoint: next scoring_year={resume_from_year}")

    # expanding cache (reuse enhanced builder)
    X_cache, y_cache, ns_cache, oc_cache = E._build_train_arrays_range(
        slices, months, char_cols, data_start_year, start_year - 1)
    cache_end = start_year - 1

    for sy in scoring_years:
        if resume_from_year is not None and sy < resume_from_year:
            continue
        tey = sy - 1
        if tey < data_start_year:
            continue
        if tey > cache_end:
            Xn, yn, nsn, ocn = E._build_train_arrays_range(
                slices, months, char_cols, cache_end + 1, tey)
            if len(yn):
                X_cache = np.vstack([X_cache, Xn]); y_cache = np.concatenate([y_cache, yn])
                ns_cache += nsn; oc_cache += ocn
            cache_end = tey

        nan_frac = np.where(oc_cache > 0, ns_cache / oc_cache, 1.0)
        idx = np.where(nan_frac <= E.MAX_NAN_FRAC)[0]
        cur_cols = [char_cols[j] for j in idx]
        X_train = X_cache[:, idx]

        # recency weights aligned to the cached rows
        if np.isfinite(half_life):
            eom_rows = build_eom_per_row(slices, months, char_cols,
                                         data_start_year, tey,
                                         y_col=E.Y_COL, ret_cap=E.RET_CAP)
            sw = exponential_sample_weights(eom_rows, half_life)
            if len(sw) != len(y_cache):           # safety: lengths must match
                logger.warning(f"[{sy}] sample_weight len {len(sw)} != y {len(y_cache)}; "
                               f"disabling recency weighting this year.")
                sw = None
        else:
            sw = None

        models = _train_ensemble_weighted(X_train, y_cache, cur_cols, sw, logger)
        best_iters = [m.best_iteration for m in models]
        logger.info(f"[{sy}] trained {len(models)} seeds  n={len(y_cache):,}  "
                    f"feats={len(cur_cols)}  recency={'on' if sw is not None else 'off'}")

        for t in [m for m in months if m.year == sy and m in slices]:
            sdf = slices[t][cur_cols + [E.Y_COL]]
            if "prc" in slices[t].columns:
                prc = slices[t]["prc"].reindex(sdf.index)
                sdf = sdf[(prc.abs() >= 5.0) | prc.isna()]
            predicted, realized = E.generate_signals(models, sdf, cur_cols)
            if predicted is None:
                continue
            ic_p, ic_s = E.compute_ic(predicted, realized)
            me_t = slices[t]["me_company"].reindex(sdf.index)

            # ── A: equal-weight decile (leaky baseline) ──
            sA = None
            ra = construct_portfolio_ew(predicted, realized, prevA_long, prevA_short,
                                        tc_bps, E.RET_CAP, E.DECILE)
            if ra[0] is not None:
                prevA_long, prevA_short, sA = ra
                rows["A_decile_ew"].append({"eom": t, "ic_spearman": ic_s, **sA})

            # ── B: rank-weight VW, no balancing ──
            sB = None
            pB, sB = construct_portfolio_v2(predicted, realized, me_t, prevB, tc_bps,
                    decile=E.DECILE, micro_pctl=0.20, ret_cap=E.RET_CAP, cap_pctl=0.80,
                    use_rank_weight=True, value_tilt=True, balance_legs=False)
            if pB is not None:
                prevB = dict(zip(pB["id"], pB["weight"]))
                rows["B_rankw_vw"].append({"eom": t, "ic_spearman": ic_s, **sB})

            # ── C: rank-weight VW + inverse-vol leg balancing ──
            sC = None
            lv = float(np.std(recL)) * np.sqrt(12) if len(recL) >= 12 else None
            sv = float(np.std(recS)) * np.sqrt(12) if len(recS) >= 12 else None
            pC, sC = construct_portfolio_v2(predicted, realized, me_t, prevC, tc_bps,
                    decile=E.DECILE, micro_pctl=0.20, ret_cap=E.RET_CAP, cap_pctl=0.80,
                    use_rank_weight=True, value_tilt=True,
                    balance_legs=True, leg_vol_long=lv, leg_vol_short=sv)
            if pC is not None:
                prevC = dict(zip(pC["id"], pC["weight"]))
                recL.append(sC["long_ret"]); recS.append(-sC["short_ret"])
                rows["C_rankw_vw_balanced"].append({"eom": t, "ic_spearman": ic_s, **sC})

            if sA is not None and sB is not None and sC is not None:
                logger.info(f"  {t.date()}  IC={ic_s:+.3f}  "
                            f"A={sA['qspread']:+.4f} B={sB['qspread']:+.4f} C={sC['qspread']:+.4f}")
            else:
                skipped = [n for n, s in (("A", sA), ("B", sB), ("C", sC)) if s is None]
                logger.warning(f"  {t.date()}  IC={ic_s:+.3f}  skipped configs: {skipped}")

        # ── flush + checkpoint after each scoring year ─────────────────────────
        for c in cfgs:
            _flush_rows(rows[c], c, out, first_flush[c])
            first_flush[c] = False
            rows[c] = []
        _save_checkpoint(out, sy, prevA_long, prevA_short, prevB, prevC, recL, recS)
        logger.info(f"  ✓ checkpoint saved — scoring_year={sy}")

    # ── final summary + comparison ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("A/B/C COMPARISON — net Sharpe by sub-period (recency half-life "
          f"= {half_life}m)")
    print("=" * 78)
    summary_lines = []
    for c in cfgs:
        csv_path = out / f"monthly_returns_{c}.csv"
        if not csv_path.exists():
            logger.warning(f"No output file for {c} — skipping summary.")
            continue
        df = pd.read_csv(csv_path, parse_dates=["eom"])
        tbl = subperiod_table(df, col="net_qspread")
        print(f"\n### {c} (net) ###")
        print(tbl.to_string())
        full = tbl.loc["FULL"]; last120 = tbl.loc["last_120m"]
        summary_lines.append((c, full["sharpe"], last120["sharpe"], full["ann_ret"]))

    print("\n" + "=" * 78)
    print("HEADLINE: net Sharpe — full sample vs last 120 months")
    print("=" * 78)
    print(f"{'config':<26}{'full_SR':>10}{'last10y_SR':>12}{'full_annret':>14}")
    for name, fsr, lsr, far in summary_lines:
        print(f"{name:<26}{fsr:>10.3f}{lsr:>12.3f}{far:>14.2%}")
    print("\nKeep the config with the best LAST-10-YEAR Sharpe, not the best full "
          "sample. If A wins out-of-sample, the upgrades didn't help — keep it simple.")
    logger.info(f"Outputs written to {out}")

    ckpt = _ckpt_path(out)
    if ckpt.exists():
        ckpt.unlink()
        logger.info("checkpoint.pkl removed (run complete)")


def _train_ensemble_weighted(X_train, y_train, cur_cols, sample_weight, logger):
    """train_ensemble, but threads an optional sample_weight into .fit()."""
    from xgboost import XGBRegressor
    n = len(y_train)
    split = min(max(E.MIN_STOCKS, int(n * (1 - E.VAL_FRAC))), n - 1)
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]
    sw_tr = sample_weight[:split] if sample_weight is not None else None
    models = []
    for seed in E.ENSEMBLE_SEEDS:
        m = XGBRegressor(**{**E.XGB_PARAMS, "random_state": seed},
                         early_stopping_rounds=E.EARLY_STOPPING_ROUNDS, eval_metric="rmse")
        m.fit(X_tr, y_tr, sample_weight=sw_tr,
              eval_set=[(X_val, y_val)], verbose=False)
        models.append(m)
    return models


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A/B/C portfolio-construction harness.")
    p.add_argument("--start", type=int, default=1986)
    p.add_argument("--burn-in", type=int, default=5)
    p.add_argument("--tc", type=float, default=10.0)
    p.add_argument("--half-life", default="120",
                   help="recency half-life in months, or 'inf' to disable")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint and restart from scratch.")
    p.add_argument("--run-name", default=None,
                   help="Output folder name under data/backtest/ (default: xgboost_ab_hl<N>).")
    a = p.parse_args()
    hl = float("inf") if str(a.half_life).lower() in ("inf", "none") else float(a.half_life)
    run_ab(start_year=a.start, burn_in_years=a.burn_in, tc_bps=a.tc, half_life=hl,
           fresh=a.fresh, run_name=a.run_name)
