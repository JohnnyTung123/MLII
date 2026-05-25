"""
portfolio_fixed.py — corrected portfolio construction for xg_boost_expanding.py

WHAT CHANGED vs. the original construct_portfolio (and WHY):

  1. NO MEMBERSHIP LEAK.
     Original: `realized = realized[realized.abs() <= RET_CAP]` was applied
     BEFORE forming legs, so any name whose realized next-month return blew
     past +/-100% was deleted from the universe entirely. That conditions
     portfolio membership on the outcome you're measuring — a forward-looking
     filter that disproportionately removes short-leg losers and inflates win
     rate + smooths the curve. FIX: clip (winsorize) realized returns to
     [-RET_CAP, +RET_CAP] so every selected name still counts; nothing is
     dropped from membership.

  2. CAPPED VALUE-WEIGHTING (JKP-style) instead of equal weight.
     Stocks are weighted by market equity (me_company), winsorized at the
     `cap_pctl` cross-sectional percentile so no single mega-cap dominates a
     leg. This mirrors JKP's "capped value weight" construction and removes
     the equal-weight microcap overstatement.

  3. MICROCAP SCREEN.
     You have no NYSE-breakpoint column, so we approximate JKP's "non-micro"
     universe by dropping the bottom `micro_pctl` of the cross-section by
     me_company each month BEFORE ranking. Set micro_pctl=0.0 to disable.

  4. HONEST TURNOVER COSTS computed on ACTUAL WEIGHTS, not name overlap.
     cost_t = tc * sum_i |w_{i,t} - w_{i,t-1}|  (one-way tc on each side of
     the rebalance). This is the real L1 weight-change cost, which is what a
     value-weighted book actually pays.

Drop-in usage in xg_boost_expanding.py:

    from portfolio_fixed import construct_portfolio_vw

    # in the scoring loop, replace the construct_portfolio(...) call with:
    me_t = slices[t]["me_company"].reindex(signal_df.index)
    port_df, summary = construct_portfolio_vw(
        predicted, realized, me_t,
        prev_weights, tc_bps,
        cap_pctl=0.80, micro_pctl=0.20, ret_cap=RET_CAP, decile=DECILE,
    )
    prev_weights = dict(zip(port_df["id"], port_df["weight"]))

`prev_weights` replaces prev_long_ids/prev_short_ids — initialise to None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _capped_vw_weights(
    ids: list,
    me: pd.Series,
    cap_pctl: float,
    sign: float,
) -> pd.Series:
    """
    Capped value weights for one leg.

    me is winsorized at the `cap_pctl` percentile *of this leg* before
    normalising, so a single huge name can't dominate. Names with missing or
    non-positive me fall back to the median cap of the leg (so they still get
    a sensible, non-zero weight rather than being silently dropped).
    Returns weights that sum to `sign` (+1 for long, -1 for short).
    """
    cap = me.reindex(ids).astype(float)
    med = cap[cap > 0].median()
    cap = cap.where(cap > 0, med)            # fix non-positive / NaN
    if not np.isfinite(med) or med <= 0:     # whole leg has no cap info
        w = pd.Series(sign / len(ids), index=ids)
        return w
    ceiling = cap.quantile(cap_pctl)
    cap = cap.clip(upper=ceiling)            # winsorize the cap, JKP-style
    w = cap / cap.sum() * sign
    return w


def construct_portfolio_vw(
    predicted: pd.Series,
    realized: pd.Series,
    me: pd.Series,
    prev_weights: dict | None,
    tc_bps: float,
    cap_pctl: float = 0.80,
    micro_pctl: float = 0.20,
    ret_cap: float = 1.0,
    decile: float = 0.10,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Capped value-weighted long-short decile portfolio with no membership leak.

    Parameters
    ----------
    predicted   : model scores, indexed by id
    realized     : ret_exc_lead1m, indexed by id  (return earned next month)
    me           : me_company, indexed by id  (market equity, millions USD)
    prev_weights : {id: weight} from previous month, or None
    tc_bps       : one-way transaction cost in basis points
    cap_pctl     : winsorize cap weights at this cross-sectional percentile
    micro_pctl   : drop bottom this fraction of the universe by me each month
    ret_cap      : winsorize realized returns to [-ret_cap, +ret_cap]
    decile       : top/bottom fraction for the legs
    """
    tc = tc_bps / 10_000

    # ── align predicted / realized / me on common ids ────────────────────────
    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index)
    me_c   = me.reindex(pred_c.index)

    # CLIP realized returns — NO drop, so membership never depends on outcome
    real_c = real_c.clip(lower=-ret_cap, upper=ret_cap)

    # ── microcap screen (approximate JKP non-micro universe) ─────────────────
    if micro_pctl > 0 and me_c.notna().sum() > 0:
        thresh = me_c.quantile(micro_pctl)
        keep   = me_c >= thresh
        pred_c = pred_c[keep]
        real_c = real_c.reindex(pred_c.index)
        me_c   = me_c.reindex(pred_c.index)

    n        = len(pred_c)
    n_decile = max(1, int(n * decile))
    if n < 2 * n_decile:
        return None, None

    ranked    = pred_c.sort_values()
    short_ids = list(ranked.index[:n_decile])
    long_ids  = list(ranked.index[-n_decile:])

    # ── capped value weights per leg ─────────────────────────────────────────
    w_long  = _capped_vw_weights(long_ids,  me_c, cap_pctl, sign=+1.0)
    w_short = _capped_vw_weights(short_ids, me_c, cap_pctl, sign=-1.0)
    weights = pd.concat([w_long, w_short])

    # ── leg returns are now WEIGHTED, not simple means ───────────────────────
    long_ret  = float((w_long  * real_c.reindex(long_ids)).sum())
    short_ret = float((-w_short * real_c.reindex(short_ids)).sum())  # w_short<0
    qspread   = long_ret - short_ret

    # ── honest turnover: L1 change in actual weights ─────────────────────────
    if prev_weights:
        all_ids = set(weights.index) | set(prev_weights)
        cur  = pd.Series(weights, index=list(all_ids)).fillna(0.0)
        prev = pd.Series(prev_weights, index=list(all_ids)).fillna(0.0)
        turnover = float((cur - prev).abs().sum()) / 2.0   # /2 → one-way
    else:
        turnover = 1.0   # initial month: assume full deployment

    # cost = tc applied to total one-way weight traded (both legs)
    cost        = tc * (2.0 * turnover)
    net_qspread = qspread - cost

    port_df = pd.DataFrame({
        "id":            list(weights.index),
        "leg":           ["long"] * len(long_ids) + ["short"] * len(short_ids),
        "weight":        weights.values,
        "predicted_ret": pred_c.reindex(weights.index).values,
        "realized_ret":  real_c.reindex(weights.index).values,
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
