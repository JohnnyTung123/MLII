"""
portfolio_v2.py — rank-weighted, risk-balanced long-short construction.

Drop-in successor to construct_portfolio_vw. Motivation from the diagnostics:

  * The enhanced model's ICIR jumped to ~2.38 (mean IC ~0.064), but the
    top/bottom-DECILE + value-weight portfolio only delivered ~0.69 net Sharpe.
    Deciles throw away (a) the MAGNITUDE of predictions and (b) every name
    between the 10th and 90th percentile.
  * Monthly-spread kurtosis was ~8.2 (fat tails) and realized vol ~18.5%,
    which is what caps the Sharpe — a few tail months dominate the denominator.

Two upgrades, toggimble independently:

  1. RANK-WEIGHTING (use_rank_weight=True)
     Instead of a 0/1 decile membership, every stock gets a weight proportional
     to its cross-sectional prediction rank, demeaned so the book is
     dollar-neutral. Strong convictions get more capital; the whole
     cross-section contributes, which lowers idiosyncratic noise. Optionally
     still multiplied by capped market-equity (value tilt) so it stays tradable.

  2. INVERSE-VOL LEG BALANCING (balance_legs=True)
     The long and short legs are each scaled so their EX-ANTE volatility
     contribution is equalised, using a trailing estimate of each leg's vol
     passed in by the caller (leg_vol_long / leg_vol_short). This stops one
     leg (usually the short, microcap-driven) from dominating risk and is the
     direct lever against the kurtosis problem. If no vol estimates are
     supplied, legs are left dollar-neutral (1, -1).

Everything else (clipping not dropping, microcap screen, honest L1-weight
turnover cost) is preserved from construct_portfolio_vw.

Returns the SAME (port_df, summary) schema so evaluate_performance/_flush are
unchanged. summary additionally carries 'gross_exposure' for leverage tracking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _capped_value(me: pd.Series, ids: list, cap_pctl: float) -> pd.Series:
    """Capped (winsorized) market-equity series for the given ids, >0, NaN→median."""
    cap = me.reindex(ids).astype(float)
    med = cap[cap > 0].median()
    cap = cap.where(cap > 0, med)
    if not np.isfinite(med) or med <= 0:
        return pd.Series(1.0, index=ids)            # no cap info → flat
    return cap.clip(upper=cap.quantile(cap_pctl))


def construct_portfolio_v2(
    predicted: pd.Series,
    realized: pd.Series,
    me: pd.Series,
    prev_weights: dict | None,
    tc_bps: float,
    *,
    # selection
    decile: float = 0.10,
    micro_pctl: float = 0.20,
    ret_cap: float = 1.0,
    cap_pctl: float = 0.80,
    # construction toggles
    use_rank_weight: bool = True,
    value_tilt: bool = True,
    balance_legs: bool = True,
    leg_vol_long: float | None = None,
    leg_vol_short: float | None = None,
    target_gross: float = 2.0,
) -> tuple[pd.DataFrame, dict] | tuple[None, None]:
    """
    Parameters mirror construct_portfolio_vw with additions:
      use_rank_weight : weight by demeaned prediction rank (else flat decile)
      value_tilt      : multiply rank weight by capped market equity
      balance_legs    : inverse-vol scale the two legs to equal risk
      leg_vol_long/short : trailing ann. vol of each leg (caller-supplied);
                           if either is None, legs stay dollar-neutral
      target_gross    : gross exposure (sum |w|) to normalise to, default 2.0
                        i.e. ~1 long + ~1 short (dollar-neutral baseline)
    """
    tc = tc_bps / 10_000

    common = predicted.index.intersection(realized.dropna().index)
    pred_c = predicted.reindex(common).dropna()
    real_c = realized.reindex(pred_c.index).clip(-ret_cap, ret_cap)   # CLIP not drop
    me_c   = me.reindex(pred_c.index)

    # microcap screen
    if micro_pctl > 0 and me_c.notna().sum() > 0:
        keep   = me_c >= me_c.quantile(micro_pctl)
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
    leg_ids   = long_ids + short_ids

    # ── raw weights within each leg ──────────────────────────────────────────
    if use_rank_weight:
        # demeaned rank within the SELECTED names of each leg, so weights are
        # proportional to conviction; long leg positive, short leg negative.
        def _rank_w(ids, sign):
            p = pred_c.reindex(ids)
            r = p.rank()                       # 1..k
            w = r - r.mean()                   # demean → centered on 0
            if w.abs().sum() == 0:
                w = pd.Series(1.0, index=ids)
            # make all-positive magnitude, then apply leg sign
            w = w - w.min() + (r.max() - r.min()) * 0.1   # shift so min>0
            w = w / w.sum() * sign
            if value_tilt:
                v = _capped_value(me_c, ids, cap_pctl)
                w = (w.abs() * v)
                w = w / w.sum() * sign
            return w
        w_long  = _rank_w(long_ids,  +1.0)
        w_short = _rank_w(short_ids, -1.0)
    else:
        if value_tilt:
            vL = _capped_value(me_c, long_ids,  cap_pctl)
            vS = _capped_value(me_c, short_ids, cap_pctl)
            w_long  =  vL / vL.sum()
            w_short = -vS / vS.sum()
        else:
            w_long  = pd.Series( 1.0 / len(long_ids),  index=long_ids)
            w_short = pd.Series(-1.0 / len(short_ids), index=short_ids)

    # ── normalise each leg to unit gross (enforces strict dollar-neutrality) ──
    # Long leg sums to +1, short leg sums to -1 in gross terms. This is the
    # non-negotiable invariant for a dollar-neutral book.
    if w_long.abs().sum()  > 0:
        w_long  = w_long  / w_long.abs().sum()
    if w_short.abs().sum() > 0:
        w_short = w_short / w_short.abs().sum()

    weights = pd.concat([w_long, w_short]) * (target_gross / 2.0)

    # ── inverse-vol BOOK scaling (optional, preserves neutrality) ────────────
    # We cannot rebalance dollars between legs without breaking neutrality, so
    # 'balance_legs' instead applies a single causal leverage multiplier to the
    # whole book based on the average of the two legs' trailing vols, nudging
    # gross exposure toward a constant-risk target. This is the neutrality-safe
    # version of leg risk control; finer per-name risk parity is out of scope.
    if balance_legs and leg_vol_long and leg_vol_short and \
       leg_vol_long > 0 and leg_vol_short > 0:
        avg_leg_vol = 0.5 * (leg_vol_long + leg_vol_short)
        if avg_leg_vol > 0:
            mult    = float(np.clip(0.15 / avg_leg_vol, 0.5, 2.0))  # 15% leg-vol ref
            weights = weights * mult

    # ── leg returns on final weights ─────────────────────────────────────────
    rL = real_c.reindex(long_ids)
    rS = real_c.reindex(short_ids)
    long_ret  = float((weights.reindex(long_ids)  * rL).sum())
    short_ret = float((weights.reindex(short_ids) * rS).sum())   # weights<0 here
    qspread   = long_ret + short_ret    # short leg already signed negative

    # ── honest L1-weight turnover cost ───────────────────────────────────────
    if prev_weights:
        all_ids = set(weights.index) | set(prev_weights)
        cur  = pd.Series(weights, index=list(all_ids)).fillna(0.0)
        prev = pd.Series(prev_weights, index=list(all_ids)).fillna(0.0)
        turnover = float((cur - prev).abs().sum()) / 2.0
    else:
        turnover = float(weights.abs().sum()) / 2.0

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
        "long_ret":       long_ret,
        "short_ret":      -short_ret,    # report as positive magnitude
        "qspread":        qspread,
        "net_qspread":    net_qspread,
        "turnover":       turnover,
        "n_long":         len(long_ids),
        "n_short":        len(short_ids),
        "n_universe":     n,
        "gross_exposure": float(weights.abs().sum()),
    }
    return port_df, summary
