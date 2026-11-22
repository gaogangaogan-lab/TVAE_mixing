"""Fidelity of the 100%-mixing training set (Section 3.5).

KSComplement for continuous predictors, TVComplement for discrete predictors,
combined score F = unweighted mean over predictors only; the target is reported
separately as a construction check.
"""
import warnings

import numpy as np
import pandas as pd

_FALLBACK_WARNED = False


def _sdm():
    try:
        from sdmetrics.single_column import KSComplement, TVComplement
        return KSComplement, TVComplement
    except ImportError:
        return None, None


def ks_complement(real, syn):
    KS, _ = _sdm()
    if KS is not None:
        return float(KS.compute(real_data=pd.Series(real), synthetic_data=pd.Series(syn)))
    _warn_fallback()
    from scipy.stats import ks_2samp
    return 1.0 - float(ks_2samp(np.asarray(real, float), np.asarray(syn, float)).statistic)


def tv_complement(real, syn):
    _, TV = _sdm()
    if TV is not None:
        return float(TV.compute(real_data=pd.Series(real), synthetic_data=pd.Series(syn)))
    _warn_fallback()
    r = pd.Series(real).value_counts(normalize=True)
    s = pd.Series(syn).value_counts(normalize=True)
    cats = r.index.union(s.index)
    tvd = 0.5 * np.abs(r.reindex(cats, fill_value=0) - s.reindex(cats, fill_value=0)).sum()
    return 1.0 - float(tvd)


def _warn_fallback():
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        warnings.warn("SDMetrics not installed - using equivalent scipy/pandas formulas. "
                      "Install sdmetrics for the reported results.")
        _FALLBACK_WARNED = True


def evaluate(real_train, syn_set, spec):
    """Return (rows, summary) for one repetition's 100%-mixing set."""
    rows = []
    for c in spec.predictors:
        if c in spec.discrete:
            rows.append({"variable": c, "type": "discrete", "measure": "TVComplement",
                         "score": tv_complement(real_train[c], syn_set[c])})
        else:
            rows.append({"variable": c, "type": "continuous", "measure": "KSComplement",
                         "score": ks_complement(real_train[c], syn_set[c])})
    pred = pd.DataFrame(rows)
    target_check = tv_complement(real_train[spec.target], syn_set[spec.target])
    rows.append({"variable": spec.target, "type": "target (construction check)",
                 "measure": "TVComplement", "score": target_check})
    cont = pred.loc[pred["type"] == "continuous", "score"]
    disc = pred.loc[pred["type"] == "discrete", "score"]
    summary = {
        "continuous_mean": float(cont.mean()) if len(cont) else np.nan,
        "discrete_mean": float(disc.mean()) if len(disc) else np.nan,   # NaN = not applicable
        "combined_predictors": float(pred["score"].mean()),
        "n_predictors": int(len(pred)),
        "target_check": target_check,
    }
    return rows, summary
