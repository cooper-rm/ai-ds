"""
Regression line distortion scoring.

Measures how much an imputation method changes the relationship between
the imputed column and its strongest correlate.

Score = |Δslope| / |slope_orig| + |ΔR²|   (lower = less distortion)

Falls back to mean/std shift when no numeric correlate exists.
"""
import numpy as np
from scipy import stats

NUMERIC_DTYPES = ("int8","int16","int32","int64","float32","float64")


def score_distortion(df, col, imputed_series, ev) -> dict:
    """
    Returns a score dict with delta_slope, delta_r2, total, and ref_col.
    Lower total = imputation preserved relationships better.
    """
    correlates = [
        c for c in ev.get("mar_correlates", [])
        if c in df.columns and str(df[c].dtype) in NUMERIC_DTYPES
    ]

    if not correlates or not ev.get("is_numeric"):
        return _score_marginal(df, col, imputed_series)

    ref_col = correlates[0]
    X_ref   = df[ref_col].values.astype(float)

    # Original line — complete cases only
    mask_orig = ~np.isnan(df[col].values) & ~np.isnan(X_ref)
    if mask_orig.sum() < 5:
        return _score_marginal(df, col, imputed_series)

    sl_orig, ic_orig, r_orig, _, _ = stats.linregress(
        X_ref[mask_orig], df[col].values[mask_orig]
    )
    r2_orig = r_orig ** 2

    # Post-imputation line
    imp_vals = imputed_series.values.astype(float)
    mask_imp = ~np.isnan(imp_vals) & ~np.isnan(X_ref)
    if mask_imp.sum() < 5:
        return {"delta_slope": 0.0, "delta_r2": 0.0, "total": 0.0,
                "ref_col": ref_col, "r2_original": round(r2_orig, 4)}

    sl_imp, _, r_imp, _, _ = stats.linregress(X_ref[mask_imp], imp_vals[mask_imp])
    r2_imp = r_imp ** 2

    delta_slope = abs(sl_imp - sl_orig) / (abs(sl_orig) + 1e-8)
    delta_r2    = abs(r2_imp - r2_orig)
    total       = round(delta_slope + delta_r2, 6)

    return {
        "delta_slope":    round(delta_slope, 6),
        "delta_r2":       round(delta_r2, 6),
        "total":          total,
        "ref_col":        ref_col,
        "slope_original": round(sl_orig, 4),
        "slope_imputed":  round(sl_imp,  4),
        "r2_original":    round(r2_orig, 4),
        "r2_imputed":     round(r2_imp,  4),
    }


def _score_marginal(df, col, imputed_series):
    """
    Fallback scoring when no numeric correlate exists.
    Numeric columns: normalised mean + std shift.
    Categorical columns: total variation distance on value proportions.
    """
    is_numeric = str(df[col].dtype) in NUMERIC_DTYPES

    if is_numeric:
        try:
            complete = df[col].dropna().astype(float)
            imp_vals = imputed_series.astype(float)
            std_ref    = float(complete.std()) + 1e-8
            mean_shift = abs(float(imp_vals.mean()) - float(complete.mean())) / std_ref
            std_shift  = abs(float(imp_vals.std())  - float(complete.std()))  / std_ref
            total = round(mean_shift + std_shift, 6)
            return {
                "delta_slope": 0.0, "delta_r2": 0.0, "total": total,
                "ref_col": None,
                "mean_shift": round(mean_shift, 4), "std_shift": round(std_shift, 4),
            }
        except Exception:
            pass

    # Categorical — total variation distance on value proportions
    try:
        orig_props = df[col].dropna().value_counts(normalize=True)
        imp_props  = imputed_series.dropna().value_counts(normalize=True)
        all_cats   = orig_props.index.union(imp_props.index)
        tvd = sum(abs(orig_props.get(c, 0.0) - imp_props.get(c, 0.0)) for c in all_cats) / 2
        return {
            "delta_slope": 0.0, "delta_r2": 0.0,
            "total": round(float(tvd), 6),
            "ref_col": None, "tvd": round(float(tvd), 4),
        }
    except Exception:
        return {"delta_slope": 0.0, "delta_r2": 0.0, "total": 0.0, "ref_col": None}
