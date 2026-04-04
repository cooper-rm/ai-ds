"""
Regression line distortion scoring — multi-correlate + distribution shape.

Measures how much an imputation method changes:
  1. Regression relationships (slope + R²) across multiple correlates
  2. Marginal distribution shape (KS statistic + skewness/kurtosis shift)

Score = avg(Δslope) + avg(ΔR²) + KS_weight * KS_stat + shape_penalty
        (lower = less distortion)

Falls back to mean/std shift when no numeric correlate exists.
Categorical columns use Total Variation Distance (TVD).
"""
import numpy as np
from scipy import stats

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")

# How many correlates to use for multi-regression scoring
MAX_CORRELATES = 3

# Weight for distribution shape component (relative to regression component)
KS_WEIGHT = 0.3
SHAPE_WEIGHT = 0.1


def score_distortion(df, col, imputed_series, ev) -> dict:
    """
    Returns a score dict with regression distortion, distribution shape metrics,
    and a combined total. Lower total = imputation preserved data better.
    """
    correlates = [
        c for c in ev.get("mar_correlates", [])
        if c in df.columns and str(df[c].dtype) in NUMERIC_DTYPES
    ]

    if not correlates or not ev.get("is_numeric"):
        return _score_marginal(df, col, imputed_series)

    # ── Multi-correlate regression scoring ───────────────────────────────
    ref_cols = correlates[:MAX_CORRELATES]
    regression_results = []

    for ref_col in ref_cols:
        reg = _score_single_regression(df, col, imputed_series, ref_col)
        if reg is not None:
            regression_results.append(reg)

    if not regression_results:
        return _score_marginal(df, col, imputed_series)

    # Average regression distortion across correlates
    avg_delta_slope = np.mean([r["delta_slope"] for r in regression_results])
    avg_delta_r2 = np.mean([r["delta_r2"] for r in regression_results])

    # ── Distribution shape preservation ──────────────────────────────────
    shape = _score_distribution_shape(df, col, imputed_series)

    # ── Combined score ───────────────────────────────────────────────────
    regression_component = avg_delta_slope + avg_delta_r2
    distribution_component = KS_WEIGHT * shape["ks_statistic"] + SHAPE_WEIGHT * shape["shape_penalty"]
    total = round(regression_component + distribution_component, 6)

    return {
        "delta_slope":       round(avg_delta_slope, 6),
        "delta_r2":          round(avg_delta_r2, 6),
        "ks_statistic":      shape["ks_statistic"],
        "skew_shift":        shape["skew_shift"],
        "kurtosis_shift":    shape["kurtosis_shift"],
        "shape_penalty":     round(shape["shape_penalty"], 6),
        "regression_component":    round(regression_component, 6),
        "distribution_component":  round(distribution_component, 6),
        "total":             total,
        "ref_cols":          [r["ref_col"] for r in regression_results],
        "per_correlate":     regression_results,
    }


def _score_single_regression(df, col, imputed_series, ref_col) -> dict | None:
    """Score distortion against a single reference column."""
    X_ref = df[ref_col].values.astype(float)

    # Original line — complete cases only
    mask_orig = ~np.isnan(df[col].values.astype(float)) & ~np.isnan(X_ref)
    if mask_orig.sum() < 5:
        return None

    sl_orig, _, r_orig, _, _ = stats.linregress(
        X_ref[mask_orig], df[col].values[mask_orig].astype(float)
    )
    r2_orig = r_orig ** 2

    # Post-imputation line
    imp_vals = imputed_series.values.astype(float)
    mask_imp = ~np.isnan(imp_vals) & ~np.isnan(X_ref)
    if mask_imp.sum() < 5:
        return {
            "ref_col": ref_col,
            "delta_slope": 0.0, "delta_r2": 0.0,
            "slope_original": round(sl_orig, 4),
            "slope_imputed": round(sl_orig, 4),
            "r2_original": round(r2_orig, 4),
            "r2_imputed": round(r2_orig, 4),
        }

    sl_imp, _, r_imp, _, _ = stats.linregress(X_ref[mask_imp], imp_vals[mask_imp])
    r2_imp = r_imp ** 2

    delta_slope = abs(sl_imp - sl_orig) / (abs(sl_orig) + 1e-8)
    delta_r2 = abs(r2_imp - r2_orig)

    return {
        "ref_col":        ref_col,
        "delta_slope":    round(delta_slope, 6),
        "delta_r2":       round(delta_r2, 6),
        "slope_original": round(sl_orig, 4),
        "slope_imputed":  round(sl_imp, 4),
        "r2_original":    round(r2_orig, 4),
        "r2_imputed":     round(r2_imp, 4),
    }


def _score_distribution_shape(df, col, imputed_series) -> dict:
    """
    Measure how much the imputation changed the marginal distribution.
    Returns KS statistic, skewness shift, kurtosis shift, and a combined penalty.
    """
    complete = df[col].dropna().astype(float)
    imputed = imputed_series.astype(float)

    if len(complete) < 5 or len(imputed) < 5:
        return {"ks_statistic": 0.0, "skew_shift": 0.0,
                "kurtosis_shift": 0.0, "shape_penalty": 0.0}

    # KS test: compare original complete-case distribution vs full imputed distribution
    ks_stat, _ = stats.ks_2samp(complete, imputed)

    # Skewness and kurtosis shift
    skew_orig = float(complete.skew())
    skew_imp = float(imputed.skew())
    skew_shift = abs(skew_imp - skew_orig)

    kurt_orig = float(complete.kurtosis())
    kurt_imp = float(imputed.kurtosis())
    kurt_shift = abs(kurt_imp - kurt_orig)

    # Normalise kurtosis shift (kurtosis can be large, scale it down)
    shape_penalty = skew_shift + kurt_shift / (abs(kurt_orig) + 1.0)

    return {
        "ks_statistic":   round(float(ks_stat), 6),
        "skew_shift":     round(skew_shift, 4),
        "kurtosis_shift": round(kurt_shift, 4),
        "shape_penalty":  round(shape_penalty, 6),
    }


def _score_marginal(df, col, imputed_series):
    """
    Fallback scoring when no numeric correlate exists.
    Numeric columns: normalised mean + std shift + KS statistic.
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

            # Add KS + shape even for marginal fallback
            shape = _score_distribution_shape(df, col, imputed_series)
            total = round(
                mean_shift + std_shift
                + KS_WEIGHT * shape["ks_statistic"]
                + SHAPE_WEIGHT * shape["shape_penalty"],
                6,
            )
            return {
                "delta_slope": 0.0, "delta_r2": 0.0, "total": total,
                "ref_cols": None,
                "mean_shift": round(mean_shift, 4), "std_shift": round(std_shift, 4),
                "ks_statistic": shape["ks_statistic"],
                "skew_shift": shape["skew_shift"],
                "kurtosis_shift": shape["kurtosis_shift"],
                "shape_penalty": round(shape["shape_penalty"], 6),
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
            "ref_cols": None, "tvd": round(float(tvd), 4),
        }
    except Exception:
        return {"delta_slope": 0.0, "delta_r2": 0.0, "total": 0.0, "ref_cols": None}
