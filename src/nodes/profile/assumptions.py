"""
Assumptions node.

Tests statistical assumptions that underpin the transforms and tests used
elsewhere in the pipeline:

  1. Homogeneity of variance (Levene's test) — validates ANOVA results from
     the bivariate node by checking whether group variances are equal.
  2. Independence (Durbin-Watson) — checks for autocorrelation in each
     numeric column's row order (lag-1 residuals from the mean).
  3. Multivariate normality — reports the fraction of numeric columns that
     passed normality tests (from the distributions node) and computes
     multivariate skewness/kurtosis on standardized numeric data.
  4. Homoscedasticity (Breusch-Pagan, manual) — for the top correlated
     numeric pairs, tests whether residual variance changes with x.

Produces a single "traffic-light" grid summary image.
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

# ── Type helpers ────────────────────────────────────────────────────────────
NUMERIC_TYPES  = {"continuous", "discrete"}
CATEGORY_TYPES = {"categorical_nominal", "categorical_ordinal", "binary"}

# ── Caps and thresholds ─────────────────────────────────────────────────────
MAX_LEVENE_TESTS       = 50
CAT_CARDINALITY_RANGE  = (2, 20)
LEVENE_ALPHA           = 0.05
DW_LOW                 = 1.5
DW_HIGH                = 2.5
BP_ALPHA               = 0.05
TOP_CORR_PAIRS_BP      = 3


def assumptions(state: dict) -> dict:
    """Test statistical assumptions and produce a traffic-light summary."""
    df              = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})

    images  = []
    results = {}

    # ── Column partitions ───────────────────────────────────────────────────
    numeric_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in NUMERIC_TYPES
    ]
    cat_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in CATEGORY_TYPES
        and CAT_CARDINALITY_RANGE[0] <= df[col].nunique() <= CAT_CARDINALITY_RANGE[1]
    ]

    # ════════════════════════════════════════════════════════════════════════
    # 1. Homogeneity of variance — Levene's test
    # ════════════════════════════════════════════════════════════════════════
    levene_results = _test_levene(df, numeric_cols, cat_cols)
    results["levene"] = levene_results

    n_flagged = sum(1 for r in levene_results if r["flag"])
    print_info(f"Levene's test: {len(levene_results)} pairs tested, "
               f"{n_flagged} with unequal variance (p < {LEVENE_ALPHA})")
    for r in levene_results:
        if r["flag"]:
            print_warning(
                f"  Unequal variance: {r['num_col']} grouped by {r['cat_col']}  "
                f"p={r['p_value']:.4f}"
            )

    # ════════════════════════════════════════════════════════════════════════
    # 2. Independence — Durbin-Watson statistic
    # ════════════════════════════════════════════════════════════════════════
    dw_results = _test_durbin_watson(df, numeric_cols)
    results["durbin_watson"] = dw_results

    n_dw_flagged = sum(1 for r in dw_results if r["flag"])
    print_info(f"Durbin-Watson: {len(dw_results)} columns tested, "
               f"{n_dw_flagged} with potential autocorrelation")
    for r in dw_results:
        if r["flag"]:
            print_warning(f"  Autocorrelation: {r['column']}  DW={r['dw']:.3f}")

    # ════════════════════════════════════════════════════════════════════════
    # 3. Multivariate normality
    # ════════════════════════════════════════════════════════════════════════
    mv_results = _test_multivariate_normality(state, df, numeric_cols)
    results["multivariate_normality"] = mv_results

    pct = mv_results.get("pct_normal", 0)
    print_info(f"Multivariate normality: {pct:.0f}% of numeric columns "
               f"passed univariate normality")
    if mv_results.get("mardia_skewness") is not None:
        print_detail("mardia skewness", round(mv_results["mardia_skewness"], 4))
        print_detail("mardia kurtosis", round(mv_results["mardia_kurtosis"], 4))

    # ════════════════════════════════════════════════════════════════════════
    # 4. Homoscedasticity — Breusch-Pagan (manual)
    # ════════════════════════════════════════════════════════════════════════
    bp_results = _test_breusch_pagan(state, df, numeric_cols)
    results["breusch_pagan"] = bp_results

    n_hetero = sum(1 for r in bp_results if r["flag"])
    print_info(f"Breusch-Pagan: {len(bp_results)} pairs tested, "
               f"{n_hetero} heteroscedastic (p < {BP_ALPHA})")
    for r in bp_results:
        if r["flag"]:
            print_warning(
                f"  Heteroscedastic: {r['x_col']} -> {r['y_col']}  "
                f"p={r['p_value']:.4f}"
            )

    # ════════════════════════════════════════════════════════════════════════
    # Traffic-light grid
    # ════════════════════════════════════════════════════════════════════════
    grid_img = _plot_traffic_light(results, numeric_cols, state)
    if grid_img:
        images.append(grid_img)

    # ── Store results ───────────────────────────────────────────────────────
    state["nodes"]["assumptions"] = {
        "status": "analyzed",
        "results": results,
        "images": [str(p) for p in images],
    }

    # ── Narrative + PDF section ─────────────────────────────────────────────
    narrative = narrate("Statistical Assumptions", {
        "levene_total": len(levene_results),
        "levene_flagged": n_flagged,
        "dw_total": len(dw_results),
        "dw_flagged": n_dw_flagged,
        "pct_normal": pct,
        "mardia_skewness": mv_results.get("mardia_skewness"),
        "mardia_kurtosis": mv_results.get("mardia_kurtosis"),
        "bp_total": len(bp_results),
        "bp_flagged": n_hetero,
    })
    add_section(state, "Statistical Assumptions", narrative, images)

    return state


# ── 1. Levene's test ────────────────────────────────────────────────────────

def _test_levene(df, numeric_cols, cat_cols):
    """
    For each numeric column grouped by each categorical column, run Levene's
    test for equality of variances.  Capped at MAX_LEVENE_TESTS.
    """
    results = []
    test_count = 0

    for cat_col, num_col in itertools.product(cat_cols, numeric_cols):
        if test_count >= MAX_LEVENE_TESTS:
            break

        groups = [
            grp[num_col].dropna().values
            for _, grp in df.groupby(cat_col)
            if len(grp[num_col].dropna()) >= 2
        ]
        if len(groups) < 2:
            continue

        try:
            stat, p_value = stats.levene(*groups)
        except Exception:
            continue

        test_count += 1
        p_value = float(p_value)
        flag = p_value < LEVENE_ALPHA

        # Marginal: 0.01 < p < 0.05 is yellow territory
        if p_value < 0.01:
            status = "fail"
        elif p_value < LEVENE_ALPHA:
            status = "marginal"
        else:
            status = "pass"

        results.append({
            "cat_col":  cat_col,
            "num_col":  num_col,
            "statistic": round(float(stat), 4),
            "p_value":  round(p_value, 6),
            "flag":     flag,
            "status":   status,
        })

    return results


# ── 2. Durbin-Watson ────────────────────────────────────────────────────────

def _test_durbin_watson(df, numeric_cols):
    """
    Compute the Durbin-Watson statistic for each numeric column.
    DW ~ 2 means no autocorrelation; < 1.5 or > 2.5 is flagged.
    """
    results = []

    for col in numeric_cols:
        series = df[col].dropna().values.astype(float)
        if len(series) < 4:
            continue

        # Residuals from the mean
        e = series - np.mean(series)
        ss = np.sum(e ** 2)
        if ss == 0:
            continue

        diff = np.diff(e)
        dw = float(np.sum(diff ** 2) / ss)

        if dw < 1.0 or dw > 3.0:
            status = "fail"
        elif dw < DW_LOW or dw > DW_HIGH:
            status = "marginal"
        else:
            status = "pass"

        flag = dw < DW_LOW or dw > DW_HIGH

        results.append({
            "column":    col,
            "dw":        round(dw, 4),
            "flag":      flag,
            "status":    status,
        })

    return results


# ── 3. Multivariate normality ──────────────────────────────────────────────

def _test_multivariate_normality(state, df, numeric_cols):
    """
    Two-pronged check:
      a) What percentage of numeric columns passed normality tests in the
         distributions node (majority-vote result)?
      b) Mardia's approximation of multivariate skewness and kurtosis on
         the standardized numeric data.
    """
    # (a) Fraction from distributions node
    dist_info = state["nodes"].get("distributions", {}).get("per_column", {})
    n_tested = 0
    n_normal = 0
    per_col_normal = {}
    for col in numeric_cols:
        normality = dist_info.get(col, {}).get("normality", {})
        is_normal = normality.get("is_normal")
        if is_normal is not None:
            n_tested += 1
            if is_normal:
                n_normal += 1
            per_col_normal[col] = is_normal

    pct_normal = (100.0 * n_normal / n_tested) if n_tested > 0 else 0.0

    result = {
        "n_tested":       n_tested,
        "n_normal":       n_normal,
        "pct_normal":     round(pct_normal, 1),
        "per_col_normal": per_col_normal,
    }

    # (b) Mardia's skewness and kurtosis
    usable = [c for c in numeric_cols if c in df.columns]
    df_num = df[usable].dropna()
    if len(df_num) >= 10 and len(usable) >= 2:
        try:
            # Standardize
            X = df_num.values.astype(float)
            mu = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            Z = (X - mu) / std

            n, p = Z.shape
            S_inv = np.linalg.pinv(np.cov(Z, rowvar=False))

            # Mardia's skewness: (1/n^2) * sum_i sum_j [ (z_i - mu)' S^-1 (z_j - mu) ]^3
            # Approximation using sample covariance
            D = Z @ S_inv @ Z.T  # n x n matrix of Mahalanobis-like products
            mardia_skew = float(np.mean(D ** 3))

            # Mardia's kurtosis: (1/n) * sum_i [ (z_i - mu)' S^-1 (z_i - mu) ]^2
            diag_D = np.diag(D)
            mardia_kurt = float(np.mean(diag_D ** 2))

            # Expected kurtosis under normality: p*(p+2)
            expected_kurt = p * (p + 2)

            result["mardia_skewness"] = round(mardia_skew, 4)
            result["mardia_kurtosis"] = round(mardia_kurt, 4)
            result["expected_kurtosis"] = round(expected_kurt, 4)

            # Flag: kurtosis far from expected, or skewness far from 0
            kurt_dev = abs(mardia_kurt - expected_kurt)
            if mardia_skew > 2.0 or kurt_dev > expected_kurt * 0.5:
                result["mv_status"] = "fail"
            elif mardia_skew > 1.0 or kurt_dev > expected_kurt * 0.25:
                result["mv_status"] = "marginal"
            else:
                result["mv_status"] = "pass"
        except Exception:
            result["mardia_skewness"] = None
            result["mardia_kurtosis"] = None
            result["mv_status"] = "unknown"
    else:
        result["mardia_skewness"] = None
        result["mardia_kurtosis"] = None
        result["mv_status"] = "unknown"

    return result


# ── 4. Breusch-Pagan (manual) ──────────────────────────────────────────────

def _test_breusch_pagan(state, df, numeric_cols):
    """
    Manual Breusch-Pagan test for the top correlated numeric pairs.
    Fit OLS y ~ x, compute squared residuals, regress on x, use F-test.
    """
    top_pairs = (
        state["nodes"]
        .get("correlations", {})
        .get("top_pairs", [])
    )[:TOP_CORR_PAIRS_BP]

    results = []

    for pair in top_pairs:
        x_col = pair.get("col1", pair.get("col_a"))
        y_col = pair.get("col2", pair.get("col_b"))
        if x_col not in df.columns or y_col not in df.columns:
            continue
        if x_col not in numeric_cols or y_col not in numeric_cols:
            continue

        valid = df[[x_col, y_col]].dropna()
        if len(valid) < 10:
            continue

        x = valid[x_col].values.astype(float)
        y = valid[y_col].values.astype(float)
        n = len(x)

        # OLS: y = a + b*x
        X_mat = np.column_stack([np.ones(n), x])
        try:
            beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        except Exception:
            continue

        y_hat = X_mat @ beta
        residuals = y - y_hat
        resid_sq = residuals ** 2

        # Regress squared residuals on x: resid_sq = c + d*x
        try:
            beta2 = np.linalg.lstsq(X_mat, resid_sq, rcond=None)[0]
        except Exception:
            continue

        resid_sq_hat = X_mat @ beta2
        ss_reg = np.sum((resid_sq_hat - np.mean(resid_sq)) ** 2)
        ss_res = np.sum((resid_sq - resid_sq_hat) ** 2)

        if ss_res == 0:
            continue

        # F-statistic: (SS_reg / k) / (SS_res / (n - k - 1)), k = 1
        k = 1
        f_stat = (ss_reg / k) / (ss_res / (n - k - 1))
        p_value = float(1.0 - stats.f.cdf(f_stat, k, n - k - 1))

        if p_value < 0.01:
            status = "fail"
        elif p_value < BP_ALPHA:
            status = "marginal"
        else:
            status = "pass"

        flag = p_value < BP_ALPHA

        results.append({
            "x_col":     x_col,
            "y_col":     y_col,
            "f_stat":    round(float(f_stat), 4),
            "p_value":   round(p_value, 6),
            "flag":      flag,
            "status":    status,
        })

    return results


# ── Traffic-light grid ─────────────────────────────────────────────────────

def _plot_traffic_light(results, numeric_cols, state):
    """
    Create a grid: rows = columns being tested, cols = assumption tests.
    Cells colored green (pass), yellow (marginal), red (fail), gray (not tested).
    """
    if not numeric_cols:
        return None

    test_names = ["Levene", "Durbin-Watson", "Normality", "Breusch-Pagan"]

    # Build a status matrix: column_name -> {test_name -> status}
    status_map = {col: {} for col in numeric_cols}

    # Levene: a column can appear in multiple groupings; take the worst
    for r in results.get("levene", []):
        col = r["num_col"]
        if col not in status_map:
            continue
        cur = status_map[col].get("Levene", "pass")
        status_map[col]["Levene"] = _worst_status(cur, r["status"])

    # Durbin-Watson
    for r in results.get("durbin_watson", []):
        col = r["column"]
        if col not in status_map:
            continue
        status_map[col]["Durbin-Watson"] = r["status"]

    # Normality (from distributions node via multivariate_normality results)
    per_col_normal = results.get("multivariate_normality", {}).get("per_col_normal", {})
    for col in numeric_cols:
        if col in per_col_normal:
            status_map[col]["Normality"] = "pass" if per_col_normal[col] else "fail"

    # Breusch-Pagan: map to both x and y columns
    for r in results.get("breusch_pagan", []):
        for col_key in ("x_col", "y_col"):
            col = r[col_key]
            if col not in status_map:
                continue
            cur = status_map[col].get("Breusch-Pagan", "pass")
            status_map[col]["Breusch-Pagan"] = _worst_status(cur, r["status"])

    # Limit rows for readability
    display_cols = numeric_cols[:25]

    # Build matrix
    n_rows = len(display_cols)
    n_cols = len(test_names)

    if n_rows == 0:
        return None

    color_map = {
        "pass":     "#2ecc71",  # green
        "marginal": "#f39c12",  # yellow/amber
        "fail":     "#e74c3c",  # red
        "unknown":  "#bdc3c7",  # gray
    }

    fig_height = max(4, n_rows * 0.45 + 1.5)
    fig_width  = max(6, n_cols * 1.8 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")

    for i, col in enumerate(display_cols):
        for j, test in enumerate(test_names):
            status = status_map.get(col, {}).get(test, "unknown")
            color = color_map.get(status, color_map["unknown"])
            rect = plt.Rectangle((j, n_rows - 1 - i), 1, 1,
                                 facecolor=color, edgecolor="white",
                                 linewidth=2)
            ax.add_patch(rect)

            # Status label in cell
            label = status[0].upper()  # P / M / F / U
            ax.text(j + 0.5, n_rows - 1 - i + 0.5, label,
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(test_names, fontsize=10, fontweight="bold")
    ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(display_cols, fontsize=9)
    ax.tick_params(length=0)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["pass"],     label="Pass"),
        Patch(facecolor=color_map["marginal"], label="Marginal"),
        Patch(facecolor=color_map["fail"],     label="Fail"),
        Patch(facecolor=color_map["unknown"],  label="Not tested"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.0, -0.05), ncol=4, fontsize=9,
              frameon=False)

    ax.set_title("Statistical Assumptions — Traffic Light Summary",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("auto")

    plt.tight_layout()
    path = save_and_show(fig, state, "assumptions_traffic_light.png")
    plt.close()
    return path


def _worst_status(a, b):
    """Return the worse of two statuses: fail > marginal > pass."""
    order = {"fail": 2, "marginal": 1, "pass": 0, "unknown": -1}
    return a if order.get(a, -1) >= order.get(b, -1) else b
