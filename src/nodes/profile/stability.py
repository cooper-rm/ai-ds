"""
Feature stability node.

Checks feature stability across row-order segments to detect temporal drift
or collection-order effects.  Splits training data into 4 equal segments
(Q1-Q4) by row order, then for each feature measures how much its
distribution shifts between segments.

Numeric features:  KS test + PSI (10-bin) between Q1 and Q4, plus
                   coefficient of variation of segment means.
Categorical features:  Chi-squared test between Q1 and Q4 frequencies.

Produces a stability scorecard (horizontal bar chart of PSI) and drift
line plots for the top 4 most unstable features.  Results feed into
synthesis and downstream feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")
SEGMENT_LABELS = ["Q1", "Q2", "Q3", "Q4"]
PSI_BINS = 10
PSI_STABLE = 0.1
PSI_MODERATE = 0.2
KS_ALPHA = 0.05
CHI2_ALPHA = 0.05
# Small constant to avoid log(0) and division by zero in PSI
_EPS = 1e-4


# ── PSI computation ─────────────────────────────────────────────────────────

def _compute_psi(baseline: np.ndarray, comparison: np.ndarray,
                 n_bins: int = PSI_BINS) -> float:
    """
    Population Stability Index between two numeric arrays.

    Uses ``n_bins`` equal-width bins derived from the combined range of both
    arrays.  Each bin proportion is floored at ``_EPS`` to keep the log term
    finite.
    """
    combined = np.concatenate([baseline, comparison])
    bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
    # Ensure the last edge captures the maximum value
    bin_edges[-1] += 1e-10

    base_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    comp_counts = np.histogram(comparison, bins=bin_edges)[0].astype(float)

    base_pct = base_counts / base_counts.sum()
    comp_pct = comp_counts / comp_counts.sum()

    # Floor to avoid log(0)
    base_pct = np.maximum(base_pct, _EPS)
    comp_pct = np.maximum(comp_pct, _EPS)

    psi = float(np.sum((comp_pct - base_pct) * np.log(comp_pct / base_pct)))
    return round(psi, 6)


# ── Main node ────────────────────────────────────────────────────────────────

def stability(state: dict) -> dict:
    """Check feature stability across row-order segments."""
    df = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})
    n_rows = len(df)

    results = {}
    unstable_features = []
    stable_features = []
    images = []

    # Split into 4 equal segments by row order
    seg_size = n_rows // 4
    segments = {
        "Q1": df.iloc[:seg_size],
        "Q2": df.iloc[seg_size:2 * seg_size],
        "Q3": df.iloc[2 * seg_size:3 * seg_size],
        "Q4": df.iloc[3 * seg_size:],
    }

    # ── Numeric feature stability ────────────────────────────────────────────
    numeric_cols = [
        col for col in df.columns
        if str(df[col].dtype) in NUMERIC_DTYPES
        and classifications.get(col, {}).get("type") not in ("identifier",)
    ]

    for col in numeric_cols:
        seg_stats = {}
        seg_means = []
        for label in SEGMENT_LABELS:
            series = segments[label][col].dropna().astype(float)
            if len(series) == 0:
                seg_stats[label] = {"mean": None, "std": None, "median": None}
                continue
            m = float(series.mean())
            s = float(series.std())
            med = float(series.median())
            seg_stats[label] = {
                "mean": round(m, 6),
                "std": round(s, 6),
                "median": round(med, 6),
            }
            seg_means.append(m)

        q1_series = segments["Q1"][col].dropna().astype(float).values
        q4_series = segments["Q4"][col].dropna().astype(float).values

        if len(q1_series) < 5 or len(q4_series) < 5:
            continue

        # KS test Q1 vs Q4
        ks_stat, ks_p = stats.ks_2samp(q1_series, q4_series)

        # PSI Q1 vs Q4
        psi = _compute_psi(q1_series, q4_series)

        # Coefficient of variation of segment means
        if len(seg_means) >= 2 and abs(np.mean(seg_means)) > 1e-10:
            cv = float(np.std(seg_means) / abs(np.mean(seg_means)))
        else:
            cv = 0.0

        # Classify drift severity
        if psi < PSI_STABLE:
            drift_level = "stable"
        elif psi < PSI_MODERATE:
            drift_level = "moderate_drift"
        else:
            drift_level = "significant_drift"

        is_unstable = psi > PSI_STABLE or ks_p < KS_ALPHA

        result = {
            "type": "numeric",
            "segment_stats": seg_stats,
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_p), 6),
            "psi": psi,
            "cv_of_means": round(cv, 6),
            "drift_level": drift_level,
            "unstable": is_unstable,
        }
        results[col] = result

        if is_unstable:
            unstable_features.append(col)
            print_warning(
                f"{col}: PSI={psi:.4f} ({drift_level})  "
                f"KS p={ks_p:.4f}  CV={cv:.4f}"
            )
        else:
            stable_features.append(col)
            print_info(
                f"{col}: PSI={psi:.4f} (stable)  "
                f"KS p={ks_p:.4f}  CV={cv:.4f}"
            )

    # ── Categorical feature stability ────────────────────────────────────────
    categorical_cols = [
        col for col in df.columns
        if str(df[col].dtype) not in NUMERIC_DTYPES
        and 2 <= df[col].nunique() <= 50
        and classifications.get(col, {}).get("type") not in ("identifier", "text")
    ]

    for col in categorical_cols:
        q1_counts = segments["Q1"][col].value_counts()
        q4_counts = segments["Q4"][col].value_counts()

        # Align categories
        all_cats = sorted(set(q1_counts.index) | set(q4_counts.index))
        q1_freq = np.array([q1_counts.get(c, 0) for c in all_cats], dtype=float)
        q4_freq = np.array([q4_counts.get(c, 0) for c in all_cats], dtype=float)

        if q1_freq.sum() == 0 or q4_freq.sum() == 0:
            continue

        # Chi-squared test: use Q1 proportions as expected, Q4 as observed
        q1_expected = q1_freq / q1_freq.sum() * q4_freq.sum()
        # Floor expected counts to avoid zero-division
        q1_expected = np.maximum(q1_expected, _EPS)

        try:
            chi2_stat, chi2_p = stats.chisquare(q4_freq, f_exp=q1_expected)
        except Exception:
            continue

        # Segment proportions for reporting
        seg_proportions = {}
        for label in SEGMENT_LABELS:
            counts = segments[label][col].value_counts()
            total = counts.sum()
            if total > 0:
                seg_proportions[label] = {
                    str(k): round(v / total, 4) for k, v in counts.items()
                }
            else:
                seg_proportions[label] = {}

        is_unstable = chi2_p < CHI2_ALPHA

        result = {
            "type": "categorical",
            "n_categories": len(all_cats),
            "chi2_statistic": round(float(chi2_stat), 6),
            "chi2_pvalue": round(float(chi2_p), 6),
            "segment_proportions": seg_proportions,
            "unstable": is_unstable,
            "psi": None,  # no PSI for categorical; use for sorting
        }
        results[col] = result

        if is_unstable:
            unstable_features.append(col)
            print_warning(
                f"{col}: chi2 p={chi2_p:.4f} — categorical drift detected "
                f"({len(all_cats)} categories)"
            )
        else:
            stable_features.append(col)
            print_info(
                f"{col}: chi2 p={chi2_p:.4f} — stable "
                f"({len(all_cats)} categories)"
            )

    # ── Rank features by stability (most stable first) ───────────────────────
    # For numeric: sort ascending by PSI; for categorical: sort ascending by
    # (1 - chi2_pvalue) so that high p-values (stable) come first.
    def _sort_key(col_name):
        r = results[col_name]
        if r["type"] == "numeric":
            return r["psi"]
        # Categorical: invert p-value so lower = more stable
        return 1.0 - r["chi2_pvalue"]

    ranked_features = sorted(results.keys(), key=_sort_key)
    stable_features = [c for c in ranked_features if not results[c]["unstable"]]
    unstable_features = [c for c in ranked_features if results[c]["unstable"]]

    # ── Visualizations ───────────────────────────────────────────────────────
    # 1. Stability scorecard (PSI bar chart for numeric features)
    numeric_results = {c: r for c, r in results.items() if r["type"] == "numeric"}
    if numeric_results:
        scorecard_img = _plot_stability_scorecard(numeric_results, state)
        if scorecard_img:
            images.append(scorecard_img)

    # 2. Drift line plots (top 4 most unstable numeric features)
    unstable_numeric = [
        c for c in reversed(ranked_features)
        if results[c]["type"] == "numeric"
    ][:4]
    if unstable_numeric:
        drift_img = _plot_drift_lines(unstable_numeric, results, state)
        if drift_img:
            images.append(drift_img)

    print_detail("features analyzed", len(results))
    print_detail("unstable features", len(unstable_features))
    print_detail("stable features", len(stable_features))

    state["nodes"]["stability"] = {
        "status": "analyzed",
        "results": results,
        "unstable_features": unstable_features,
        "stable_features": stable_features,
        "ranked_features": ranked_features,
        "images": [str(p) for p in images],
    }

    narrative = narrate("Feature Stability", {
        "n_features": len(results),
        "n_unstable": len(unstable_features),
        "unstable_features": {
            col: {
                "type": results[col]["type"],
                "psi": results[col].get("psi"),
                "ks_pvalue": results[col].get("ks_pvalue"),
                "chi2_pvalue": results[col].get("chi2_pvalue"),
                "drift_level": results[col].get("drift_level"),
            }
            for col in unstable_features
        },
        "stable_features": stable_features,
        "ranked_features": ranked_features,
    })
    add_section(state, "Feature Stability", narrative, images)

    return state


# ── Stability scorecard ─────────────────────────────────────────────────────

def _plot_stability_scorecard(numeric_results: dict, state: dict):
    """
    Horizontal bar chart of PSI per numeric feature, sorted descending.
    Green < 0.1, yellow 0.1-0.2, red > 0.2.
    """
    sorted_cols = sorted(numeric_results.keys(),
                         key=lambda c: numeric_results[c]["psi"],
                         reverse=True)
    psi_values = [numeric_results[c]["psi"] for c in sorted_cols]

    colors = []
    for psi in psi_values:
        if psi < PSI_STABLE:
            colors.append("#2ecc71")      # green
        elif psi < PSI_MODERATE:
            colors.append("#f39c12")      # yellow/amber
        else:
            colors.append("#e74c3c")      # red

    fig_height = max(3, len(sorted_cols) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor("white")

    y_pos = range(len(sorted_cols))
    bars = ax.barh(y_pos, psi_values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_cols, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("PSI (Population Stability Index)", fontsize=10)

    # Threshold reference lines
    ax.axvline(PSI_STABLE, color="#f39c12", linestyle="--", linewidth=1,
               alpha=0.7, label=f"Moderate ({PSI_STABLE})")
    ax.axvline(PSI_MODERATE, color="#e74c3c", linestyle="--", linewidth=1,
               alpha=0.7, label=f"Significant ({PSI_MODERATE})")

    # Value labels
    for bar, psi in zip(bars, psi_values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{psi:.4f}", va="center", fontsize=8, color="#333333")

    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Feature Stability Scorecard — PSI (Q1 vs Q4)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = save_and_show(fig, state, "stability_scorecard.png")
    plt.close()
    return path


# ── Drift line plots ────────────────────────────────────────────────────────

def _plot_drift_lines(top_cols: list, results: dict, state: dict):
    """
    2x2 grid of line plots showing segment means +/- 1 std for the top 4
    most unstable numeric features.
    """
    n = len(top_cols)
    nrows = 2
    ncols = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    x = np.arange(len(SEGMENT_LABELS))

    for i, col in enumerate(top_cols):
        ax = axes_flat[i]
        r = results[col]
        seg_stats = r["segment_stats"]

        means = []
        stds = []
        for label in SEGMENT_LABELS:
            s = seg_stats[label]
            means.append(s["mean"] if s["mean"] is not None else 0)
            stds.append(s["std"] if s["std"] is not None else 0)

        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(x, means, yerr=stds, fmt="-o", color="#4C72B0",
                    linewidth=2, markersize=8, capsize=5, capthick=1.5,
                    ecolor="#C44E52", elinewidth=1.5,
                    markerfacecolor="#4C72B0", markeredgecolor="white",
                    markeredgewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(SEGMENT_LABELS, fontsize=10)
        ax.set_ylabel("Mean value", fontsize=9)

        psi = r["psi"]
        ks_p = r["ks_pvalue"]
        ax.set_title(f"{col}\nPSI={psi:.4f}  KS p={ks_p:.4f}",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Drift Line Plots — Top Unstable Features (mean +/- 1 std)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "stability_drift_lines.png")
    plt.close()
    return path
