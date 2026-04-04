"""
Imputation comparison plots.

4-panel layout per column:
  1. Missingness pattern (row scatter)
  2. Distribution overlay — complete cases + each candidate imputation
  3. Regression line before vs after best imputation
  4. Distortion bar chart — total score per method
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.utils import save_and_show

NUMERIC_DTYPES = ("int8","int16","int32","int64","float32","float64")


def plot_comparison(df, col, method_results, best_method, ev, state):
    """Dispatch to numeric or categorical plot."""
    if ev.get("is_numeric"):
        return _plot_numeric(df, col, method_results, best_method, ev, state)
    return _plot_categorical(df, col, method_results, best_method, state)


def _plot_numeric(df, col, method_results, best_method, ev, state):
    correlates = [
        c for c in ev.get("mar_correlates", [])
        if c in df.columns and str(df[c].dtype) in NUMERIC_DTYPES
    ]
    ref_col  = correlates[0] if correlates else None
    complete = df[col].dropna().astype(float)
    bins     = min(30, max(10, len(complete) // 10))
    colors   = plt.cm.tab10(np.linspace(0, 0.9, len(method_results)))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.patch.set_facecolor("white")

    # ── Panel 1: missingness pattern ─────────────────────────────────────────
    missing_idx = df[df[col].isnull()].index
    axes[0].scatter(missing_idx, np.ones(len(missing_idx)),
                    color="#C44E52", s=6, alpha=0.6, label="missing")
    axes[0].scatter(df[col].dropna().index, np.zeros(len(complete)),
                    color="#4C72B0", s=4, alpha=0.3, label="present")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["present", "missing"], fontsize=8)
    axes[0].set_xlabel("Row index", fontsize=8)
    axes[0].set_title("Missingness Pattern", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=7)

    # ── Panel 2: distribution overlay ────────────────────────────────────────
    axes[1].hist(complete, bins=bins, color="#4C72B0", alpha=0.5,
                 density=True, label="complete cases")
    for (method, res), color in zip(method_results.items(), colors):
        imputed_only = res["imputed"][df[col].isnull()]
        if len(imputed_only) == 0:
            continue
        lw   = 2.5 if method == best_method else 1.0
        alph = 0.9 if method == best_method else 0.45
        lbl  = f"{method} ★" if method == best_method else method
        axes[1].hist(
            imputed_only.astype(float), bins=bins, density=True,
            alpha=alph, linewidth=lw, histtype="step", color=color, label=lbl,
        )
    axes[1].set_title("Imputed Value Distributions", fontsize=10, fontweight="bold")
    axes[1].set_xlabel(col, fontsize=8)
    axes[1].legend(fontsize=6, ncol=2)

    # ── Panel 3: regression line before vs after ──────────────────────────────
    if ref_col:
        X_ref    = df[ref_col].values.astype(float)
        mask_obs = ~np.isnan(df[col].values) & ~np.isnan(X_ref)
        xline    = np.linspace(np.nanmin(X_ref), np.nanmax(X_ref), 100)

        # Complete-case scatter + original line
        axes[2].scatter(X_ref[mask_obs], df[col].values[mask_obs],
                        color="#4C72B0", s=8, alpha=0.35, label="complete cases")
        sl, ic, r, _, _ = stats.linregress(X_ref[mask_obs], df[col].values[mask_obs])
        axes[2].plot(xline, sl * xline + ic, color="#4C72B0", linewidth=2,
                     linestyle="--", label=f"original  R²={r**2:.2f}")

        # Best method: imputed points + new line
        best_imp = method_results[best_method]["imputed"].values.astype(float)
        mask_imp = ~np.isnan(best_imp) & ~np.isnan(X_ref)
        axes[2].scatter(X_ref[df[col].isnull()], best_imp[df[col].isnull()],
                        color="#C44E52", s=20, alpha=0.75, zorder=5,
                        label=f"imputed ({best_method})")
        sl2, ic2, r2, _, _ = stats.linregress(X_ref[mask_imp], best_imp[mask_imp])
        axes[2].plot(xline, sl2 * xline + ic2, color="#C44E52", linewidth=2,
                     label=f"{best_method}  R²={r2**2:.2f}")

        axes[2].set_xlabel(ref_col, fontsize=8)
        axes[2].set_ylabel(col, fontsize=8)
        axes[2].set_title("Regression Line: Before vs After", fontsize=10, fontweight="bold")
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, "No numeric\ncorrelate", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=10)
        axes[2].set_title("Regression Line", fontsize=10, fontweight="bold")

    # ── Panel 4: distortion bar chart ────────────────────────────────────────
    names      = list(method_results.keys())
    totals     = [method_results[m]["score"]["total"] for m in names]
    bar_colors = ["#2ecc71" if m == best_method else "#95a5a6" for m in names]
    short      = [m.replace("stochastic_regression", "stoch_reg") for m in names]

    bars = axes[3].bar(range(len(names)), totals, color=bar_colors, edgecolor="white")
    axes[3].set_xticks(range(len(names)))
    axes[3].set_xticklabels(short, rotation=28, ha="right", fontsize=7)
    axes[3].set_title("Total Distortion by Method\n(lower = better)", fontsize=10, fontweight="bold")
    axes[3].set_ylabel("Distortion Score", fontsize=8)
    axes[3].bar_label(bars, fmt="%.3f", fontsize=7)

    plt.suptitle(f"Imputation Analysis: {col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_and_show(fig, state, f"imputation_{col.lower()}.png")
    plt.close()
    return path


def _plot_categorical(df, col, method_results, best_method, state):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    counts = df[col].value_counts()
    axes[0].barh(range(len(counts)), counts.values, color="#4C72B0", alpha=0.7)
    axes[0].set_yticks(range(len(counts)))
    axes[0].set_yticklabels(counts.index.tolist(), fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title(f"{col}: Original", fontsize=10, fontweight="bold")

    best_imp   = method_results[best_method]["imputed"]
    imp_counts = best_imp.value_counts()
    axes[1].barh(range(len(imp_counts)), imp_counts.values, color="#2ecc71", alpha=0.7)
    axes[1].set_yticks(range(len(imp_counts)))
    axes[1].set_yticklabels(imp_counts.index.tolist(), fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title(f"{col}: After {best_method}", fontsize=10, fontweight="bold")

    plt.suptitle(f"Imputation Analysis: {col}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_and_show(fig, state, f"imputation_{col.lower()}.png")
    plt.close()
    return path
