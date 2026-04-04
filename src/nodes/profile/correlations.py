"""
Correlations node.

Computes correlation matrices and multicollinearity diagnostics for all
numeric columns:

  - Pearson correlation matrix (linear relationships)
  - Spearman rank correlation matrix (monotonic relationships)
  - Variance Inflation Factor (VIF) for multicollinearity detection
  - Top correlated pairs ranking

Generates three visualizations:
  - Pearson correlation heatmap (lower triangle, annotated)
  - VIF bar chart (color-coded by severity)
  - Spearman vs Pearson scatter (highlights non-linear relationships)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")

# Correlation thresholds
STRONG_THRESHOLD      = 0.7
VERY_STRONG_THRESHOLD = 0.9

# VIF thresholds
VIF_MODERATE = 5
VIF_SEVERE   = 10

# Max columns for readable heatmap
HEATMAP_MAX_COLS = 20


def correlations(state: dict) -> dict:
    """Compute correlation matrices, VIF, and generate visualizations."""
    df     = state["data"]
    images = []

    # ── Select numeric columns ────────────────────────────────────────────
    numeric_cols = [
        col for col in df.columns
        if str(df[col].dtype) in NUMERIC_DTYPES
    ]

    # Drop constant columns (zero variance)
    constant_cols = [
        col for col in numeric_cols
        if df[col].dropna().nunique() <= 1
    ]
    if constant_cols:
        print_warning(f"Dropping constant columns from correlation: {constant_cols}")
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    if len(numeric_cols) < 2:
        print_warning("Fewer than 2 numeric columns — skipping correlation analysis.")
        state["nodes"]["correlations"] = {
            "status": "skipped",
            "reason": "fewer than 2 numeric columns",
            "images": [],
        }
        return state

    df_numeric = df[numeric_cols]

    # ── Pearson correlation ───────────────────────────────────────────────
    pearson_matrix = df_numeric.corr(method="pearson", min_periods=3)

    # Flag strong pairs
    pearson_pairs = _extract_pairs(pearson_matrix)
    strong_pearson = [
        p for p in pearson_pairs if abs(p["r"]) > STRONG_THRESHOLD
    ]
    very_strong_pearson = [
        p for p in pearson_pairs if abs(p["r"]) > VERY_STRONG_THRESHOLD
    ]

    # ── Spearman correlation ──────────────────────────────────────────────
    spearman_matrix = df_numeric.corr(method="spearman", min_periods=3)

    spearman_pairs = _extract_pairs(spearman_matrix)
    strong_spearman = [
        p for p in spearman_pairs if abs(p["r"]) > STRONG_THRESHOLD
    ]
    very_strong_spearman = [
        p for p in spearman_pairs if abs(p["r"]) > VERY_STRONG_THRESHOLD
    ]

    # ── Top correlated pairs (by absolute Pearson) ────────────────────────
    top_pairs = sorted(pearson_pairs, key=lambda x: abs(x["r"]), reverse=True)[:10]

    # ── Terminal output: top 5 pairs ──────────────────────────────────────
    print_info("Top correlated pairs (Pearson):")
    for p in top_pairs[:5]:
        label = ""
        abs_r = abs(p["r"])
        if abs_r > VERY_STRONG_THRESHOLD:
            label = " [very strong]"
        elif abs_r > STRONG_THRESHOLD:
            label = " [strong]"
        print_info(f"  {p['col1']} <-> {p['col2']}: r={p['r']:+.4f}{label}")

    # ── VIF ───────────────────────────────────────────────────────────────
    vif_results = _compute_vif(df_numeric, numeric_cols)

    # ── Visualizations ────────────────────────────────────────────────────

    # 1. Pearson heatmap
    heatmap_img = _plot_pearson_heatmap(pearson_matrix, numeric_cols, state)
    if heatmap_img:
        images.append(heatmap_img)

    # 2. VIF bar chart
    if vif_results is not None:
        vif_img = _plot_vif_bars(vif_results, state)
        if vif_img:
            images.append(vif_img)

    # 3. Spearman vs Pearson scatter
    scatter_img = _plot_spearman_vs_pearson(pearson_pairs, spearman_pairs, state)
    if scatter_img:
        images.append(scatter_img)

    # ── Store results ─────────────────────────────────────────────────────
    state["nodes"]["correlations"] = {
        "status": "analyzed",
        "pearson_matrix": pearson_matrix.to_dict(),
        "spearman_matrix": spearman_matrix.to_dict(),
        "strong_pearson": strong_pearson,
        "very_strong_pearson": very_strong_pearson,
        "strong_spearman": strong_spearman,
        "very_strong_spearman": very_strong_spearman,
        "top_pairs": top_pairs,
        "vif": vif_results,
        "constant_columns": constant_cols,
        "images": [str(p) for p in images],
    }

    # ── Narrative + report section ────────────────────────────────────────
    narrative = narrate("Correlations", {
        "top_pairs": top_pairs[:5],
        "strong_pearson_count": len(strong_pearson),
        "very_strong_pearson_count": len(very_strong_pearson),
        "strong_spearman_count": len(strong_spearman),
        "vif_warnings": (
            [v for v in vif_results if v["vif"] > VIF_MODERATE]
            if vif_results is not None else []
        ),
        "constant_columns": constant_cols,
    })
    add_section(state, "Correlations", narrative, images)

    return state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_pairs(corr_matrix):
    """Extract unique column pairs with their correlation, excluding self."""
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if pd.notna(r):
                pairs.append({
                    "col1": cols[i],
                    "col2": cols[j],
                    "r": round(float(r), 4),
                })
    return pairs


def _compute_vif(df_numeric, numeric_cols):
    """Compute VIF for each numeric column. Returns None if statsmodels unavailable."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print_warning("statsmodels not installed — skipping VIF calculation.")
        return None

    # Prepare a clean matrix: drop NaN rows, drop zero-variance columns
    df_clean = df_numeric.dropna()
    if len(df_clean) < 3:
        print_warning("Too few complete rows for VIF calculation.")
        return None

    # Drop columns with zero variance in the clean subset
    variances = df_clean.var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print_warning(f"Dropping zero-variance columns from VIF: {zero_var}")
        df_clean = df_clean.drop(columns=zero_var)

    if df_clean.shape[1] < 2:
        print_warning("Fewer than 2 columns after cleanup — skipping VIF.")
        return None

    vif_data = []
    dropped_cols = []

    # Add intercept for VIF calculation
    X = df_clean.values.astype(float)
    col_names = df_clean.columns.tolist()

    for i, col in enumerate(col_names):
        try:
            vif_val = float(variance_inflation_factor(X, i))
            if np.isinf(vif_val) or np.isnan(vif_val):
                dropped_cols.append(col)
                print_warning(f"VIF for '{col}' is infinite/NaN — perfect collinearity detected.")
                continue
            severity = "ok"
            if vif_val > VIF_SEVERE:
                severity = "severe"
            elif vif_val > VIF_MODERATE:
                severity = "moderate"
            vif_data.append({
                "column": col,
                "vif": round(vif_val, 2),
                "severity": severity,
            })
        except Exception as e:
            dropped_cols.append(col)
            print_warning(f"VIF failed for '{col}': {e}")

    # Terminal output for VIF warnings
    warnings = [v for v in vif_data if v["severity"] != "ok"]
    if warnings:
        print_info("VIF warnings:")
        for w in warnings:
            label = "severe multicollinearity" if w["severity"] == "severe" else "moderate multicollinearity"
            print_info(f"  {w['column']}: VIF={w['vif']:.1f} [{label}]")

    if dropped_cols:
        print_warning(f"Columns dropped from VIF (perfect collinearity): {dropped_cols}")

    return vif_data


# ── Visualizations ───────────────────────────────────────────────────────────


def _plot_pearson_heatmap(pearson_matrix, numeric_cols, state):
    """Lower-triangle Pearson heatmap with annotations."""
    # Cap at HEATMAP_MAX_COLS — pick the most variable columns
    if len(numeric_cols) > HEATMAP_MAX_COLS:
        df = state["data"]
        col_variance = df[numeric_cols].var().sort_values(ascending=False)
        top_cols = col_variance.head(HEATMAP_MAX_COLS).index.tolist()
        pearson_matrix = pearson_matrix.loc[top_cols, top_cols]
        print_info(f"Heatmap limited to top {HEATMAP_MAX_COLS} most variable columns.")
    else:
        top_cols = numeric_cols

    n = len(pearson_matrix)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(pearson_matrix, dtype=bool))

    fig_width  = max(8, n * 0.7)
    fig_height = max(6, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")

    # Annotate only if small enough to read
    annot = n <= 15
    fmt = ".2f" if annot else ""

    sns.heatmap(
        pearson_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )

    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    path = save_and_show(fig, state, "corr_pearson_heatmap.png")
    plt.close()
    return path


def _plot_vif_bars(vif_results, state):
    """Horizontal bar chart of VIF values, color-coded by severity."""
    if not vif_results:
        return None

    # Sort by VIF descending
    vif_sorted = sorted(vif_results, key=lambda x: x["vif"], reverse=True)

    columns = [v["column"] for v in vif_sorted]
    values  = [v["vif"] for v in vif_sorted]
    colors  = []
    for v in vif_sorted:
        if v["vif"] > VIF_SEVERE:
            colors.append("#e74c3c")   # red
        elif v["vif"] > VIF_MODERATE:
            colors.append("#e67e22")   # orange
        else:
            colors.append("#3498db")   # blue

    fig_height = max(4, len(columns) * 0.4)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    fig.patch.set_facecolor("white")

    bars = ax.barh(range(len(columns)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("VIF", fontsize=11)
    ax.set_title("Variance Inflation Factor (VIF)", fontsize=13, fontweight="bold")

    # Reference lines
    ax.axvline(x=VIF_MODERATE, color="#e67e22", linestyle="--", alpha=0.7, label=f"VIF={VIF_MODERATE} (moderate)")
    ax.axvline(x=VIF_SEVERE, color="#e74c3c", linestyle="--", alpha=0.7, label=f"VIF={VIF_SEVERE} (severe)")
    ax.legend(fontsize=8, loc="lower right")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#333333")

    plt.tight_layout()
    path = save_and_show(fig, state, "corr_vif_bars.png")
    plt.close()
    return path


def _plot_spearman_vs_pearson(pearson_pairs, spearman_pairs, state):
    """Scatter of Pearson r vs Spearman rho; color by distance from diagonal."""
    if not pearson_pairs or not spearman_pairs:
        return None

    # Build a lookup for spearman by pair key
    spearman_lookup = {}
    for p in spearman_pairs:
        key = (p["col1"], p["col2"])
        spearman_lookup[key] = p["r"]

    pearson_vals  = []
    spearman_vals = []
    labels        = []

    for p in pearson_pairs:
        key = (p["col1"], p["col2"])
        if key in spearman_lookup:
            pearson_vals.append(p["r"])
            spearman_vals.append(spearman_lookup[key])
            labels.append(f"{p['col1']} / {p['col2']}")

    if len(pearson_vals) < 2:
        return None

    pearson_arr  = np.array(pearson_vals)
    spearman_arr = np.array(spearman_vals)
    distance     = np.abs(pearson_arr - spearman_arr)

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")

    scatter = ax.scatter(
        pearson_arr, spearman_arr,
        c=distance,
        cmap="YlOrRd",
        s=40,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

    # Diagonal reference
    ax.plot([-1, 1], [-1, 1], color="#888888", linestyle="--", linewidth=1,
            alpha=0.6, label="Perfect agreement")

    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_ylabel("Spearman rho", fontsize=11)
    ax.set_title("Spearman vs Pearson Correlation", fontsize=13, fontweight="bold")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("|Pearson - Spearman|", fontsize=9)

    # Annotate the most divergent points
    if len(distance) > 0:
        n_annotate = min(3, len(distance))
        top_idx = np.argsort(distance)[-n_annotate:]
        for idx in top_idx:
            if distance[idx] > 0.05:  # only annotate meaningful divergence
                ax.annotate(
                    labels[idx],
                    (pearson_arr[idx], spearman_arr[idx]),
                    fontsize=7,
                    alpha=0.8,
                    textcoords="offset points",
                    xytext=(5, 5),
                )

    plt.tight_layout()
    path = save_and_show(fig, state, "corr_spearman_vs_pearson.png")
    plt.close()
    return path
