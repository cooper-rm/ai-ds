"""
Bivariate analysis node.

Examines pairwise relationships across three categories:

  1. Numeric x Numeric   — scatter plots with regression lines for top
                           correlated pairs (Pearson r, Spearman rho, p-value).
  2. Categorical x Numeric — one-way ANOVA with eta-squared effect size;
                             box plot grids for the most significant pairs.
  3. Categorical x Categorical — chi-squared test of independence with
                                  Cramer's V; heatmap of association strength.

Performance guards cap the number of tests to keep runtime bounded.
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

# ── Type sets ────────────────────────────────────────────────────────────────
NUMERIC_TYPES    = {"continuous", "discrete"}
CATEGORY_TYPES   = {"categorical_nominal", "categorical_ordinal", "binary"}

# ── Performance caps ─────────────────────────────────────────────────────────
MAX_NUM_SCATTER      = 10
MAX_CAT_NUM_TESTS    = 200
MAX_CAT_CAT_TESTS    = 100
TOP_ANOVA_PLOTS      = 6
CAT_CARDINALITY_RANGE = (2, 20)

# ── Effect-size thresholds ───────────────────────────────────────────────────
ANOVA_P_THRESHOLD   = 0.05
ETA_SQ_THRESHOLD    = 0.06
CHI2_P_THRESHOLD    = 0.05
CRAMERS_V_THRESHOLD = 0.3


def bivariate(state: dict) -> dict:
    """Analyse pairwise relationships between columns."""
    df              = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})

    images  = []
    results = {}

    # ── Partition columns by type ────────────────────────────────────────────
    numeric_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in NUMERIC_TYPES
    ]
    cat_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in CATEGORY_TYPES
        and CAT_CARDINALITY_RANGE[0] <= df[col].nunique() <= CAT_CARDINALITY_RANGE[1]
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Numeric x Numeric
    # ══════════════════════════════════════════════════════════════════════════
    top_pairs = (
        state["nodes"]
        .get("correlations", {})
        .get("top_pairs", [])
    )[:MAX_NUM_SCATTER]

    num_num_results = []
    for pair in top_pairs:
        col_a, col_b = pair.get("col1", pair.get("col_a")), pair.get("col2", pair.get("col_b"))
        if col_a not in df.columns or col_b not in df.columns:
            continue
        valid = df[[col_a, col_b]].dropna()
        if len(valid) < 5:
            continue

        x, y = valid[col_a].values, valid[col_b].values
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, _      = stats.spearmanr(x, y)

        entry = {
            "col_a":        col_a,
            "col_b":        col_b,
            "pearson_r":    round(float(pearson_r), 4),
            "spearman_rho": round(float(spearman_rho), 4),
            "p_value":      float(pearson_p),
            "n":            len(valid),
        }
        num_num_results.append(entry)

        if abs(pearson_r) >= 0.7:
            print_warning(
                f"Strong correlation: {col_a} vs {col_b}  "
                f"r={pearson_r:+.3f}  rho={spearman_rho:+.3f}"
            )
        else:
            print_info(
                f"{col_a} vs {col_b}  r={pearson_r:+.3f}  rho={spearman_rho:+.3f}"
            )

    results["numeric_x_numeric"] = num_num_results

    if num_num_results:
        img = _plot_scatter_grid(df, num_num_results, state)
        if img:
            images.append(img)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Categorical x Numeric (ANOVA + eta-squared)
    # ══════════════════════════════════════════════════════════════════════════
    cat_num_pairs = list(itertools.product(cat_cols, numeric_cols))[:MAX_CAT_NUM_TESTS]
    cat_num_results = []

    for cat_col, num_col in cat_num_pairs:
        groups = [
            grp[num_col].dropna().values
            for _, grp in df.groupby(cat_col)
            if len(grp[num_col].dropna()) >= 2
        ]
        if len(groups) < 2:
            continue

        try:
            f_stat, p_value = stats.f_oneway(*groups)
        except Exception:
            continue

        # Eta-squared
        all_vals    = np.concatenate(groups)
        grand_mean  = all_vals.mean()
        ss_between  = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total    = np.sum((all_vals - grand_mean) ** 2)
        eta_sq      = float(ss_between / ss_total) if ss_total > 0 else 0.0

        entry = {
            "cat_col":   cat_col,
            "num_col":   num_col,
            "f_stat":    round(float(f_stat), 4),
            "p_value":   float(p_value),
            "eta_sq":    round(eta_sq, 4),
            "n_groups":  len(groups),
            "significant": bool(p_value < ANOVA_P_THRESHOLD and eta_sq > ETA_SQ_THRESHOLD),
        }
        cat_num_results.append(entry)

        if entry["significant"]:
            print_warning(
                f"ANOVA significant: {cat_col} x {num_col}  "
                f"F={f_stat:.2f}  p={p_value:.4f}  eta2={eta_sq:.3f}"
            )

    # Sort by eta-squared descending
    cat_num_results.sort(key=lambda e: e["eta_sq"], reverse=True)
    results["categorical_x_numeric"] = cat_num_results

    significant_cn = [e for e in cat_num_results if e["significant"]]
    if significant_cn:
        img = _plot_anova_boxplots(df, significant_cn[:TOP_ANOVA_PLOTS], state)
        if img:
            images.append(img)

    print_detail("cat x num pairs tested", len(cat_num_results))
    print_detail("significant (p<0.05, eta2>0.06)", len(significant_cn))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Categorical x Categorical (chi-squared + Cramer's V)
    # ══════════════════════════════════════════════════════════════════════════
    cat_cat_pairs = list(itertools.combinations(cat_cols, 2))[:MAX_CAT_CAT_TESTS]
    cat_cat_results = []

    for col_a, col_b in cat_cat_pairs:
        contingency = pd.crosstab(df[col_a], df[col_b])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        except Exception:
            continue

        n = contingency.values.sum()
        r, c = contingency.shape
        min_dim = min(r - 1, c - 1)
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n * min_dim > 0 else 0.0

        entry = {
            "col_a":     col_a,
            "col_b":     col_b,
            "chi2":      round(float(chi2), 4),
            "p_value":   float(p_value),
            "dof":       int(dof),
            "cramers_v": round(cramers_v, 4),
            "significant": bool(p_value < CHI2_P_THRESHOLD and cramers_v > CRAMERS_V_THRESHOLD),
        }
        cat_cat_results.append(entry)

        if entry["significant"]:
            print_warning(
                f"Associated categoricals: {col_a} x {col_b}  "
                f"V={cramers_v:.3f}  p={p_value:.4f}"
            )

    cat_cat_results.sort(key=lambda e: e["cramers_v"], reverse=True)
    results["categorical_x_categorical"] = cat_cat_results

    significant_cc = [e for e in cat_cat_results if e["significant"]]

    if len(cat_cols) >= 2 and cat_cat_results:
        img = _plot_cramers_heatmap(cat_cols, cat_cat_results, state)
        if img:
            images.append(img)

    print_detail("cat x cat pairs tested", len(cat_cat_results))
    print_detail("significant (p<0.05, V>0.3)", len(significant_cc))

    # ── Store results ────────────────────────────────────────────────────────
    state["nodes"]["bivariate"] = {
        "status":   "analyzed",
        "results":  results,
        "images":   [str(p) for p in images],
    }

    # ── Narrative + PDF section ──────────────────────────────────────────────
    narrative = narrate("Bivariate Analysis", {
        "numeric_x_numeric": [
            {"pair": f"{e['col_a']} vs {e['col_b']}", "r": e["pearson_r"],
             "rho": e["spearman_rho"]}
            for e in num_num_results
        ],
        "categorical_x_numeric_significant": [
            {"pair": f"{e['cat_col']} x {e['num_col']}", "eta_sq": e["eta_sq"],
             "p": e["p_value"]}
            for e in significant_cn[:10]
        ],
        "categorical_x_categorical_significant": [
            {"pair": f"{e['col_a']} x {e['col_b']}", "cramers_v": e["cramers_v"],
             "p": e["p_value"]}
            for e in significant_cc[:10]
        ],
    })
    add_section(state, "Bivariate Analysis", narrative, images)

    return state


# ── Scatter grid: top numeric pairs ─────────────────────────────────────────

def _plot_scatter_grid(df, num_num_results, state):
    """2-row grid of scatter plots with regression lines for top pairs."""
    n = len(num_num_results)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.5))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, entry in enumerate(num_num_results):
        ax = axes_flat[i]
        col_a, col_b = entry["col_a"], entry["col_b"]
        valid = df[[col_a, col_b]].dropna()
        x, y = valid[col_a].values, valid[col_b].values

        ax.scatter(x, y, alpha=0.35, s=12, color="#4C72B0", edgecolors="none")

        # Regression line
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#C44E52", linewidth=1.8)

        r = entry["pearson_r"]
        ax.set_title(f"{col_a} vs {col_b}\nr={r:+.3f}", fontsize=8,
                     fontweight="bold")
        ax.set_xlabel(col_a, fontsize=7)
        ax.set_ylabel(col_b, fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Top Correlated Numeric Pairs", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "bivariate_scatter_grid.png")
    plt.close()
    return path


# ── ANOVA box plots ─────────────────────────────────────────────────────────

def _plot_anova_boxplots(df, significant_pairs, state):
    """Box plot grid for the top significant categorical x numeric pairs."""
    n = len(significant_pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, entry in enumerate(significant_pairs):
        ax = axes_flat[i]
        cat_col, num_col = entry["cat_col"], entry["num_col"]

        plot_df = df[[cat_col, num_col]].dropna()
        # Sort categories by median for cleaner visual
        order = (plot_df.groupby(cat_col)[num_col]
                 .median()
                 .sort_values()
                 .index.tolist())

        sns.boxplot(data=plot_df, x=cat_col, y=num_col, order=order,
                    ax=ax, hue=cat_col, palette="Blues", legend=False, fliersize=3)

        eta = entry["eta_sq"]
        p   = entry["p_value"]
        ax.set_title(f"{cat_col} x {num_col}\neta2={eta:.3f}  p={p:.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel(cat_col, fontsize=8)
        ax.set_ylabel(num_col, fontsize=8)
        ax.tick_params(labelsize=7, axis="x", rotation=30)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Significant ANOVA Results — Box Plots", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "bivariate_anova_boxplots.png")
    plt.close()
    return path


# ── Cramer's V heatmap ──────────────────────────────────────────────────────

def _plot_cramers_heatmap(cat_cols, cat_cat_results, state):
    """Symmetric heatmap of Cramer's V for all categorical pairs."""
    # Build square matrix
    matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))),
                          index=cat_cols, columns=cat_cols)
    for i in range(len(cat_cols)):
        matrix.iloc[i, i] = 1.0

    for entry in cat_cat_results:
        a, b = entry["col_a"], entry["col_b"]
        v    = entry["cramers_v"]
        if a in matrix.index and b in matrix.columns:
            matrix.loc[a, b] = v
            matrix.loc[b, a] = v

    fig, ax = plt.subplots(figsize=(max(6, len(cat_cols) * 0.9),
                                    max(5, len(cat_cols) * 0.8)))
    fig.patch.set_facecolor("white")

    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"label": "Cramer's V"}, ax=ax)

    ax.set_title("Categorical Associations — Cramer's V", fontsize=13,
                 fontweight="bold")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = save_and_show(fig, state, "bivariate_cramers_heatmap.png")
    plt.close()
    return path
