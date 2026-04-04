"""
Target analysis node.

Deep analysis of the target variable and its relationship with all features:

  1. Target distribution — class balance for categorical targets, histogram +
     KDE + skewness + normality for numeric targets. Flags imbalance when the
     minority class represents less than 20 percent of the data.

  2. Feature-target importance — point-biserial / ANOVA for numeric features
     with binary targets, Pearson + Spearman for numeric-numeric, chi-squared +
     Cramer's V for categorical features. All features ranked by association
     strength.

  3. Feature-target separation — overlapping density plots (binary target) or
     grouped box plots (categorical target) for the top 6 most important
     features.

  4. Class-conditional statistics — per-class mean/std of numeric features,
     highlighting features where means differ by more than 0.5 std.

  5. Target correlations summary — ranked list of all features by importance.

Only runs when a target column has been identified in the research context.
"""
import warnings

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

# ── Target-type thresholds ──────────────────────────────────────────────────
BINARY_MAX_CLASSES     = 2
CATEGORICAL_MAX_CLASSES = 20
IMBALANCE_THRESHOLD    = 0.20

# ── Number of top features for separation plots ────────────────────────────
TOP_SEPARATION_FEATURES = 6


def target_analysis(state: dict) -> dict:
    """Analyse the target variable and its relationship with all features."""
    target_col = state.get("research_context", {}).get("target_column")

    if target_col is None:
        state["nodes"]["target_analysis"] = {"status": "no_target"}
        return state

    df = state["data"]
    if target_col not in df.columns:
        print_warning(f"Target column '{target_col}' not found in data.")
        state["nodes"]["target_analysis"] = {"status": "no_target"}
        return state

    images = []
    target_series = df[target_col].dropna()
    n_classes = target_series.nunique()

    # ── Determine target type ───────────────────────────────────────────────
    if n_classes <= BINARY_MAX_CLASSES:
        target_type = "binary"
    elif n_classes <= CATEGORICAL_MAX_CLASSES and (
        target_series.dtype == "object"
        or str(target_series.dtype) in ("category", "bool")
        or n_classes <= 10
    ):
        target_type = "categorical"
    else:
        target_type = "numeric"

    print_info(f"Target: {target_col}  type={target_type}  classes={n_classes}")

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Target distribution
    # ═════════════════════════════════════════════════════════════════════════
    class_balance = {}
    imbalanced = False
    distribution_stats = {}

    if target_type in ("binary", "categorical"):
        class_balance, imbalanced = _analyse_class_distribution(
            target_series, target_col
        )
        img = _plot_class_distribution(df, target_col, class_balance, state)
        if img:
            images.append(img)
    else:
        distribution_stats = _analyse_numeric_distribution(target_series)
        img = _plot_numeric_target(df, target_col, distribution_stats, state)
        if img:
            images.append(img)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Feature-target importance
    # ═════════════════════════════════════════════════════════════════════════
    feature_importance = _compute_feature_importance(
        df, target_col, target_type
    )
    feature_importance.sort(key=lambda x: x["strength"], reverse=True)

    img = _plot_feature_importance(feature_importance, state)
    if img:
        images.append(img)

    # ── Terminal: ranked list ────────────────────────────────────────────────
    print_info("Feature importance ranking (target correlations):")
    for entry in feature_importance[:15]:
        print_info(
            f"  {entry['feature']}: {entry['method']}={entry['strength']:.4f}"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Feature-target separation (top 6)
    # ═════════════════════════════════════════════════════════════════════════
    top_features = [
        e["feature"] for e in feature_importance[:TOP_SEPARATION_FEATURES]
    ]
    if top_features and target_type in ("binary", "categorical"):
        img = _plot_separation_grid(df, target_col, target_type, top_features, state)
        if img:
            images.append(img)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. Class-conditional statistics
    # ═════════════════════════════════════════════════════════════════════════
    divergent_features = []
    if target_type in ("binary", "categorical"):
        divergent_features = _class_conditional_stats(
            df, target_col, feature_importance
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 5. Target correlations summary
    # ═════════════════════════════════════════════════════════════════════════
    print_info("Target correlations summary (all features ranked):")
    for i, entry in enumerate(feature_importance, 1):
        flag = ""
        if entry["feature"] in [d["feature"] for d in divergent_features]:
            flag = "  [class means diverge]"
        print_info(
            f"  {i:>3}. {entry['feature']:<30s}  "
            f"{entry['method']}={entry['strength']:+.4f}{flag}"
        )

    # ── Store results ───────────────────────────────────────────────────────
    feature_importance_ranking = [
        {
            "feature": e["feature"],
            "method": e["method"],
            "strength": e["strength"],
        }
        for e in feature_importance
    ]

    state["nodes"]["target_analysis"] = {
        "status": "analyzed",
        "target_column": target_col,
        "target_type": target_type,
        "class_balance": class_balance,
        "feature_importance_ranking": feature_importance_ranking,
        "imbalanced": imbalanced,
        "distribution_stats": distribution_stats,
        "divergent_features": divergent_features,
        "images": [str(p) for p in images],
    }

    # ── Narrative + report section ──────────────────────────────────────────
    narrative = narrate("Target Analysis", {
        "target_column": target_col,
        "target_type": target_type,
        "class_balance": class_balance,
        "imbalanced": imbalanced,
        "top_features": feature_importance_ranking[:10],
        "divergent_features": divergent_features[:10],
        "distribution_stats": distribution_stats,
    })
    add_section(state, "Target Analysis", narrative, images)

    return state


# ═════════════════════════════════════════════════════════════════════════════
# 1. Target distribution helpers
# ═════════════════════════════════════════════════════════════════════════════


def _analyse_class_distribution(target_series, target_col):
    """Compute class counts, balance ratio, minority percentage."""
    counts = target_series.value_counts()
    total = counts.sum()
    minority_count = counts.min()
    majority_count = counts.max()

    balance_ratio = round(float(minority_count / majority_count), 4) if majority_count > 0 else 0.0
    minority_pct = round(float(minority_count / total) * 100, 2) if total > 0 else 0.0

    class_balance = {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "balance_ratio": balance_ratio,
        "minority_class": str(counts.idxmin()),
        "minority_pct": minority_pct,
    }

    imbalanced = minority_pct < (IMBALANCE_THRESHOLD * 100)

    if imbalanced:
        print_warning(
            f"Target '{target_col}' is imbalanced: minority class "
            f"'{class_balance['minority_class']}' = {minority_pct:.1f}%"
        )
    else:
        print_info(
            f"Target '{target_col}' class balance ratio: {balance_ratio:.3f}  "
            f"minority={minority_pct:.1f}%"
        )

    return class_balance, imbalanced


def _analyse_numeric_distribution(target_series):
    """Compute skewness, kurtosis, and normality test for numeric target."""
    skewness = round(float(target_series.skew()), 4)
    kurtosis_val = round(float(target_series.kurtosis()), 4)

    # Normality test (Shapiro-Wilk, capped at 5000 samples)
    normality_p = None
    is_normal = False
    try:
        _, normality_p = stats.shapiro(target_series.values[:5000])
        normality_p = round(float(normality_p), 6)
        is_normal = normality_p > 0.05
    except Exception:
        pass

    distribution_stats = {
        "skewness": skewness,
        "kurtosis": kurtosis_val,
        "normality_p": normality_p,
        "is_normal": is_normal,
        "mean": round(float(target_series.mean()), 4),
        "std": round(float(target_series.std()), 4),
        "median": round(float(target_series.median()), 4),
    }

    shape = "symmetric"
    if skewness > 1.0:
        shape = "heavily right-skewed"
    elif skewness > 0.5:
        shape = "right-skewed"
    elif skewness < -1.0:
        shape = "heavily left-skewed"
    elif skewness < -0.5:
        shape = "left-skewed"

    print_info(
        f"Target distribution: skew={skewness:+.2f}  kurtosis={kurtosis_val:.2f}  "
        f"[{shape}]  normal={'yes' if is_normal else 'no'}"
    )

    return distribution_stats


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature-target importance
# ═════════════════════════════════════════════════════════════════════════════


def _compute_feature_importance(df, target_col, target_type):
    """
    Compute association strength between every feature and the target.

    Returns a list of dicts: [{feature, method, strength, p_value}, ...].
    """
    results = []
    feature_cols = [c for c in df.columns if c != target_col]

    for feat in feature_cols:
        series = df[feat].dropna()
        if len(series) < 5:
            continue

        feat_is_numeric = str(df[feat].dtype) in (
            "int8", "int16", "int32", "int64", "float32", "float64",
        )

        entry = _measure_association(
            df, feat, target_col, feat_is_numeric, target_type
        )
        if entry is not None:
            results.append(entry)

    return results


def _measure_association(df, feat, target_col, feat_is_numeric, target_type):
    """Pick the right statistical test based on feature and target types."""
    valid = df[[feat, target_col]].dropna()
    if len(valid) < 5:
        return None

    x = valid[feat]
    y = valid[target_col]

    # ── Numeric feature x binary target ─────────────────────────────────
    if feat_is_numeric and target_type == "binary":
        try:
            classes = y.unique()
            group_a = x[y == classes[0]].astype(float)
            group_b = x[y == classes[1]].astype(float)

            # Point-biserial correlation
            y_numeric = y.map({classes[0]: 0, classes[1]: 1}).astype(float)
            pb_r, pb_p = stats.pointbiserialr(y_numeric, x.astype(float))

            # ANOVA (F-test)
            f_stat, f_p = stats.f_oneway(group_a.values, group_b.values)

            return {
                "feature": feat,
                "method": "point_biserial",
                "strength": round(abs(float(pb_r)), 4),
                "raw_value": round(float(pb_r), 4),
                "p_value": round(float(pb_p), 6),
                "anova_f": round(float(f_stat), 4),
                "anova_p": round(float(f_p), 6),
            }
        except Exception:
            return None

    # ── Numeric feature x numeric target ────────────────────────────────
    if feat_is_numeric and target_type == "numeric":
        try:
            x_float = x.astype(float)
            y_float = y.astype(float)
            pearson_r, pearson_p = stats.pearsonr(x_float, y_float)
            spearman_rho, spearman_p = stats.spearmanr(x_float, y_float)

            # Use whichever is stronger
            strength = max(abs(pearson_r), abs(spearman_rho))
            method = "pearson" if abs(pearson_r) >= abs(spearman_rho) else "spearman"

            return {
                "feature": feat,
                "method": method,
                "strength": round(float(strength), 4),
                "pearson_r": round(float(pearson_r), 4),
                "spearman_rho": round(float(spearman_rho), 4),
                "p_value": round(float(pearson_p), 6),
            }
        except Exception:
            return None

    # ── Categorical feature x binary/categorical target ─────────────────
    if not feat_is_numeric and target_type in ("binary", "categorical"):
        try:
            contingency = pd.crosstab(x, y)
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                return None

            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            n = contingency.values.sum()
            r, c = contingency.shape
            min_dim = min(r - 1, c - 1)
            cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n * min_dim > 0 else 0.0

            return {
                "feature": feat,
                "method": "cramers_v",
                "strength": round(float(cramers_v), 4),
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
            }
        except Exception:
            return None

    # ── Numeric feature x categorical target (multi-class ANOVA) ────────
    if feat_is_numeric and target_type == "categorical":
        try:
            groups = [
                grp[feat].dropna().astype(float).values
                for _, grp in valid.groupby(target_col)
                if len(grp[feat].dropna()) >= 2
            ]
            if len(groups) < 2:
                return None

            f_stat, p_value = stats.f_oneway(*groups)

            # Eta-squared
            all_vals = np.concatenate(groups)
            grand_mean = all_vals.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

            return {
                "feature": feat,
                "method": "eta_squared",
                "strength": round(float(eta_sq), 4),
                "f_stat": round(float(f_stat), 4),
                "p_value": round(float(p_value), 6),
            }
        except Exception:
            return None

    # ── Categorical feature x numeric target ────────────────────────────
    if not feat_is_numeric and target_type == "numeric":
        try:
            # Encode categories and compute Spearman
            codes = x.astype("category").cat.codes.astype(float)
            y_float = y.astype(float)
            rho, p_value = stats.spearmanr(codes, y_float)

            return {
                "feature": feat,
                "method": "spearman",
                "strength": round(abs(float(rho)), 4),
                "raw_value": round(float(rho), 4),
                "p_value": round(float(p_value), 6),
            }
        except Exception:
            return None

    return None


# ═════════════════════════════════════════════════════════════════════════════
# 4. Class-conditional statistics
# ═════════════════════════════════════════════════════════════════════════════


def _class_conditional_stats(df, target_col, feature_importance):
    """
    Group by target, compute mean/std per numeric feature per class.
    Flag features where class means differ by more than 0.5 std.
    """
    numeric_feats = [
        e["feature"] for e in feature_importance
        if str(df[e["feature"]].dtype) in (
            "int8", "int16", "int32", "int64", "float32", "float64",
        )
    ]

    if not numeric_feats:
        return []

    grouped = df.groupby(target_col)[numeric_feats]
    means = grouped.mean()
    stds = grouped.std()

    divergent = []
    overall_std = df[numeric_feats].std()

    for feat in numeric_feats:
        if overall_std[feat] == 0:
            continue
        class_means = means[feat]
        mean_range = class_means.max() - class_means.min()
        if mean_range > 0.5 * overall_std[feat]:
            divergent.append({
                "feature": feat,
                "mean_range": round(float(mean_range), 4),
                "overall_std": round(float(overall_std[feat]), 4),
                "class_means": {
                    str(k): round(float(v), 4)
                    for k, v in class_means.items()
                },
            })
            print_info(
                f"  {feat}: class means differ by {mean_range:.3f} "
                f"(> 0.5 * std={overall_std[feat]:.3f})"
            )

    if divergent:
        print_info(
            f"Features with divergent class means: {len(divergent)} / {len(numeric_feats)}"
        )
    else:
        print_info("No features with class-mean divergence > 0.5 std.")

    return divergent


# ═════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═════════════════════════════════════════════════════════════════════════════


def _plot_class_distribution(df, target_col, class_balance, state):
    """Bar chart of target class counts."""
    counts = df[target_col].value_counts()

    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.5)))
    fig.patch.set_facecolor("white")

    colors = []
    minority = class_balance.get("minority_class")
    for label in counts.index:
        if str(label) == minority and class_balance.get("minority_pct", 100) < 20:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(v) for v in counts.index], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title(
        f"Target Distribution: {target_col}",
        fontsize=13, fontweight="bold",
    )

    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / total
        ax.text(
            bar.get_width() + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)",
            va="center", fontsize=9, color="#333333",
        )

    plt.tight_layout()
    path = save_and_show(fig, state, "target_distribution.png")
    plt.close()
    return path


def _plot_numeric_target(df, target_col, distribution_stats, state):
    """Histogram + KDE for numeric target."""
    series = df[target_col].dropna().astype(float)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    bins = min(max(int(np.ceil(np.log2(len(series)) + 1)), 10), 50)
    ax.hist(series, bins=bins, color="#3498db", alpha=0.7,
            edgecolor="white", density=True, label="Histogram")

    if len(series) > 5:
        kde_x = np.linspace(series.min(), series.max(), 300)
        kde = stats.gaussian_kde(series)
        ax.plot(kde_x, kde(kde_x), color="#e74c3c", linewidth=2, label="KDE")

    skew = distribution_stats.get("skewness", 0)
    kurt = distribution_stats.get("kurtosis", 0)
    normal_label = "normal" if distribution_stats.get("is_normal") else "non-normal"
    ax.set_title(
        f"Target: {target_col}  |  skew={skew:+.2f}  kurtosis={kurt:.2f}  [{normal_label}]",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel(target_col, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = save_and_show(fig, state, "target_distribution.png")
    plt.close()
    return path


def _plot_feature_importance(feature_importance, state):
    """Horizontal bar chart of feature importances, sorted descending."""
    if not feature_importance:
        return None

    # Show at most 30 features
    to_plot = feature_importance[:30]
    features = [e["feature"] for e in to_plot]
    strengths = [e["strength"] for e in to_plot]

    fig_height = max(4, len(features) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor("white")

    colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(features)))

    bars = ax.barh(range(len(features)), strengths, color=colors, edgecolor="white")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Association Strength", fontsize=11)
    ax.set_title(
        "Feature Importance (vs Target)", fontsize=13, fontweight="bold"
    )

    for bar, val, entry in zip(bars, strengths, to_plot):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f} ({entry['method']})",
            va="center", fontsize=7, color="#333333",
        )

    plt.tight_layout()
    path = save_and_show(fig, state, "target_feature_importance.png")
    plt.close()
    return path


def _plot_separation_grid(df, target_col, target_type, top_features, state):
    """
    2x3 grid of separation plots for top features.

    Binary target  -> overlapping KDE density plots per class.
    Categorical    -> grouped box plots.
    """
    n = min(len(top_features), 6)
    if n == 0:
        return None

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    classes = df[target_col].dropna().unique()
    palette = sns.color_palette("Set2", n_colors=len(classes))

    for i in range(n):
        ax = axes_flat[i]
        feat = top_features[i]
        feat_is_numeric = str(df[feat].dtype) in (
            "int8", "int16", "int32", "int64", "float32", "float64",
        )

        if target_type == "binary" and feat_is_numeric:
            # Overlapping KDE density plots
            for j, cls in enumerate(classes):
                subset = df.loc[df[target_col] == cls, feat].dropna().astype(float)
                if len(subset) < 2:
                    continue
                sns.kdeplot(
                    subset, ax=ax, label=str(cls),
                    color=palette[j], fill=True, alpha=0.3, linewidth=1.5,
                )
            ax.legend(title=target_col, fontsize=7, title_fontsize=8)
        else:
            # Grouped box plots
            plot_df = df[[feat, target_col]].dropna()
            if feat_is_numeric:
                sns.boxplot(
                    data=plot_df, x=target_col, y=feat,
                    ax=ax, hue=target_col, palette="Set2",
                    legend=False, fliersize=2,
                )
            else:
                # Categorical feature: bar counts per class
                ct = pd.crosstab(plot_df[feat], plot_df[target_col])
                ct_pct = ct.div(ct.sum(axis=0), axis=1)
                ct_pct.plot(kind="bar", ax=ax, color=palette[:len(classes)],
                            edgecolor="white", alpha=0.8)
                ax.legend(title=target_col, fontsize=6, title_fontsize=7)

        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    # Hide unused cells
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        "Feature-Target Separation (Top Features)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = save_and_show(fig, state, "target_separation_grid.png")
    plt.close()
    return path
