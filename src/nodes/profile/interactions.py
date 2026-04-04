"""
Interactions node.

Analyzes interaction effects between features, especially in relation to a
target variable:

  1. Two-way interactions with target — For the top categorical features,
     compute groupby interactions and generate interaction plots (non-parallel
     lines indicate interaction effects).
  2. Pairplot — Manual scatter/KDE grid for the top correlated numeric
     features, colored by target when available.
  3. Interaction strength scoring — Quantifies how the effect of one
     categorical feature on the target changes across levels of another.

If no target column is present, target-dependent analyses are skipped and
only the pairplot is produced.
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

# ── Type sets (mirrors bivariate / classify conventions) ─────────────────────
NUMERIC_TYPES  = {"continuous", "discrete"}
CATEGORY_TYPES = {"categorical_nominal", "categorical_ordinal", "binary"}

# ── Caps ─────────────────────────────────────────────────────────────────────
MAX_INTERACTION_PLOTS = 6          # 3x2 grid
MAX_PAIRPLOT_FEATURES = 6          # 6x6 grid
TOP_INTERACTION_PAIRS = 5
MAX_CAT_LEVELS        = 15         # skip high-cardinality categoricals


def interactions(state: dict) -> dict:
    """Analyze interaction effects between features."""
    df              = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})
    images          = []
    results         = {}

    # ── Resolve target column ────────────────────────────────────────────────
    target_col = (
        state.get("research_context", {}).get("target_column")
    )
    has_target = (
        target_col is not None
        and target_col in df.columns
        and df[target_col].notna().sum() >= 10
    )

    # ── Identify column groups ───────────────────────────────────────────────
    numeric_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in NUMERIC_TYPES
    ]
    cat_cols = [
        col for col in df.columns
        if classifications.get(col, {}).get("type") in CATEGORY_TYPES
        and 2 <= df[col].nunique() <= MAX_CAT_LEVELS
        and col != target_col
    ]

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Two-way interactions with target
    # ═════════════════════════════════════════════════════════════════════════
    interaction_plots_data = []

    if has_target and len(cat_cols) >= 2:
        # Pick the top categorical features by importance.
        # Prefer those flagged as significant in bivariate (ANOVA eta-sq or
        # chi-sq against target); fall back to first available.
        important_cats = _pick_important_cats(state, cat_cols, target_col, df)

        # Generate interaction pairs
        cat_pairs = list(itertools.combinations(important_cats, 2))[:MAX_INTERACTION_PLOTS]

        # Determine the numeric proxy for the target
        target_is_numeric = classifications.get(target_col, {}).get("type") in NUMERIC_TYPES
        if target_is_numeric:
            target_agg_col = target_col
        else:
            # For categorical targets, use the first numeric column as a proxy
            # or compute target mean by encoding
            target_agg_col = None
            if numeric_cols:
                target_agg_col = numeric_cols[0]

        if target_agg_col is not None and cat_pairs:
            img = _plot_interaction_grid(df, cat_pairs, target_col, target_agg_col,
                                         target_is_numeric, state)
            if img:
                images.append(img)
                interaction_plots_data = [
                    {"cat_1": a, "cat_2": b} for a, b in cat_pairs
                ]

        print_detail("interaction plots generated", len(interaction_plots_data))
    elif has_target:
        print_info("Fewer than 2 categorical features — skipping interaction plots.")
    else:
        print_info("No target column — skipping target interaction analysis.")

    results["interaction_plots"] = interaction_plots_data

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Pairplot (manual scatter/KDE grid)
    # ═════════════════════════════════════════════════════════════════════════
    pairplot_features = _pick_pairplot_features(state, numeric_cols)

    if len(pairplot_features) >= 2:
        # Determine hue column: use target if binary/categorical
        hue_col = None
        if has_target:
            target_type = classifications.get(target_col, {}).get("type")
            if target_type in CATEGORY_TYPES or df[target_col].nunique() <= 5:
                hue_col = target_col

        img = _plot_pairplot(df, pairplot_features, hue_col, state)
        if img:
            images.append(img)
        results["pairplot_features"] = pairplot_features
        print_detail("pairplot features", len(pairplot_features))
    else:
        print_warning("Fewer than 2 numeric features — skipping pairplot.")
        results["pairplot_features"] = []

    # ═════════════════════════════════════════════════════════════════════════
    # 3. Interaction strength scoring
    # ═════════════════════════════════════════════════════════════════════════
    interaction_scores = []

    if has_target and len(cat_cols) >= 2:
        target_is_numeric = classifications.get(target_col, {}).get("type") in NUMERIC_TYPES

        if target_is_numeric:
            interaction_scores = _score_interactions(df, cat_cols, target_col)
            interaction_scores.sort(key=lambda x: x["score"], reverse=True)
            interaction_scores = interaction_scores[:TOP_INTERACTION_PAIRS]

            if interaction_scores:
                print_info("Top interaction pairs (strength score):")
                for entry in interaction_scores:
                    print_info(
                        f"  {entry['feature_a']} x {entry['feature_b']}: "
                        f"score={entry['score']:.3f}"
                    )
        else:
            print_info(
                "Target is not numeric — skipping interaction strength scoring."
            )

    results["interaction_scores"] = interaction_scores

    # ── Store results ────────────────────────────────────────────────────────
    state["nodes"]["interactions"] = {
        "status": "analyzed",
        "target_column": target_col if has_target else None,
        "interaction_plots": interaction_plots_data,
        "pairplot_features": results.get("pairplot_features", []),
        "interaction_scores": interaction_scores,
        "images": [str(p) for p in images],
    }

    # ── Narrative + report section ───────────────────────────────────────────
    narrative = narrate("Feature Interactions", {
        "target_column": target_col if has_target else None,
        "interaction_plots_count": len(interaction_plots_data),
        "pairplot_features": results.get("pairplot_features", []),
        "top_interactions": interaction_scores[:3] if interaction_scores else [],
    })
    add_section(state, "Feature Interactions", narrative, images)

    return state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pick_important_cats(state, cat_cols, target_col, df):
    """Return up to 5 most important categorical features for interactions."""
    # Try bivariate ANOVA results first (features with highest eta-sq)
    bivariate_results = (
        state["nodes"]
        .get("bivariate", {})
        .get("results", {})
        .get("categorical_x_numeric", [])
    )

    ranked = {}
    for entry in bivariate_results:
        col = entry.get("cat_col")
        if col in cat_cols:
            # Keep the best eta-sq per feature
            ranked[col] = max(ranked.get(col, 0), entry.get("eta_sq", 0))

    if ranked:
        sorted_cats = sorted(ranked, key=ranked.get, reverse=True)
        return sorted_cats[:5]

    # Fallback: pick categoricals with lowest cardinality (more interpretable)
    return sorted(cat_cols, key=lambda c: df[c].nunique())[:5]


def _pick_pairplot_features(state, numeric_cols):
    """Return up to MAX_PAIRPLOT_FEATURES most correlated numeric features."""
    top_pairs = (
        state["nodes"]
        .get("correlations", {})
        .get("top_pairs", [])
    )

    # Collect features that appear in the strongest correlations
    feature_set = []
    seen = set()
    for pair in top_pairs:
        for key in ("col1", "col2", "col_a", "col_b"):
            col = pair.get(key)
            if col and col in numeric_cols and col not in seen:
                seen.add(col)
                feature_set.append(col)
            if len(feature_set) >= MAX_PAIRPLOT_FEATURES:
                break
        if len(feature_set) >= MAX_PAIRPLOT_FEATURES:
            break

    # If we still have room, fill with remaining numeric cols
    for col in numeric_cols:
        if col not in seen:
            feature_set.append(col)
            seen.add(col)
        if len(feature_set) >= MAX_PAIRPLOT_FEATURES:
            break

    return feature_set[:MAX_PAIRPLOT_FEATURES]


# ── Interaction strength scoring ─────────────────────────────────────────────


def _score_interactions(df, cat_cols, target_col):
    """
    Score interaction strength for each pair of categorical features.

    For each pair (A, B), the score measures whether the effect of A on the
    target changes across levels of B:

        score = std(group_means_by_A_within_each_B_level) /
                std(overall_group_means_by_A)

    A score near 1 means no interaction; significantly above 1 means
    the effect of A varies depending on B.
    """
    scores = []

    pairs = list(itertools.combinations(cat_cols, 2))
    for feat_a, feat_b in pairs:
        score = _compute_interaction_score(df, feat_a, feat_b, target_col)
        if score is not None:
            scores.append({
                "feature_a": feat_a,
                "feature_b": feat_b,
                "score": round(float(score), 4),
            })

    return scores


def _compute_interaction_score(df, feat_a, feat_b, target_col):
    """Compute interaction score for a single pair."""
    subset = df[[feat_a, feat_b, target_col]].dropna()
    if len(subset) < 10:
        return None

    # Overall group means of target by feat_a
    overall_means_a = subset.groupby(feat_a)[target_col].mean()
    if len(overall_means_a) < 2:
        return None

    overall_std = overall_means_a.std()
    if overall_std == 0 or np.isnan(overall_std):
        return None

    # Group means of target by feat_a, within each level of feat_b
    conditional_stds = []
    for level_b, group_b in subset.groupby(feat_b):
        if len(group_b) < 5:
            continue
        means_a_within_b = group_b.groupby(feat_a)[target_col].mean()
        if len(means_a_within_b) >= 2:
            conditional_stds.append(means_a_within_b.std())

    if len(conditional_stds) < 2:
        return None

    avg_conditional_std = np.mean(conditional_stds)
    score = avg_conditional_std / overall_std

    return score


# ── Visualizations ───────────────────────────────────────────────────────────


def _plot_interaction_grid(df, cat_pairs, target_col, target_agg_col,
                           target_is_numeric, state):
    """Line-plot grid showing two-way interactions with the target."""
    n = len(cat_pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    palette = sns.color_palette("Set2", MAX_CAT_LEVELS)

    for i, (cat_1, cat_2) in enumerate(cat_pairs):
        ax = axes_flat[i]

        if target_is_numeric:
            plot_df = df[[cat_1, cat_2, target_col]].dropna()
            agg_col = target_col
        else:
            plot_df = df[[cat_1, cat_2, target_agg_col]].dropna()
            agg_col = target_agg_col

        if len(plot_df) < 5:
            ax.set_visible(False)
            continue

        # Compute grouped means
        grouped = (
            plot_df
            .groupby([cat_1, cat_2])[agg_col]
            .mean()
            .reset_index()
        )

        # Get unique levels
        levels_1 = sorted(grouped[cat_1].unique(), key=str)
        levels_2 = sorted(grouped[cat_2].unique(), key=str)

        for j, level_2 in enumerate(levels_2):
            sub = grouped[grouped[cat_2] == level_2]
            color = palette[j % len(palette)]
            ax.plot(
                sub[cat_1].astype(str),
                sub[agg_col],
                marker="o",
                label=f"{cat_2}={level_2}",
                color=color,
                linewidth=1.5,
                markersize=5,
            )

        y_label = f"Mean {agg_col}"
        ax.set_title(f"{cat_1} x {cat_2}", fontsize=9, fontweight="bold")
        ax.set_xlabel(cat_1, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=7, axis="x", rotation=30)

        # Limit legend entries
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) <= 8:
            ax.legend(fontsize=6, loc="best")
        else:
            ax.legend(handles[:8], labels[:8], fontsize=6, loc="best",
                      title="(first 8)")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Two-Way Interaction Effects on Target", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "interactions_two_way.png")
    plt.close()
    return path


def _plot_pairplot(df, features, hue_col, state):
    """
    Manual pairplot grid: KDE on diagonal, scatter on off-diagonal.

    Built with plt.subplots to keep figure control (sns.pairplot creates its
    own figure which is hard to save via save_and_show).
    """
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(n * 2.2, n * 2.2))
    fig.patch.set_facecolor("white")

    # Prepare hue info
    hue_values = None
    hue_labels = None
    palette = None
    if hue_col is not None and hue_col in df.columns:
        hue_series = df[hue_col].dropna()
        hue_labels = sorted(hue_series.unique(), key=str)
        if len(hue_labels) > 10:
            # Too many categories — drop hue
            hue_col = None
            hue_labels = None
        else:
            palette = sns.color_palette("Set2", len(hue_labels))

    for row in range(n):
        for col in range(n):
            ax = axes[row][col] if n > 1 else axes

            feat_y = features[row]
            feat_x = features[col]

            if row == col:
                # Diagonal: KDE
                if hue_col and hue_labels:
                    for k, level in enumerate(hue_labels):
                        mask = df[hue_col] == level
                        vals = df.loc[mask, feat_x].dropna()
                        if len(vals) >= 2:
                            sns.kdeplot(
                                vals, ax=ax, color=palette[k],
                                fill=True, alpha=0.3, linewidth=1,
                                label=str(level), warn_singular=False,
                            )
                else:
                    vals = df[feat_x].dropna()
                    if len(vals) >= 2:
                        sns.kdeplot(
                            vals, ax=ax, color="#4C72B0",
                            fill=True, alpha=0.4, linewidth=1,
                            warn_singular=False,
                        )
            else:
                # Off-diagonal: scatter
                plot_df = df[[feat_x, feat_y]].dropna()

                if hue_col and hue_labels:
                    merged = df[[feat_x, feat_y, hue_col]].dropna()
                    for k, level in enumerate(hue_labels):
                        mask = merged[hue_col] == level
                        sub = merged.loc[mask]
                        ax.scatter(
                            sub[feat_x], sub[feat_y],
                            alpha=0.3, s=8, color=palette[k],
                            edgecolors="none", label=str(level),
                        )
                else:
                    ax.scatter(
                        plot_df[feat_x], plot_df[feat_y],
                        alpha=0.25, s=8, color="#4C72B0",
                        edgecolors="none",
                    )

            # Axis labels: only on edges
            if row == n - 1:
                ax.set_xlabel(feat_x, fontsize=7)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel(feat_y, fontsize=7)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.tick_params(labelsize=5)

    # Add a single legend for hue if used
    if hue_col and hue_labels:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=palette[k], markersize=6,
                       label=str(level))
            for k, level in enumerate(hue_labels)
        ]
        fig.legend(
            handles=handles, title=hue_col, fontsize=7, title_fontsize=8,
            loc="upper right", bbox_to_anchor=(0.99, 0.99),
        )

    plt.suptitle("Pairplot — Top Correlated Features", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "interactions_pairplot.png")
    plt.close()
    return path
