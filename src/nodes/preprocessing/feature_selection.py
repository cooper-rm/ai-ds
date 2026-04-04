"""
Feature Selection node.

Runs multiple feature selection methods on the post-encoding numeric data
and produces a consensus recommendation for which features to remove:

  - Variance threshold (drop features with variance < 0.01 after scaling)
  - Correlation filter (drop one of each pair with |Pearson r| > 0.95)
  - Mutual information (flag features with MI < 0.01 across all others)
  - Tree-based importance (flag features with importance < 0.01)

A feature is recommended for removal if flagged by 2+ methods.
No columns are actually dropped — only recommendations are recorded.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

# Thresholds
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
MI_THRESHOLD = 0.01
IMPORTANCE_THRESHOLD = 0.01


def feature_selection(state: dict) -> dict:
    """Run multiple feature selection methods and produce consensus recommendations."""
    df = state["data"]
    images = []

    # ── Select numeric columns ───────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        state["nodes"]["feature_selection"] = {
            "status": "skipped",
            "reason": "fewer than 2 numeric columns",
        }
        print_info("fewer than 2 numeric columns, skipping feature selection")
        return state

    df_numeric = df[numeric_cols].copy()

    # Identify target column if available
    target_col = _find_target(state, numeric_cols)

    # ── Method 1: Variance Threshold ─────────────────────────────────────
    low_variance = _variance_filter(df_numeric)
    print_info(f"variance filter: {len(low_variance)} features below {VARIANCE_THRESHOLD}")
    for col in low_variance:
        print_info(f"  low variance: {col}")

    # ── Method 2: Correlation Filter ─────────────────────────────────────
    corr_drop = _correlation_filter(df_numeric, state)
    print_info(f"correlation filter: {len(corr_drop)} features redundant (|r| > {CORRELATION_THRESHOLD})")
    for col in corr_drop:
        print_info(f"  correlated: {col}")

    # ── Method 3: Mutual Information ─────────────────────────────────────
    mi_drop = _mutual_information_filter(df_numeric, target_col)
    print_info(f"mutual information filter: {len(mi_drop)} uninformative features")
    for col in mi_drop:
        print_info(f"  low MI: {col}")

    # ── Method 4: Tree-Based Importance ──────────────────────────────────
    importances, tree_drop = _tree_importance(df_numeric, target_col)
    print_info(f"tree importance filter: {len(tree_drop)} features below {IMPORTANCE_THRESHOLD}")
    for col in tree_drop:
        print_info(f"  low importance: {col}")

    # ── Consensus ────────────────────────────────────────────────────────
    flag_counts = {}
    method_labels = {
        "variance": low_variance,
        "correlation": corr_drop,
        "mutual_information": mi_drop,
        "tree_importance": tree_drop,
    }
    for method, flagged in method_labels.items():
        for col in flagged:
            flag_counts[col] = flag_counts.get(col, 0) + 1

    recommended_removal = [col for col, count in flag_counts.items() if count >= 2]
    recommended_removal.sort()

    print_info(f"consensus: {len(recommended_removal)} features recommended for removal (flagged by 2+ methods)")
    for col in recommended_removal:
        methods_hit = [m for m, flagged in method_labels.items() if col in flagged]
        print_warning(f"  remove {col}  ({', '.join(methods_hit)})")

    if not recommended_removal:
        print_info("no features recommended for removal")

    # ── Visualization ────────────────────────────────────────────────────
    img = _plot_importance(importances, recommended_removal, state)
    if img:
        images.append(img)

    # ── Store results ────────────────────────────────────────────────────
    method_results = {
        "variance": {
            "threshold": VARIANCE_THRESHOLD,
            "flagged": low_variance,
        },
        "correlation": {
            "threshold": CORRELATION_THRESHOLD,
            "flagged": corr_drop,
        },
        "mutual_information": {
            "threshold": MI_THRESHOLD,
            "flagged": mi_drop,
        },
        "tree_importance": {
            "threshold": IMPORTANCE_THRESHOLD,
            "flagged": tree_drop,
            "importances": {col: round(float(v), 6) for col, v in importances.items()},
        },
    }

    state["nodes"]["feature_selection"] = {
        "status": "completed",
        "recommendations": recommended_removal,
        "flag_counts": {col: int(count) for col, count in flag_counts.items()},
        "method_results": method_results,
        "images": [str(p) for p in images],
    }

    narrative = narrate("Feature Selection", {
        "recommendations": recommended_removal,
        "method_results": {m: {"flagged": r["flagged"]} for m, r in method_results.items()},
        "total_features": len(numeric_cols),
    })
    add_section(state, "Feature Selection", narrative, images)

    return state


# ── Helpers ──────────────────────────────────────────────────────────────


def _find_target(state: dict, numeric_cols: list) -> str | None:
    """Try to identify a target column from state metadata."""
    # Check synthesis plan
    plan = state.get("nodes", {}).get("synthesis", {})
    target = plan.get("target") or plan.get("target_column")
    if target and target in numeric_cols:
        return target

    # Check classify node
    classify = state.get("nodes", {}).get("classify", {})
    target = classify.get("target") or classify.get("target_column")
    if target and target in numeric_cols:
        return target

    return None


def _variance_filter(df: pd.DataFrame) -> list[str]:
    """Return columns with variance below threshold after standard scaling."""
    df_clean = df.dropna()
    if df_clean.empty or df_clean.shape[0] < 2:
        return []

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=df_clean.columns,
        index=df_clean.index,
    )

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    selector.fit(scaled)

    mask = selector.get_support()
    return [col for col, keep in zip(df.columns, mask) if not keep]


def _correlation_filter(df: pd.DataFrame, state: dict) -> list[str]:
    """Return columns to drop from highly correlated pairs (keep higher-variance one)."""
    # Reuse precomputed correlation matrix if available
    corr_node = state.get("nodes", {}).get("correlations", {})
    pearson = corr_node.get("pearson_matrix")

    if pearson is not None:
        corr_matrix = pd.DataFrame(pearson) if isinstance(pearson, dict) else pearson
    else:
        corr_matrix = df.corr(method="pearson")

    # Align correlation matrix to current columns
    shared = [c for c in df.columns if c in corr_matrix.columns]
    corr_matrix = corr_matrix.loc[shared, shared]

    variances = df[shared].var()
    to_drop = set()
    checked = set()

    for i, col_a in enumerate(shared):
        for j, col_b in enumerate(shared):
            if j <= i:
                continue
            if col_a in to_drop or col_b in to_drop:
                continue
            r = abs(corr_matrix.loc[col_a, col_b])
            if r > CORRELATION_THRESHOLD:
                # Drop the one with lower variance
                if variances[col_a] >= variances[col_b]:
                    to_drop.add(col_b)
                else:
                    to_drop.add(col_a)

    return sorted(to_drop)


def _mutual_information_filter(df: pd.DataFrame, target_col: str | None) -> list[str]:
    """Return features with MI < threshold across all other features (or target)."""
    df_clean = df.dropna()
    if df_clean.empty or df_clean.shape[0] < 10:
        return []

    cols = df_clean.columns.tolist()
    max_mi = pd.Series(0.0, index=cols)

    if target_col and target_col in df_clean.columns:
        # Compute MI against the target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
        for col, score in zip(X.columns, mi_scores):
            max_mi[col] = max(max_mi[col], score)
        # Target itself is never flagged
        max_mi[target_col] = 1.0
    else:
        # No target: compute MI of each feature against a sample of other features
        # Use up to 5 pseudo-targets to keep it tractable
        n_targets = min(5, len(cols))
        rng = np.random.RandomState(42)
        pseudo_targets = rng.choice(cols, size=n_targets, replace=False)

        for pt in pseudo_targets:
            X = df_clean.drop(columns=[pt])
            y = df_clean[pt]
            mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
            for col, score in zip(X.columns, mi_scores):
                max_mi[col] = max(max_mi[col], score)
            # The pseudo-target gets its own MI as a target (always informative)
            max_mi[pt] = max(max_mi[pt], max(mi_scores) if len(mi_scores) > 0 else 0.0)

    return [col for col in cols if max_mi[col] < MI_THRESHOLD]


def _tree_importance(df: pd.DataFrame, target_col: str | None) -> tuple[dict, list[str]]:
    """Compute tree-based feature importances. Return (importances_dict, flagged_list)."""
    df_clean = df.dropna()
    if df_clean.empty or df_clean.shape[0] < 10:
        importances = {col: 0.0 for col in df.columns}
        return importances, list(df.columns)

    cols = df_clean.columns.tolist()

    if target_col and target_col in df_clean.columns:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        # Determine if target is categorical (few unique values relative to size)
        n_unique = y.nunique()
        is_classification = n_unique <= 20 and n_unique / len(y) < 0.05

        if is_classification:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
            )

        model.fit(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))
        # Target itself gets importance 1.0 (never flagged)
        importances[target_col] = 1.0
    else:
        # No target: use each numeric column as pseudo-target and average
        importance_accum = pd.Series(0.0, index=cols)
        n_targets = min(5, len(cols))
        rng = np.random.RandomState(42)
        pseudo_targets = rng.choice(cols, size=n_targets, replace=False)

        for pt in pseudo_targets:
            X = df_clean.drop(columns=[pt])
            y = df_clean[pt]
            model = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1,
            )
            model.fit(X, y)
            for col, imp in zip(X.columns, model.feature_importances_):
                importance_accum[col] += imp
            # Pseudo-target gets average of its own predictive power
            importance_accum[pt] += np.mean(model.feature_importances_)

        importance_accum /= n_targets
        importances = importance_accum.to_dict()

    flagged = [col for col, imp in importances.items() if imp < IMPORTANCE_THRESHOLD]
    return importances, flagged


def _plot_importance(importances: dict, recommended_removal: list[str], state: dict):
    """Horizontal bar chart of features ranked by tree-based importance."""
    if not importances:
        return None

    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=False)
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    colors = ["#C44E52" if n in recommended_removal else "#4C72B0" for n in names]

    fig_height = max(4, len(names) * 0.35)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Feature Importance (Tree-Based)", fontsize=11)
    ax.set_title("Feature Importance — Removal Candidates in Red", fontsize=13, fontweight="bold")
    ax.axvline(x=IMPORTANCE_THRESHOLD, color="#999999", linestyle="--", linewidth=0.8, label=f"threshold = {IMPORTANCE_THRESHOLD}")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = save_and_show(fig, state, "feature_selection_importance.png")
    plt.close()
    return path
