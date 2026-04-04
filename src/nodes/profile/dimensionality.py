"""
Dimensionality reduction node.

Performs dimensionality reduction and generates visualizations to reveal
the intrinsic structure of numeric features:

  - PCA scree plot with cumulative variance and 80/90/95% thresholds
  - PCA biplot (2D scatter with feature loading arrows)
  - t-SNE 2D embedding (skipped if n_rows >= 10000)
  - UMAP 2D embedding (skipped if umap not installed)

Points are colored by the first categorical column (from classify) when
available.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")
CATEGORY_TYPES = {"categorical_nominal", "categorical_ordinal", "binary"}

TSNE_ROW_LIMIT = 10000


def dimensionality(state: dict) -> dict:
    """Run dimensionality reduction analyses and generate visualizations."""
    df = state["data"]
    images = []

    # ── Select numeric columns ───────────────────────────────────────────
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
        print_warning(f"Dropping constant columns: {constant_cols}")
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    if len(numeric_cols) < 3:
        print_warning("Fewer than 3 numeric columns — skipping dimensionality reduction.")
        state["nodes"]["dimensionality"] = {
            "status": "skipped",
            "reason": "fewer than 3 numeric columns",
            "images": [],
        }
        return state

    # ── Prepare clean numeric matrix ─────────────────────────────────────
    df_numeric = df[numeric_cols].dropna()

    if len(df_numeric) < 5:
        print_warning("Too few complete rows after dropping NaNs — skipping dimensionality reduction.")
        state["nodes"]["dimensionality"] = {
            "status": "skipped",
            "reason": "too few complete rows",
            "images": [],
        }
        return state

    print_info(f"Dimensionality reduction on {len(numeric_cols)} features, {len(df_numeric)} rows.")

    # StandardScale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    # ── Determine categorical color column ───────────────────────────────
    color_col = _get_color_column(state, df_numeric.index)

    # ── PCA ───────────────────────────────────────────────────────────────
    n_components = min(len(numeric_cols), 20)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Components needed for variance thresholds
    n_80 = int(np.searchsorted(cumulative, 0.80) + 1)
    n_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n_95 = int(np.searchsorted(cumulative, 0.95) + 1)

    # Cap at actual number of components
    n_80 = min(n_80, n_components)
    n_90 = min(n_90, n_components)
    n_95 = min(n_95, n_components)

    print_info(f"PCA: {n_80} components for 80%, {n_90} for 90%, {n_95} for 95% variance.")
    print_detail("explained variance (first 5)", [round(float(v), 4) for v in explained[:5]])

    # ── Scree plot ────────────────────────────────────────────────────────
    scree_img = _plot_scree(explained, cumulative, n_components, state)
    if scree_img:
        images.append(scree_img)

    # ── Biplot ────────────────────────────────────────────────────────────
    biplot_img = _plot_biplot(X_pca, pca, numeric_cols, explained, color_col, state)
    if biplot_img:
        images.append(biplot_img)

    # ── t-SNE ─────────────────────────────────────────────────────────────
    if len(df_numeric) >= TSNE_ROW_LIMIT:
        print_warning(
            f"Skipping t-SNE: {len(df_numeric)} rows exceeds limit of {TSNE_ROW_LIMIT}."
        )
    else:
        tsne_img = _plot_tsne(X_scaled, color_col, state)
        if tsne_img:
            images.append(tsne_img)

    # ── UMAP ──────────────────────────────────────────────────────────────
    umap_img = _plot_umap(X_scaled, color_col, state)
    if umap_img:
        images.append(umap_img)

    # ── Store results ─────────────────────────────────────────────────────
    state["nodes"]["dimensionality"] = {
        "status": "analyzed",
        "pca_explained_variance": [round(float(v), 6) for v in explained],
        "pca_cumulative_variance": [round(float(v), 6) for v in cumulative],
        "n_components_80": n_80,
        "n_components_90": n_90,
        "n_components_95": n_95,
        "n_features": len(numeric_cols),
        "n_rows_used": len(df_numeric),
        "constant_columns_dropped": constant_cols,
        "images": [str(p) for p in images],
    }

    # ── Narrative + report section ────────────────────────────────────────
    narrative = narrate("Dimensionality", {
        "n_features": len(numeric_cols),
        "n_components_80": n_80,
        "n_components_90": n_90,
        "n_components_95": n_95,
        "explained_variance_top3": [round(float(v), 4) for v in explained[:3]],
        "total_variance_explained": round(float(cumulative[-1]), 4),
    })
    add_section(state, "Dimensionality Reduction", narrative, images)

    return state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_color_column(state, valid_index):
    """
    Find the first categorical column from classify results to use for
    coloring scatter plots. Returns a Series aligned to valid_index, or None.
    """
    classifications = state.get("nodes", {}).get("classify", {}).get("classifications", {})
    if not classifications:
        return None

    df = state["data"]

    for col, info in classifications.items():
        col_type = info if isinstance(info, str) else info.get("type", "")
        if col_type in CATEGORY_TYPES:
            if col in df.columns:
                series = df.loc[valid_index, col]
                # Limit to top 10 categories for readability
                top_cats = series.value_counts().head(10).index
                series = series.where(series.isin(top_cats), other="Other")
                return series
    return None


# ── Visualizations ───────────────────────────────────────────────────────────


def _plot_scree(explained, cumulative, n_components, state):
    """Scree plot: bar chart of explained variance + cumulative line."""
    components = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    # Bar chart — individual explained variance
    ax1.bar(components, explained, color="#4C72B0", alpha=0.8,
            edgecolor="white", label="Individual")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance Ratio", fontsize=11, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(components)

    # Cumulative line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color="#C44E52", marker="o",
             markersize=5, linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance", fontsize=11, color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 1.05)

    # Threshold lines
    for threshold, label, ls in [
        (0.80, "80%", ":"),
        (0.90, "90%", "--"),
        (0.95, "95%", "-."),
    ]:
        ax2.axhline(y=threshold, color="#888888", linestyle=ls,
                     alpha=0.6, linewidth=1)
        ax2.text(n_components + 0.3, threshold, label,
                 fontsize=8, color="#888888", va="center")

    ax1.set_title("PCA Scree Plot", fontsize=13, fontweight="bold", pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    plt.tight_layout()
    path = save_and_show(fig, state, "dim_scree_plot.png")
    plt.close()
    return path


def _plot_biplot(X_pca, pca, feature_names, explained, color_col, state):
    """2D scatter of PC1 vs PC2 with feature loading arrows."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")

    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]

    # Scatter points
    if color_col is not None:
        categories = color_col.unique()
        palette = sns.color_palette("Set2", n_colors=len(categories))
        for i, cat in enumerate(categories):
            mask = color_col.values == cat
            ax.scatter(pc1[mask], pc2[mask], c=[palette[i]], label=str(cat),
                       s=20, alpha=0.6, edgecolors="none")
        ax.legend(title=color_col.name, fontsize=8, title_fontsize=9,
                  loc="best", markerscale=1.5)
    else:
        ax.scatter(pc1, pc2, c="#4C72B0", s=20, alpha=0.5, edgecolors="none")

    # Feature loading arrows
    loadings = pca.components_[:2, :]  # shape (2, n_features)

    # Scale arrows to be visible relative to the scatter
    scale = max(np.abs(pc1).max(), np.abs(pc2).max()) * 0.8
    max_loading = np.max(np.sqrt(loadings[0] ** 2 + loadings[1] ** 2))
    if max_loading > 0:
        arrow_scale = scale / max_loading
    else:
        arrow_scale = 1.0

    for j, feature in enumerate(feature_names):
        lx = loadings[0, j] * arrow_scale
        ly = loadings[1, j] * arrow_scale
        ax.annotate(
            "",
            xy=(lx, ly),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5, alpha=0.7),
        )
        ax.text(lx * 1.08, ly * 1.08, feature, fontsize=7, color="#e74c3c",
                ha="center", va="center", alpha=0.9)

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA Biplot", fontsize=13, fontweight="bold", pad=12)
    ax.axhline(0, color="#cccccc", linewidth=0.5)
    ax.axvline(0, color="#cccccc", linewidth=0.5)

    plt.tight_layout()
    path = save_and_show(fig, state, "dim_pca_biplot.png")
    plt.close()
    return path


def _plot_tsne(X_scaled, color_col, state):
    """2D t-SNE scatter plot."""
    print_info("Running t-SNE (this may take a moment)...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    if color_col is not None:
        categories = color_col.unique()
        palette = sns.color_palette("Set2", n_colors=len(categories))
        for i, cat in enumerate(categories):
            mask = color_col.values == cat
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[palette[i]],
                       label=str(cat), s=20, alpha=0.6, edgecolors="none")
        ax.legend(title=color_col.name, fontsize=8, title_fontsize=9,
                  loc="best", markerscale=1.5)
    else:
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c="#4C72B0", s=20,
                   alpha=0.5, edgecolors="none")

    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_title("t-SNE Embedding (perplexity=30)", fontsize=13,
                 fontweight="bold", pad=12)

    plt.tight_layout()
    path = save_and_show(fig, state, "dim_tsne.png")
    plt.close()
    return path


def _plot_umap(X_scaled, color_col, state):
    """2D UMAP scatter plot. Returns None if umap is not installed."""
    try:
        import umap
    except ImportError:
        print_warning("umap-learn not installed — skipping UMAP visualization.")
        return None

    print_info("Running UMAP...")

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")

    if color_col is not None:
        categories = color_col.unique()
        palette = sns.color_palette("Set2", n_colors=len(categories))
        for i, cat in enumerate(categories):
            mask = color_col.values == cat
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[palette[i]],
                       label=str(cat), s=20, alpha=0.6, edgecolors="none")
        ax.legend(title=color_col.name, fontsize=8, title_fontsize=9,
                  loc="best", markerscale=1.5)
    else:
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c="#4C72B0", s=20,
                   alpha=0.5, edgecolors="none")

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title("UMAP Embedding", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = save_and_show(fig, state, "dim_umap.png")
    plt.close()
    return path
