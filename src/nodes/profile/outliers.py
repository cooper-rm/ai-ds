"""
Outlier detection node.

For each numeric column, applies per-column methods:
  - IQR fence  (1.5 × IQR beyond Q1/Q3)
  - Z-score    (|z| > 3)
  - Modified Z-score (MAD-based, |mz| > 3.5)

Then runs multivariate methods on all numeric columns together:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - DBSCAN noise detection

Reports the union of per-column methods, the count/pct of outliers, and the
boundary values. Generates a box plot grid (color-coded by method agreement),
a method agreement heatmap, and PCA scatter with outlier highlighting.

Results feed into synthesis so the LLM can decide whether to cap,
transform, or leave outliers as-is.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")
ZSCORE_THRESHOLD = 3.0
IQR_MULTIPLIER   = 1.5
MODIFIED_Z_THRESHOLD = 3.5
# Only flag a column as "notable" if >0.5% of rows are outliers
NOTABLE_THRESHOLD = 0.005


def _modified_zscore(series: pd.Series) -> np.ndarray:
    """Compute Modified Z-scores using Median Absolute Deviation."""
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))
    return 0.6745 * (series.values - median) / mad


def _estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int = 5) -> float:
    """
    Estimate DBSCAN eps using the knee of the k-distance graph.
    Falls back to a percentile-based heuristic if the knee is unclear.
    """
    k = min(min_samples, X_scaled.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    # Simple knee detection: find the point of maximum curvature
    n = len(k_distances)
    if n < 3:
        return float(np.median(k_distances))

    # Normalize to [0, 1] for curvature calculation
    x_norm = np.linspace(0, 1, n)
    y_norm = (k_distances - k_distances[0]) / (k_distances[-1] - k_distances[0] + 1e-10)

    # Vector from first to last point
    line_vec = np.array([1.0, y_norm[-1] - y_norm[0]])
    line_len = np.linalg.norm(line_vec)

    # Distance of each point from the line
    dists = np.abs(
        (y_norm[-1] - y_norm[0]) * x_norm - y_norm + y_norm[0]
    ) / (line_len + 1e-10)

    knee_idx = np.argmax(dists)
    eps = float(k_distances[knee_idx])
    return max(eps, 1e-5)  # guard against zero


def outliers(state: dict) -> dict:
    """Detect outliers via IQR, Z-score, Modified Z-score, and multivariate methods."""
    df              = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})

    results  = {}
    notable  = []
    images   = []

    numeric_cols = [
        col for col in df.columns
        if str(df[col].dtype) in NUMERIC_DTYPES
        and classifications.get(col, {}).get("type") not in ("binary", "identifier")
    ]

    # ── Per-column outlier detection ─────────────────────────────────────────
    for col in numeric_cols:
        series = df[col].dropna().astype(float)
        if len(series) < 10:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        lower  = q1 - IQR_MULTIPLIER * iqr
        upper  = q3 + IQR_MULTIPLIER * iqr

        iqr_mask = (series < lower) | (series > upper)
        z_scores = np.abs(stats.zscore(series))
        z_mask   = z_scores > ZSCORE_THRESHOLD

        # Modified Z-score (MAD-based)
        mod_z = _modified_zscore(series)
        mod_z_mask = np.abs(mod_z) > MODIFIED_Z_THRESHOLD

        # Union of all per-column methods
        union_mask    = iqr_mask | z_mask | mod_z_mask
        n_outliers    = int(union_mask.sum())
        outlier_pct   = round(n_outliers / len(series), 4)

        result = {
            "n_outliers":       n_outliers,
            "outlier_pct":      outlier_pct,
            "iqr_count":        int(iqr_mask.sum()),
            "zscore_count":     int(z_mask.sum()),
            "modified_z_count": int(mod_z_mask.sum()),
            "iqr_lower":        round(float(lower), 4),
            "iqr_upper":        round(float(upper), 4),
            "min":              round(float(series.min()), 4),
            "max":              round(float(series.max()), 4),
            "extreme_low":      sorted([round(float(v), 4) for v in series[series < lower]])[:5],
            "extreme_high":     sorted([round(float(v), 4) for v in series[series > upper]], reverse=True)[:5],
        }
        results[col] = result

        if outlier_pct >= NOTABLE_THRESHOLD:
            notable.append(col)
            flag = f"  [yellow]\u26a0 {n_outliers} outliers ({outlier_pct:.1%})[/yellow]"
        else:
            flag = f"  {n_outliers} outliers"

        print_info(
            f"{col}: IQR=[{lower:.2f}, {upper:.2f}]  "
            f"z>{ZSCORE_THRESHOLD}={int(z_mask.sum())}  "
            f"MAD={int(mod_z_mask.sum())}{flag}"
        )

    # ── Multivariate outlier detection ───────────────────────────────────────
    # Work on columns that survived per-column analysis
    mv_cols = [c for c in numeric_cols if c in results]
    multivariate = {}

    if len(mv_cols) >= 2:
        # Build a clean numeric matrix (drop rows with any NaN in these cols)
        df_num = df[mv_cols].dropna().astype(float)
        valid_idx = df_num.index

        if len(df_num) >= 20:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_num)

            # --- Isolation Forest ---
            try:
                iso = IsolationForest(contamination="auto", random_state=42)
                iso_preds = iso.fit_predict(X_scaled)
                iso_outliers = (iso_preds == -1)
                iso_count = int(iso_outliers.sum())
                multivariate["isolation_forest"] = {
                    "outlier_count": iso_count,
                    "outlier_pct": round(iso_count / len(df_num), 4),
                    "outlier_indices": list(valid_idx[iso_outliers]),
                }
                print_info(
                    f"Isolation Forest: {iso_count} outliers "
                    f"({iso_count / len(df_num):.1%}) across {len(mv_cols)} features"
                )
            except Exception as e:
                print_info(f"Isolation Forest skipped: {e}")
                iso_outliers = np.zeros(len(df_num), dtype=bool)

            # --- Local Outlier Factor ---
            try:
                n_neighbors = min(20, len(df_num) - 1)
                lof = LocalOutlierFactor(n_neighbors=n_neighbors)
                lof_preds = lof.fit_predict(X_scaled)
                lof_outliers = (lof_preds == -1)
                lof_count = int(lof_outliers.sum())
                multivariate["lof"] = {
                    "outlier_count": lof_count,
                    "outlier_pct": round(lof_count / len(df_num), 4),
                    "outlier_indices": list(valid_idx[lof_outliers]),
                }
                print_info(
                    f"LOF (k={n_neighbors}): {lof_count} outliers "
                    f"({lof_count / len(df_num):.1%})"
                )
            except Exception as e:
                print_info(f"LOF skipped: {e}")
                lof_outliers = np.zeros(len(df_num), dtype=bool)

            # --- DBSCAN noise points ---
            try:
                min_samples = min(5, len(df_num) - 1)
                eps = _estimate_dbscan_eps(X_scaled, min_samples=min_samples)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                db_labels = dbscan.fit_predict(X_scaled)
                db_outliers = (db_labels == -1)
                db_count = int(db_outliers.sum())
                n_clusters = len(set(db_labels) - {-1})
                multivariate["dbscan"] = {
                    "outlier_count": db_count,
                    "outlier_pct": round(db_count / len(df_num), 4),
                    "outlier_indices": list(valid_idx[db_outliers]),
                    "eps": round(eps, 4),
                    "n_clusters": n_clusters,
                }
                print_info(
                    f"DBSCAN (eps={eps:.3f}): {db_count} noise points "
                    f"({db_count / len(df_num):.1%}), {n_clusters} clusters"
                )
            except Exception as e:
                print_info(f"DBSCAN skipped: {e}")
                db_outliers = np.zeros(len(df_num), dtype=bool)

            # --- Build per-row method agreement counts ---
            # Per-column methods: a row is flagged if ANY column flags it
            iqr_row_flags = np.zeros(len(df_num), dtype=bool)
            z_row_flags = np.zeros(len(df_num), dtype=bool)
            modz_row_flags = np.zeros(len(df_num), dtype=bool)

            for j, col in enumerate(mv_cols):
                col_series = df_num[col]
                r = results[col]
                iqr_row_flags |= ((col_series < r["iqr_lower"]) | (col_series > r["iqr_upper"])).values
                z_row_flags |= (np.abs(stats.zscore(col_series)) > ZSCORE_THRESHOLD)
                mz = _modified_zscore(col_series)
                modz_row_flags |= (np.abs(mz) > MODIFIED_Z_THRESHOLD)

            # Agreement matrix: rows x methods
            method_flags = np.column_stack([
                iqr_row_flags,
                z_row_flags,
                modz_row_flags,
                iso_outliers,
                lof_outliers,
                db_outliers,
            ])
            agreement_counts = method_flags.sum(axis=1)  # how many methods flag each row

            multivariate["method_names"] = [
                "IQR", "Z-score", "Modified Z", "IsoForest", "LOF", "DBSCAN"
            ]
            multivariate["agreement_counts"] = agreement_counts.tolist()
            multivariate["valid_index"] = list(valid_idx)

            # Per-column x method outlier count matrix (for heatmap)
            col_method_counts = {}
            for col in mv_cols:
                col_series = df_num[col]
                r = results[col]
                iqr_c = int(((col_series < r["iqr_lower"]) | (col_series > r["iqr_upper"])).sum())
                z_c = int((np.abs(stats.zscore(col_series)) > ZSCORE_THRESHOLD).sum())
                mz_c = int((np.abs(_modified_zscore(col_series)) > MODIFIED_Z_THRESHOLD).sum())
                # Multivariate methods don't have per-column counts, use total
                col_method_counts[col] = {
                    "IQR": iqr_c,
                    "Z-score": z_c,
                    "Modified Z": mz_c,
                    "IsoForest": int(iso_outliers.sum()),
                    "LOF": int(lof_outliers.sum()),
                    "DBSCAN": int(db_outliers.sum()),
                }
            multivariate["col_method_counts"] = col_method_counts

    # ── Box plot grid (color-coded by method agreement) ──────────────────────
    if numeric_cols:
        grid_img = _plot_boxplot_grid(df, numeric_cols, results, multivariate, state)
        if grid_img:
            images.append(grid_img)

    # ── Method agreement heatmap + PCA scatter ───────────────────────────────
    if multivariate and len(mv_cols) >= 2:
        summary_img = _plot_method_summary(df, mv_cols, multivariate, state)
        if summary_img:
            images.append(summary_img)

    # ── Strip plots for notable columns ──────────────────────────────────────
    for col in notable[:8]:   # cap at 8 individual plots
        img = _plot_outlier_strip(df, col, results[col], state)
        if img:
            images.append(img)

    print_detail("columns checked", len(results))
    print_detail("notable outlier columns", len(notable))
    if multivariate:
        print_detail("multivariate methods run", len([
            k for k in ("isolation_forest", "lof", "dbscan")
            if k in multivariate
        ]))

    state["nodes"]["outliers"] = {
        "status": "analyzed",
        "results": results,
        "notable_columns": notable,
        "multivariate": multivariate,
        "images": [str(p) for p in images],
    }

    narrative = narrate("Outlier Detection", {
        "summary": {
            col: {
                "n_outliers": r["n_outliers"],
                "outlier_pct": r["outlier_pct"],
                "iqr_bounds": [r["iqr_lower"], r["iqr_upper"]],
                "modified_z_count": r["modified_z_count"],
                "extreme_high": r["extreme_high"][:3],
                "extreme_low":  r["extreme_low"][:3],
            }
            for col, r in results.items() if r["n_outliers"] > 0
        },
        "notable_columns": notable,
        "multivariate": {
            k: {"outlier_count": v["outlier_count"], "outlier_pct": v["outlier_pct"]}
            for k, v in multivariate.items()
            if isinstance(v, dict) and "outlier_count" in v
        },
    })
    add_section(state, "Outlier Detection", narrative, images)

    return state


# ── Box plot grid ─────────────────────────────────────────────────────────────

def _plot_boxplot_grid(df, numeric_cols, results, multivariate, state):
    """
    One box plot per numeric column.
    Color-coded by method agreement:
      red    = flagged by 3+ methods
      orange = flagged by 2 methods
      blue   = normal / 0-1 methods
    """
    cols_to_plot = numeric_cols[:16]
    n    = len(cols_to_plot)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    # Precompute per-row agreement if multivariate data is available
    agreement_map = {}  # index -> count
    if multivariate and "agreement_counts" in multivariate:
        valid_idx = multivariate["valid_index"]
        counts = multivariate["agreement_counts"]
        agreement_map = dict(zip(valid_idx, counts))

    for i, col in enumerate(cols_to_plot):
        ax     = axes_flat[i]
        series = df[col].dropna().astype(float)
        res    = results.get(col, {})

        # Determine box color based on max agreement for this column's outliers
        if agreement_map and col in results:
            lower_b = res.get("iqr_lower", -np.inf)
            upper_b = res.get("iqr_upper", np.inf)
            outlier_idx = series[(series < lower_b) | (series > upper_b)].index
            max_agree = 0
            for idx in outlier_idx:
                max_agree = max(max_agree, agreement_map.get(idx, 0))
            if max_agree >= 3:
                color = "#C44E52"  # red
            elif max_agree >= 2:
                color = "#E67E22"  # orange
            else:
                color = "#4C72B0"  # blue
        else:
            color = "#C44E52" if res.get("outlier_pct", 0) >= NOTABLE_THRESHOLD else "#4C72B0"

        bp = ax.boxplot(series, patch_artist=True, widths=0.5,
                        medianprops={"color": "white", "linewidth": 2},
                        flierprops={"marker": "o", "markerfacecolor": "#e74c3c",
                                    "markersize": 3, "alpha": 0.5})
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)

        n_out = res.get("n_outliers", 0)
        pct   = res.get("outlier_pct", 0)
        ax.set_title(f"{col}\n{n_out} outliers ({pct:.1%})",
                     fontsize=8, fontweight="bold",
                     color="#c0392b" if pct >= NOTABLE_THRESHOLD else "black")
        ax.set_xticks([])
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Outlier Overview \u2014 Box Plots (red=3+ methods, orange=2, blue=normal)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "outliers_boxplot_grid.png")
    plt.close()
    return path


# ── Method agreement heatmap + PCA scatter ───────────────────────────────────

def _plot_method_summary(df, mv_cols, multivariate, state):
    """
    Two-panel summary figure:
      Left:  method agreement heatmap (columns x methods, colored by outlier count)
      Right: PCA scatter (first two components) with points colored by agreement count
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    method_names = multivariate.get("method_names", [])
    col_method_counts = multivariate.get("col_method_counts", {})

    # --- Panel 1: Heatmap ---
    ax_heat = axes[0]
    heatmap_cols = [c for c in mv_cols if c in col_method_counts][:20]  # cap for readability

    if heatmap_cols and method_names:
        matrix = np.array([
            [col_method_counts[col].get(m, 0) for m in method_names]
            for col in heatmap_cols
        ], dtype=float)

        im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax_heat.set_xticks(range(len(method_names)))
        ax_heat.set_xticklabels(method_names, rotation=45, ha="right", fontsize=9)
        ax_heat.set_yticks(range(len(heatmap_cols)))
        ax_heat.set_yticklabels(heatmap_cols, fontsize=9)

        # Annotate cells with counts
        for row in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                val = int(matrix[row, col_idx])
                text_color = "white" if val > matrix.max() * 0.6 else "black"
                ax_heat.text(col_idx, row, str(val), ha="center", va="center",
                             fontsize=8, color=text_color)

        fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Outlier Count")
        ax_heat.set_title("Method Agreement Heatmap\n(columns \u00d7 methods)",
                          fontsize=12, fontweight="bold")
    else:
        ax_heat.text(0.5, 0.5, "Insufficient data for heatmap",
                     ha="center", va="center", transform=ax_heat.transAxes)
        ax_heat.set_title("Method Agreement Heatmap", fontsize=12, fontweight="bold")

    # --- Panel 2: PCA scatter colored by agreement ---
    ax_pca = axes[1]
    agreement_counts = np.array(multivariate.get("agreement_counts", []))
    valid_idx = multivariate.get("valid_index", [])

    if len(mv_cols) >= 2 and len(valid_idx) > 0:
        df_num = df.loc[valid_idx, mv_cols].astype(float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num)

        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        if n_components == 2:
            # Color by agreement count
            max_agree = max(agreement_counts.max(), 1) if len(agreement_counts) > 0 else 1
            cmap = LinearSegmentedColormap.from_list(
                "agreement", ["#4C72B0", "#E67E22", "#C44E52"], N=256
            )

            scatter = ax_pca.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=agreement_counts, cmap=cmap, vmin=0, vmax=max(max_agree, 3),
                s=12, alpha=0.6, edgecolors="none"
            )
            fig.colorbar(scatter, ax=ax_pca, shrink=0.8, label="Methods Flagging")

            var_explained = pca.explained_variance_ratio_
            ax_pca.set_xlabel(f"PC1 ({var_explained[0]:.1%} variance)", fontsize=10)
            ax_pca.set_ylabel(f"PC2 ({var_explained[1]:.1%} variance)", fontsize=10)
        else:
            ax_pca.scatter(X_pca[:, 0], np.zeros(len(X_pca)),
                           c=agreement_counts, cmap="YlOrRd", s=12, alpha=0.6)
            ax_pca.set_xlabel("PC1", fontsize=10)

        ax_pca.set_title("PCA Projection \u2014 Outlier Agreement",
                         fontsize=12, fontweight="bold")
    else:
        ax_pca.text(0.5, 0.5, "Insufficient features for PCA",
                    ha="center", va="center", transform=ax_pca.transAxes)
        ax_pca.set_title("PCA Projection", fontsize=12, fontweight="bold")

    plt.suptitle("Outlier Detection \u2014 Method Summary",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_and_show(fig, state, "outliers_method_summary.png")
    plt.close()
    return path


# ── Strip plot ────────────────────────────────────────────────────────────────

def _plot_outlier_strip(df, col, result, state):
    """
    Two-panel plot for a notable outlier column:
      Left:  strip/jitter plot — outliers highlighted in red
      Right: histogram with IQR fence lines marked
    """
    series = df[col].dropna().astype(float)
    lower  = result["iqr_lower"]
    upper  = result["iqr_upper"]

    is_outlier = (series < lower) | (series > upper)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("white")

    # Strip plot
    jitter = np.random.uniform(-0.2, 0.2, size=len(series))
    axes[0].scatter(
        jitter[~is_outlier], series[~is_outlier],
        color="#4C72B0", alpha=0.4, s=8, label="normal"
    )
    axes[0].scatter(
        jitter[is_outlier], series[is_outlier],
        color="#C44E52", alpha=0.8, s=20, zorder=5, label="outlier"
    )
    axes[0].axhline(lower, color="#e67e22", linewidth=1.5,
                    linestyle="--", label=f"IQR lower ({lower:.2f})")
    axes[0].axhline(upper, color="#e67e22", linewidth=1.5,
                    linestyle="--", label=f"IQR upper ({upper:.2f})")
    axes[0].set_xticks([])
    axes[0].set_ylabel(col)
    axes[0].set_title(f"Strip Plot: {col}", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)

    # Histogram with fence lines
    bins = min(50, max(10, len(series) // 10))
    axes[1].hist(series[~is_outlier], bins=bins, color="#4C72B0",
                 alpha=0.7, label="normal", density=True)
    axes[1].hist(series[is_outlier],  bins=bins, color="#C44E52",
                 alpha=0.8, label="outlier", density=True)
    axes[1].axvline(lower, color="#e67e22", linewidth=1.5, linestyle="--")
    axes[1].axvline(upper, color="#e67e22", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Distribution with IQR Fences", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)

    n_out = result["n_outliers"]
    pct   = result["outlier_pct"]
    plt.suptitle(
        f"Outliers: {col}  \u2014  {n_out} flagged ({pct:.1%})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = save_and_show(fig, state, f"outliers_{col.lower()}.png")
    plt.close()
    return path
