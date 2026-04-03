"""
Outlier detection node.

For each numeric column, applies two methods:
  - IQR fence  (1.5 × IQR beyond Q1/Q3)
  - Z-score    (|z| > 3)

Reports the union of both, the count/pct of outliers, and the
boundary values. Generates a box plot grid and individual strip
plots for columns with notable outliers.

Results feed into synthesis so the LLM can decide whether to cap,
transform, or leave outliers as-is.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail

NUMERIC_DTYPES = ("int8", "int16", "int32", "int64", "float32", "float64")
ZSCORE_THRESHOLD = 3.0
IQR_MULTIPLIER   = 1.5
# Only flag a column as "notable" if >0.5% of rows are outliers
NOTABLE_THRESHOLD = 0.005


def outliers(state: dict) -> dict:
    """Detect outliers via IQR and Z-score for all numeric columns."""
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

        # Union of both methods
        union_mask    = iqr_mask | z_mask
        n_outliers    = int(union_mask.sum())
        outlier_pct   = round(n_outliers / len(series), 4)

        result = {
            "n_outliers":       n_outliers,
            "outlier_pct":      outlier_pct,
            "iqr_count":        int(iqr_mask.sum()),
            "zscore_count":     int(z_mask.sum()),
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
            flag = f"  [yellow]⚠ {n_outliers} outliers ({outlier_pct:.1%})[/yellow]"
        else:
            flag = f"  {n_outliers} outliers"

        print_info(
            f"{col}: IQR=[{lower:.2f}, {upper:.2f}]  "
            f"z>{ZSCORE_THRESHOLD}={int(z_mask.sum())}{flag}"
        )

    # ── Box plot grid for all numeric cols ────────────────────────────────────
    if numeric_cols:
        grid_img = _plot_boxplot_grid(df, numeric_cols, results, state)
        if grid_img:
            images.append(grid_img)

    # ── Strip plots for notable columns ──────────────────────────────────────
    for col in notable[:8]:   # cap at 8 individual plots
        img = _plot_outlier_strip(df, col, results[col], state)
        if img:
            images.append(img)

    print_detail("columns checked", len(results))
    print_detail("notable outlier columns", len(notable))

    state["nodes"]["outliers"] = {
        "status": "analyzed",
        "results": results,
        "notable_columns": notable,
        "images": [str(p) for p in images],
    }

    narrative = narrate("Outlier Detection", {
        "summary": {
            col: {
                "n_outliers": r["n_outliers"],
                "outlier_pct": r["outlier_pct"],
                "iqr_bounds": [r["iqr_lower"], r["iqr_upper"]],
                "extreme_high": r["extreme_high"][:3],
                "extreme_low":  r["extreme_low"][:3],
            }
            for col, r in results.items() if r["n_outliers"] > 0
        },
        "notable_columns": notable,
    })
    add_section(state, "Outlier Detection", narrative, images)

    return state


# ── Box plot grid ─────────────────────────────────────────────────────────────

def _plot_boxplot_grid(df, numeric_cols, results, state):
    """One box plot per numeric column, coloured red if notable."""
    cols_to_plot = numeric_cols[:16]
    n    = len(cols_to_plot)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 3))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols_to_plot):
        ax     = axes_flat[i]
        series = df[col].dropna().astype(float)
        res    = results.get(col, {})
        color  = "#C44E52" if res.get("outlier_pct", 0) >= NOTABLE_THRESHOLD else "#4C72B0"

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

    plt.suptitle("Outlier Overview — Box Plots", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "outliers_boxplot_grid.png")
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
        f"Outliers: {col}  —  {n_out} flagged ({pct:.1%})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = save_and_show(fig, state, f"outliers_{col.lower()}.png")
    plt.close()
    return path
