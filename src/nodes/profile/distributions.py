"""
Distributions node.

For every column, generates an appropriate visualization based on its
analytical type (from classify):

  continuous / discrete  → histogram + KDE + box plot
  categorical_nominal /
  categorical_ordinal /
  binary                 → bar chart (value counts)
  datetime               → time series line
  identifier / text      → skip

Also generates a grid summary image of all numeric distributions,
and records skewness + kurtosis for each numeric column.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning

# Types that get numeric treatment
NUMERIC_TYPES  = {"continuous", "discrete"}
CATEGORY_TYPES = {"categorical_nominal", "categorical_ordinal", "binary"}
SKIP_TYPES     = {"identifier", "text"}


def distributions(state: dict) -> dict:
    """Plot per-column distributions and record shape statistics."""
    df             = state["data"]
    classifications = state["nodes"].get("classify", {}).get("classifications", {})

    per_col  = {}
    images   = []
    skipped  = []

    numeric_cols    = []
    categorical_cols = []

    for col in df.columns:
        col_type = classifications.get(col, {}).get("type", "unknown")

        if col_type in SKIP_TYPES:
            skipped.append(col)
            continue

        if col_type in NUMERIC_TYPES or (
            col_type == "unknown"
            and df[col].dtype in ("int8","int16","int32","int64","float32","float64")
        ):
            numeric_cols.append((col, col_type))
        elif col_type in CATEGORY_TYPES or (
            col_type == "unknown"
            and df[col].dtype not in ("int8","int16","int32","int64","float32","float64")
        ):
            categorical_cols.append((col, col_type))
        elif col_type == "datetime":
            img = _plot_datetime(df, col, state)
            if img:
                images.append(img)
                per_col[col] = {"type": col_type, "plot": str(img)}
        else:
            skipped.append(col)

    # ── Numeric columns ───────────────────────────────────────────────────────
    for col, col_type in numeric_cols:
        series = df[col].dropna()
        if len(series) < 3:
            skipped.append(col)
            continue

        skewness  = round(float(series.skew()), 4)
        kurtosis  = round(float(series.kurtosis()), 4)
        shape_tag = _shape_tag(skewness)

        # ── Normality test battery ────────────────────────────────────────
        normality_results = _normality_tests(series)
        is_normal = normality_results["is_normal"]

        # ── Distribution fitting (KS goodness-of-fit) ────────────────────
        best_dist, best_ks = _best_fit_distribution(series)

        img = _plot_numeric(df, col, col_type, skewness, kurtosis, state)
        per_col[col] = {
            "type": col_type,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "shape": shape_tag,
            "normality": normality_results,
            "best_fit": {"distribution": best_dist, "ks_statistic": best_ks},
            "plot": str(img) if img else None,
        }
        if img:
            images.append(img)
        print_info(f"{col}: skew={skewness:+.2f}  kurtosis={kurtosis:.2f}  [{shape_tag}]")
        print_info(f"{col}: normality={'yes' if is_normal else 'no'}  best_fit={best_dist}")

    # ── Categorical columns ───────────────────────────────────────────────────
    rare_categories = {}

    for col, col_type in categorical_cols:
        img = _plot_categorical(df, col, col_type, state)
        n_unique = int(df[col].nunique())

        # Rare category detection: freq < 1% or count < 10
        counts = df[col].value_counts()
        total  = len(df[col].dropna())
        rare   = []
        if total > 0:
            for val, cnt in counts.items():
                freq = cnt / total
                if freq < 0.01 or cnt < 10:
                    rare.append({"value": str(val), "count": int(cnt),
                                 "freq": round(freq, 4)})
        if rare:
            rare_categories[col] = rare
            print_warning(
                f"{col}: {len(rare)} rare categories "
                f"(<1% or n<10) out of {n_unique}"
            )

        per_col[col] = {
            "type": col_type,
            "n_unique": n_unique,
            "rare_count": len(rare),
            "plot": str(img) if img else None,
        }
        if img:
            images.append(img)
        print_info(f"{col}: {n_unique} categories  [{col_type}]")

    # ── Numeric grid summary ──────────────────────────────────────────────────
    if numeric_cols:
        grid_img = _plot_numeric_grid([c for c, _ in numeric_cols], df, state)
        if grid_img:
            images.insert(0, grid_img)   # put grid first

    print_detail("plotted",  len(per_col))
    print_detail("skipped",  len(skipped))

    state["nodes"]["distributions"] = {
        "status": "analyzed",
        "per_column": per_col,
        "rare_categories": rare_categories,
        "skipped": skipped,
        "images": [str(p) for p in images],
    }

    narrative = narrate("Distributions", {
        "numeric_summary": {
            col: {
                "skewness": v["skewness"],
                "shape": v["shape"],
                "is_normal": v.get("normality", {}).get("is_normal"),
                "best_fit": v.get("best_fit", {}).get("distribution"),
                "best_fit_ks": v.get("best_fit", {}).get("ks_statistic"),
            }
            for col, v in per_col.items()
            if "skewness" in v
        },
        "categorical_summary": {
            col: {"n_unique": v["n_unique"], "rare_count": v.get("rare_count", 0)}
            for col, v in per_col.items()
            if "n_unique" in v
        },
        "rare_category_columns": list(rare_categories.keys()),
    })
    add_section(state, "Distributions", narrative, images)

    return state


# ── Numeric plot ──────────────────────────────────────────────────────────────

def _plot_numeric(df, col, col_type, skewness, kurtosis, state):
    """Histogram + KDE left panel, Q-Q middle panel, box plot right panel."""
    series = df[col].dropna().astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                             gridspec_kw={"width_ratios": [3, 2, 1]})
    fig.patch.set_facecolor("white")

    bins = _nice_bins(series)

    # Histogram
    axes[0].hist(series, bins=bins, color="#4C72B0", alpha=0.75,
                 edgecolor="white", density=True)

    # KDE overlay
    if len(series) > 5:
        kde_x = np.linspace(series.min(), series.max(), 300)
        kde   = stats.gaussian_kde(series)
        axes[0].plot(kde_x, kde(kde_x), color="#C44E52", linewidth=2, label="KDE")

    # Normal reference line
    mu, sigma = series.mean(), series.std()
    if sigma > 0:
        norm_x = np.linspace(series.min(), series.max(), 300)
        norm_y = stats.norm.pdf(norm_x, mu, sigma)
        axes[0].plot(norm_x, norm_y, color="#2ecc71", linewidth=1.5,
                     linestyle="--", alpha=0.8, label="Normal ref")

    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    skew_dir = "right-skewed" if skewness > 0.5 else "left-skewed" if skewness < -0.5 else "symmetric"
    axes[0].set_title(
        f"{col}  |  skew={skewness:+.2f}  kurtosis={kurtosis:.2f}  [{skew_dir}]",
        fontsize=11, fontweight="bold"
    )

    # Q-Q plot
    stats.probplot(series, dist="norm", plot=axes[1])
    # Add a reference diagonal line (the probplot adds one, but reinforce it)
    axes[1].get_lines()[1].set_color("#C44E52")
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].get_lines()[0].set_markerfacecolor("#4C72B0")
    axes[1].get_lines()[0].set_markeredgecolor("#4C72B0")
    axes[1].get_lines()[0].set_markersize(3)
    axes[1].set_title("Q-Q Plot (Normal)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")

    # Box plot
    bp = axes[2].boxplot(series, patch_artist=True, widths=0.5,
                         medianprops={"color": "#C44E52", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][0].set_alpha(0.6)
    axes[2].set_xticks([])
    axes[2].set_ylabel(col)
    axes[2].set_title("Box plot", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = save_and_show(fig, state, f"dist_{col.lower()}.png")
    plt.close()
    return path


# ── Categorical plot ──────────────────────────────────────────────────────────

def _plot_categorical(df, col, col_type, state):
    """Horizontal bar chart of value counts, top 20."""
    counts = df[col].value_counts().head(20)
    if len(counts) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.35)))
    fig.patch.set_facecolor("white")

    colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(counts)))
    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(v) for v in counts.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"{col}  [{col_type}]  —  {df[col].nunique()} unique values",
                 fontsize=11, fontweight="bold")

    # Value labels on bars
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / len(df)
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({pct:.1f}%)", va="center", fontsize=8, color="#555555")

    plt.tight_layout()
    path = save_and_show(fig, state, f"dist_{col.lower()}.png")
    plt.close()
    return path


# ── Datetime plot ─────────────────────────────────────────────────────────────

def _plot_datetime(df, col, state):
    """Line plot of value counts over time."""
    try:
        import pandas as pd
        series = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(series) < 3:
            return None

        counts = series.dt.to_period("M").value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("white")

        ax.plot(range(len(counts)), counts.values, color="#4C72B0",
                linewidth=1.5, marker="o", markersize=4)
        ax.set_xticks(range(0, len(counts), max(1, len(counts)//10)))
        ax.set_xticklabels(
            [str(counts.index[i]) for i in range(0, len(counts), max(1, len(counts)//10))],
            rotation=30, ha="right", fontsize=8
        )
        ax.set_title(f"{col}  [datetime]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        plt.tight_layout()
        path = save_and_show(fig, state, f"dist_{col.lower()}.png")
        plt.close()
        return path
    except Exception:
        return None


# ── Numeric grid ──────────────────────────────────────────────────────────────

def _plot_numeric_grid(numeric_cols, df, state):
    """
    Small-multiple grid of all numeric distributions — one histogram per cell.
    Max 16 columns in the grid.
    """
    cols_to_plot = numeric_cols[:16]
    n = len(cols_to_plot)
    if n == 0:
        return None

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 2.8))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols_to_plot):
        ax  = axes_flat[i]
        series = df[col].dropna().astype(float)
        bins = _nice_bins(series)
        ax.hist(series, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white")

        # KDE
        if len(series) > 5:
            kde_x = np.linspace(series.min(), series.max(), 200)
            kde   = stats.gaussian_kde(series)
            ax2   = ax.twinx()
            ax2.plot(kde_x, kde(kde_x), color="#C44E52", linewidth=1.2)
            ax2.set_yticks([])

        skew = series.skew()
        ax.set_title(f"{col}\nskew={skew:+.2f}", fontsize=8, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)

    # Hide unused cells
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Numeric Distributions — Overview", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = save_and_show(fig, state, "distributions_grid.png")
    plt.close()
    return path


# ── Normality testing ────────────────────────────────────────────────────

def _normality_tests(series):
    """
    Run a battery of normality tests and return a summary dict.
    Tests: Shapiro-Wilk, Anderson-Darling, Jarque-Bera, D'Agostino-Pearson.
    """
    results = {}
    normal_votes = 0
    total_votes  = 0

    # Shapiro-Wilk (best for n < 5000; scipy caps at 5000 internally)
    try:
        sw_stat, sw_p = stats.shapiro(series[:5000])
        results["shapiro_wilk"] = {
            "statistic": round(float(sw_stat), 6),
            "p_value": round(float(sw_p), 6),
        }
        total_votes += 1
        if sw_p > 0.05:
            normal_votes += 1
    except Exception:
        results["shapiro_wilk"] = {"error": "could not compute"}

    # Anderson-Darling
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ad_result = stats.anderson(series, dist="norm")
        # Extract the critical value at 5% significance (index 2 = 5%)
        crit_5pct = float(ad_result.critical_values[2])
        ad_stat   = float(ad_result.statistic)
        ad_normal = ad_stat < crit_5pct
        results["anderson_darling"] = {
            "statistic": round(ad_stat, 6),
            "critical_value_5pct": round(crit_5pct, 6),
            "is_normal": ad_normal,
        }
        total_votes += 1
        if ad_normal:
            normal_votes += 1
    except Exception:
        results["anderson_darling"] = {"error": "could not compute"}

    # Jarque-Bera
    try:
        jb_stat, jb_p = stats.jarque_bera(series)
        results["jarque_bera"] = {
            "statistic": round(float(jb_stat), 6),
            "p_value": round(float(jb_p), 6),
        }
        total_votes += 1
        if jb_p > 0.05:
            normal_votes += 1
    except Exception:
        results["jarque_bera"] = {"error": "could not compute"}

    # D'Agostino-Pearson omnibus test (requires n >= 20)
    try:
        if len(series) >= 20:
            dp_stat, dp_p = stats.normaltest(series)
            results["dagostino_pearson"] = {
                "statistic": round(float(dp_stat), 6),
                "p_value": round(float(dp_p), 6),
            }
            total_votes += 1
            if dp_p > 0.05:
                normal_votes += 1
        else:
            results["dagostino_pearson"] = {"error": "n < 20, test skipped"}
    except Exception:
        results["dagostino_pearson"] = {"error": "could not compute"}

    # Majority vote
    is_normal = (normal_votes > total_votes / 2) if total_votes > 0 else False
    results["is_normal"] = is_normal
    results["votes_normal"] = normal_votes
    results["votes_total"] = total_votes

    return results


def _best_fit_distribution(series):
    """
    Fit common distributions via KS goodness-of-fit and return
    (best_distribution_name, ks_statistic).
    """
    candidates = {
        "norm":    stats.norm,
        "lognorm": stats.lognorm,
        "expon":   stats.expon,
        "gamma":   stats.gamma,
        "beta":    stats.beta,
    }

    best_name = "norm"
    best_ks   = float("inf")

    for name, dist in candidates.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(series)
                ks_stat, _ = stats.kstest(series, name, args=params)
            ks_stat = round(float(ks_stat), 6)
            if ks_stat < best_ks:
                best_ks   = ks_stat
                best_name = name
        except Exception:
            continue

    return best_name, best_ks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nice_bins(series):
    """Sturges + Scott hybrid, capped at 50."""
    n = len(series)
    if n < 10:
        return 5
    sturges = int(np.ceil(np.log2(n) + 1))
    return min(max(sturges, 10), 50)


def _shape_tag(skewness):
    if skewness > 1.0:
        return "heavily right-skewed"
    if skewness > 0.5:
        return "right-skewed"
    if skewness < -1.0:
        return "heavily left-skewed"
    if skewness < -0.5:
        return "left-skewed"
    return "approximately symmetric"
