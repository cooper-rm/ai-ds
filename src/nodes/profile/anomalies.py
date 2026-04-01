import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_and_show


def anomalies(state: dict) -> dict:
    """Check for distributional anomalies — zero-variance, extreme skewness, high cardinality."""
    df = state["data"]

    zero_variance = _check_zero_variance(df)
    extreme_skew = _check_skewness(df)
    high_cardinality = _check_high_cardinality(df)

    has_anomalies = bool(zero_variance or extreme_skew or high_cardinality)

    state["nodes"]["anomalies"] = {
        "status": "anomalies_found" if has_anomalies else "clean",
        "zero_variance": zero_variance,
        "extreme_skew": extreme_skew,
        "high_cardinality": high_cardinality,
        "images": [],
    }

    # Generate visualizations
    images = []
    if extreme_skew:
        images += _plot_skewed_distributions(df, extreme_skew, state)
    images += _plot_missing_heatmap(df, state)
    state["nodes"]["anomalies"]["images"] = images

    # Print summary
    print(f"   Zero-variance: {len(zero_variance)} columns")
    for item in zero_variance:
        print(f"     {item['column']}: constant = {item['constant_value']}")

    print(f"   Extreme skew: {len(extreme_skew)} columns")
    for item in extreme_skew[:5]:
        print(f"     {item['column']}: skew={item['skew']} (mean={item['mean']}, median={item['median']})")

    print(f"   High cardinality: {len(high_cardinality)} columns")
    for item in high_cardinality:
        print(f"     {item['column']}: {item['nunique']} unique ({item['ratio_pct']}% of rows)")

    from src.report import narrate, add_section
    narrative = narrate("Data Anomalies", {
        "zero_variance": zero_variance,
        "extreme_skew": extreme_skew,
        "high_cardinality": high_cardinality,
    })
    add_section(state, "Data Anomalies", narrative, images)

    return state


def _check_zero_variance(df) -> list:
    """Find columns with only one unique value."""
    results = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            results.append({"column": col, "constant_value": str(val)})
    return results


def _check_skewness(df) -> list:
    """Find numeric columns with extreme skewness (|skew| > 2)."""
    results = []
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        skew_val = round(float(numeric[col].skew()), 2)
        if abs(skew_val) > 2:
            results.append({
                "column": col,
                "skew": skew_val,
                "mean": round(float(numeric[col].mean()), 2),
                "median": round(float(numeric[col].median()), 2),
            })
    return sorted(results, key=lambda x: abs(x["skew"]), reverse=True)


def _check_high_cardinality(df) -> list:
    """Find categorical columns where >50% of values are unique."""
    results = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique()
        ratio = nunique / len(df) if len(df) > 0 else 0
        if ratio > 0.5:
            results.append({
                "column": col,
                "nunique": nunique,
                "ratio_pct": round(ratio * 100, 1),
            })
    return results


def _plot_skewed_distributions(df, skewed_cols, state) -> list:
    """Plot distributions of highly skewed columns."""
    images = []
    cols = [item["column"] for item in skewed_cols[:6]]
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
        skew_val = [item["skew"] for item in skewed_cols if item["column"] == col][0]
        ax.set_title(f"{col}\nskew={skew_val}", fontsize=12, fontweight="bold")
        ax.axvline(data.mean(), color="#C44E52", linestyle="--", label="mean")
        ax.axvline(data.median(), color="#55A868", linestyle="--", label="median")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Skewed Distributions", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_and_show(fig, state, "skewed_distributions.png")
    images.append(path)
    plt.close()

    return images


def _plot_missing_heatmap(df, state) -> list:
    """Plot missing value heatmap."""
    images = []

    missing_cols = df.columns[df.isnull().any()]
    if len(missing_cols) == 0:
        return images

    fig, ax = plt.subplots(figsize=(max(8, len(missing_cols) * 0.8), 6))
    sns.heatmap(
        df[missing_cols].isnull().astype(int),
        cbar=False,
        yticklabels=False,
        cmap=["#E8E8E8", "#C44E52"],
        ax=ax,
    )
    ax.set_title("Missing Values Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    plt.tight_layout()
    path = save_and_show(fig, state, "missing_heatmap.png")
    images.append(path)
    plt.close()

    return images
