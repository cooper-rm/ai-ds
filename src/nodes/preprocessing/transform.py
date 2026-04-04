import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.report import narrate, add_section
from src.utils import save_and_show, snapshot
from src.terminal import print_info, print_detail


def transform(state: dict) -> dict:
    """Apply numerical transformations based on the synthesis plan."""
    df = state["data"]
    plan = state["nodes"]["synthesis"]
    to_transform = plan.get("transform", [])

    if not to_transform:
        state["nodes"]["transform"] = {"status": "nothing_to_transform"}
        print_info("nothing to transform")
        return state

    results = []
    images = []

    for item in to_transform:
        col = item["column"]
        method = item["method"]

        if col not in df.columns:
            results.append({"column": col, "status": "not_found"})
            print_info(f"{col}: not found, skipping")
            continue

        before_data = df[col].copy()

        if method == "log1p":
            df[col] = np.log1p(df[col])
            results.append({"column": col, "method": "log1p"})

        elif method == "sqrt":
            df[col] = np.sqrt(df[col])
            results.append({"column": col, "method": "sqrt"})

        elif method == "standard_scale":
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
                results.append({"column": col, "method": "standard_scale", "mean": round(float(mean), 4), "std": round(float(std), 4)})
            else:
                results.append({"column": col, "status": "zero_std", "method": "standard_scale"})
                print_info(f"{col}: zero std, skipping standard_scale")
                continue

        elif method == "minmax_scale":
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                results.append({"column": col, "method": "minmax_scale", "min": round(float(min_val), 4), "max": round(float(max_val), 4)})
            else:
                results.append({"column": col, "status": "constant", "method": "minmax_scale"})
                print_info(f"{col}: constant value, skipping minmax_scale")
                continue

        elif method == "boxcox":
            from scipy.stats import boxcox as _boxcox
            series = df[col].dropna()
            min_val = series.min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                series = series + shift
                df[col] = df[col] + shift
            else:
                shift = 0
            transformed, fitted_lambda = _boxcox(series)
            df[col] = np.nan
            df.loc[series.index, col] = transformed
            results.append({"column": col, "method": "boxcox", "fitted_lambda": round(float(fitted_lambda), 4), "shift": round(float(shift), 4)})

        elif method == "yeojohnson":
            from scipy.stats import yeojohnson as _yeojohnson
            series = df[col].dropna()
            transformed, fitted_lambda = _yeojohnson(series)
            df[col] = np.nan
            df.loc[series.index, col] = transformed
            results.append({"column": col, "method": "yeojohnson", "fitted_lambda": round(float(fitted_lambda), 4)})

        elif method == "robust_scale":
            median = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[col] = (df[col] - median) / iqr
                results.append({"column": col, "method": "robust_scale", "median": round(float(median), 4), "iqr": round(float(iqr), 4)})
            else:
                results.append({"column": col, "status": "zero_iqr", "method": "robust_scale"})
                print_info(f"{col}: zero IQR, skipping robust_scale")
                continue

        elif method == "power":
            from sklearn.preprocessing import PowerTransformer as _PowerTransformer
            pt = _PowerTransformer(method="yeo-johnson")
            series = df[col].dropna()
            transformed = pt.fit_transform(series.values.reshape(-1, 1)).ravel()
            df[col] = np.nan
            df.loc[series.index, col] = transformed
            fitted_lambda = pt.lambdas_[0]
            results.append({"column": col, "method": "power", "fitted_lambda": round(float(fitted_lambda), 4)})

        elif method == "quantile":
            from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer
            series = df[col].dropna()
            n_quantiles = min(1000, len(series))
            qt = _QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles)
            transformed = qt.fit_transform(series.values.reshape(-1, 1)).ravel()
            df[col] = np.nan
            df.loc[series.index, col] = transformed
            results.append({"column": col, "method": "quantile", "n_quantiles": n_quantiles})

        elif method == "rank":
            df[col] = df[col].rank(method="average") / len(df)
            results.append({"column": col, "method": "rank"})

        else:
            results.append({"column": col, "status": "unknown_method", "method": method})
            print_info(f"{col}: unknown method '{method}', skipping")
            continue

        img = _plot_transform(before_data, df[col], col, method, state)
        if img:
            images.append(img)

        print_info(f"{col}: {method}")

    state["data"] = df
    state["nodes"]["transform"] = {
        "status": "transformed",
        "results": results,
        "images": [str(p) for p in images],
    }

    snapshot(state, "transform")
    narrative = narrate("Numerical Transformations", {"results": results})
    add_section(state, "Numerical Transformations", narrative, images)

    return state


def _plot_transform(before, after, col_name, method, state):
    """Plot distribution before and after transformation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(before.dropna(), bins=30, color="#C44E52", edgecolor="white", alpha=0.8)
    axes[0].set_title("Before", fontsize=12)
    axes[0].set_xlabel(col_name)

    axes[1].hist(after.dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[1].set_title(f"After ({method})", fontsize=12)
    axes[1].set_xlabel(col_name)

    plt.suptitle(f"Transform: {col_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = save_and_show(fig, state, f"transform_{col_name.lower()}.png")
    plt.close()
    return path
