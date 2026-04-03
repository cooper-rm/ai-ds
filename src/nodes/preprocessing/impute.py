from src.report import narrate, add_section
from src.utils import snapshot
from src.terminal import print_info, print_detail

import src.nodes.imputation.methods.simple      as simple_methods
import src.nodes.imputation.methods.statistical as stat_methods
import src.nodes.imputation.methods.deep        as deep_methods

METHOD_DISPATCH = {
    "mean":                  (simple_methods, "mean"),
    "median":                (simple_methods, "median"),
    "mode":                  (simple_methods, "mode"),
    "grouped_median":        (simple_methods, "grouped_median"),
    "knn":                   (simple_methods, "knn"),
    "regression":            (stat_methods,   "regression"),
    "stochastic_regression": (stat_methods,   "stochastic_regression"),
    "pmm":                   (stat_methods,   "pmm"),
    "hotdeck":               (stat_methods,   "hotdeck"),
    "mice":                  (stat_methods,   "mice"),
    "missforest":            (stat_methods,   "missforest"),
    "em":                    (stat_methods,   "em"),
    "softimpute":            (stat_methods,   "softimpute"),
    "gain":                  (deep_methods,   "gain"),
    "mida":                  (deep_methods,   "mida"),
    "hivae":                 (deep_methods,   "hivae"),
}


def impute(state: dict) -> dict:
    """Execute the winning imputation method per column (chosen during profile phase)."""
    df = state["data"]
    plan = state["nodes"]["synthesis"]
    to_impute = plan.get("impute", [])

    # Pull imputation evidence for method dispatch
    imputation_results = state["nodes"].get("imputation", {}).get("results", {})

    if not to_impute:
        state["nodes"]["impute"] = {"status": "nothing_to_impute"}
        print_info("nothing to impute")
        return state

    results = []

    for item in to_impute:
        col       = item["column"]
        method    = item["method"]
        group_by  = item.get("group_by")
        regressors = item.get("regressors")

        if col not in df.columns:
            results.append({"column": col, "status": "not_found"})
            print_info(f"{col}: not found, skipping")
            continue

        before_missing = int(df[col].isnull().sum())
        if before_missing == 0:
            results.append({"column": col, "status": "no_missing", "method": method})
            print_info(f"{col}: no missing values, skipping")
            continue

        before_data = df[col].copy()
        is_numeric  = str(df[col].dtype) in ("float64", "float32", "int64", "int32", "int16", "int8")

        # Build evidence dict for method runners (same interface as profile phase)
        imp_result = imputation_results.get(col, {})
        ev = {
            "is_numeric":               is_numeric,
            "dtype":                    str(df[col].dtype),
            "mar_correlates":           imp_result.get("regressors", regressors or []),
            "top_categorical_correlate": (group_by[0] if group_by else None),
        }

        if method == "drop_rows":
            before_rows = len(df)
            df = df.dropna(subset=[col])
            dropped = before_rows - len(df)
            results.append({"column": col, "method": "drop_rows", "rows_dropped": dropped})
        elif method in METHOD_DISPATCH:
            try:
                mod, fn  = METHOD_DISPATCH[method]
                imputed  = getattr(mod, fn)(df, col, ev)
                if imputed is not None:
                    df[col] = imputed
                    results.append({"column": col, "method": method, "filled": before_missing})
                else:
                    raise ValueError("runner returned None")
            except Exception as e:
                # Fallback to median/mode
                print_info(f"  {method} failed ({e}), falling back to median/mode")
                if is_numeric:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                results.append({"column": col, "method": f"fallback_{method}", "filled": before_missing})
        else:
            # Unknown method — simple fallback
            if is_numeric:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if df[col].notnull().any() else "unknown")
            results.append({"column": col, "method": f"fallback_{method}", "filled": before_missing})

        print_info(f"{col}: {method}  —  filled {before_missing} values")

    after_missing = int(df.isnull().sum().sum())

    state["data"] = df
    state["nodes"]["impute"] = {
        "status": "imputed",
        "results": results,
        "remaining_missing": after_missing,
    }

    print_detail("remaining missing", after_missing)
    snapshot(state, "impute")

    narrative = narrate("Missing Value Imputation", {
        "results": results,
        "remaining_missing": after_missing,
    })
    add_section(state, "Missing Value Imputation", narrative)
    return state

