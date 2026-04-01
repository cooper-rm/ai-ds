import numpy as np


def optimize_dtypes(state: dict) -> dict:
    """Optimize numeric dtypes for memory. Leaves strings untouched for preprocessing phase."""
    df = state["data"]
    classifications = state["nodes"]["classify"]["classifications"]

    before_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)
    changes = []

    for col in df.columns:
        col_class = classifications.get(col, {}).get("type", "unknown")
        old_dtype = str(df[col].dtype)
        new_dtype = None

        if df[col].dtype == "int64":
            min_val, max_val = df[col].min(), df[col].max()
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                df[col] = df[col].astype("int8")
                new_dtype = "int8"
            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                df[col] = df[col].astype("int16")
                new_dtype = "int16"
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")
                new_dtype = "int32"

        elif df[col].dtype == "float64":
            if not df[col].isnull().any():
                df[col] = df[col].astype("float32")
                new_dtype = "float32"

        if new_dtype:
            changes.append({
                "column": col,
                "from": old_dtype,
                "to": new_dtype,
                "classification": col_class,
            })

    after_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 4)
    savings_mb = round(before_mb - after_mb, 4)
    savings_pct = round((1 - after_mb / before_mb) * 100, 1) if before_mb > 0 else 0

    state["data"] = df
    state["nodes"]["optimize_dtypes"] = {
        "before_mb": before_mb,
        "after_mb": after_mb,
        "savings_mb": savings_mb,
        "savings_pct": savings_pct,
        "changes": changes,
    }

    print(f"   Before: {before_mb} MB → After: {after_mb} MB")
    print(f"   Saved: {savings_mb} MB ({savings_pct}%)")
    print(f"   Changes: {len(changes)} columns")
    for c in changes:
        print(f"     {c['column']}: {c['from']} → {c['to']} ({c['classification']})")

    return state
