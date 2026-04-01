def summarize(state: dict) -> dict:
    df = state["data"]

    state["nodes"]["summarize"] = {
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "numeric_stats": df.describe().round(2).to_dict() if len(df.select_dtypes(include="number").columns) > 0 else {},
    }

    print(f"   Shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    return state
