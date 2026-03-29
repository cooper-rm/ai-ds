def summarize(state: dict) -> dict:
    df = state["data"]

    state["summary"] = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
    }

    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return state
