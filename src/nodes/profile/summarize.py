from src.report import narrate, add_section


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

    narrative = narrate("Dataset Summary", {
        "shape": list(df.shape),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_columns": {k: v for k, v in state["nodes"]["summarize"]["missing_pct"].items() if v > 0},
        "dtypes": state["nodes"]["summarize"]["dtypes"],
    })
    add_section(state, "Dataset Summary", narrative)

    return state
