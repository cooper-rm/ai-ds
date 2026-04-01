import pandas as pd


def load_data(state: dict) -> dict:
    filepath = state["filepath"]
    df = pd.read_csv(filepath)
    state["data"] = df

    state["nodes"]["load_data"] = {
        "status": "loaded",
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    return state
