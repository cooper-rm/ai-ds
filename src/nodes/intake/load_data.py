import pandas as pd

from src.utils import snapshot
from src.terminal import print_info


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

    print_info(f"loaded {len(df)} rows × {len(df.columns)} columns")
    snapshot(state, "load_data")
    return state
