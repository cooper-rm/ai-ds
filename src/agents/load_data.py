import pandas as pd


def load_data(state: dict) -> dict:
    filepath = state["filepath"]
    state["data"] = pd.read_csv(filepath)
    print(f"Loaded {len(state['data'])} rows, {len(state['data'].columns)} columns")
    return state
