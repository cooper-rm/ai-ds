import pandas as pd

from src.report import narrate, add_section
from src.terminal import print_info, print_detail
from src.utils import snapshot


def encode(state: dict) -> dict:
    """Encode categorical columns based on the synthesis plan."""
    df = state["data"]
    plan = state["nodes"]["synthesis"]
    to_encode = plan.get("encode", [])

    if not to_encode:
        state["nodes"]["encode"] = {"status": "nothing_to_encode"}
        print_info("nothing to encode")
        return state

    results = []
    mappings = {}

    for item in to_encode:
        col = item["column"]
        method = item["method"]
        categories = item.get("categories")

        if col not in df.columns:
            results.append({"column": col, "status": "not_found"})
            print_info(f"{col}: not found, skipping")
            continue

        if method == "label":
            unique_vals = sorted(df[col].dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            mappings[col] = {str(k): int(v) for k, v in mapping.items()}
            results.append({"column": col, "method": "label", "mapping": mappings[col]})
            print_info(f"{col}: label  —  {mappings[col]}")

        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, dtype="int8")
            df = df.drop(columns=[col])
            df = pd.concat([df, dummies], axis=1)
            new_cols = list(dummies.columns)
            results.append({"column": col, "method": "onehot", "new_columns": new_cols})
            print_info(f"{col}: onehot  —  {len(new_cols)} columns")

        elif method == "ordinal":
            if categories:
                mapping = {v: i for i, v in enumerate(categories)}
            else:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                mapping = {v: i for i, v in enumerate(unique_vals)}
            df[col] = df[col].map(mapping)
            mappings[col] = {str(k): int(v) for k, v in mapping.items()}
            results.append({"column": col, "method": "ordinal", "mapping": mappings[col]})
            print_info(f"{col}: ordinal  —  {mappings[col]}")

        else:
            results.append({"column": col, "status": "unknown_method", "method": method})
            print_info(f"{col}: unknown method '{method}', skipping")

    state["data"] = df
    state["nodes"]["encode"] = {
        "status": "encoded",
        "results": results,
        "mappings": mappings,
        "column_count": len(df.columns),
    }

    print_detail("columns after encoding", len(df.columns))
    snapshot(state, "encode")
    narrative = narrate("Categorical Encoding", {"results": results, "mappings": mappings})
    add_section(state, "Categorical Encoding", narrative)

    return state
