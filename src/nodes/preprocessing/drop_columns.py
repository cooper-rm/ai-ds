from src.report import narrate, add_section
from src.terminal import print_info, print_detail
from src.utils import snapshot


def drop_columns(state: dict) -> dict:
    """Drop columns flagged by the synthesis plan."""
    df = state["data"]
    plan = state["nodes"]["synthesis"]
    to_drop = plan.get("drop_columns", [])

    if not to_drop:
        state["nodes"]["drop_columns"] = {"status": "nothing_to_drop"}
        print_info("nothing to drop")
        return state

    column_names = [d["column"] for d in to_drop if d["column"] in df.columns]
    missing = [d["column"] for d in to_drop if d["column"] not in df.columns]

    before_cols = len(df.columns)
    df = df.drop(columns=column_names)
    after_cols = len(df.columns)

    state["data"] = df
    state["nodes"]["drop_columns"] = {
        "status": "dropped",
        "dropped": [{"column": d["column"], "reason": d["reason"]} for d in to_drop if d["column"] in column_names],
        "not_found": missing,
        "before_column_count": before_cols,
        "after_column_count": after_cols,
    }

    print_detail("dropped", ", ".join(column_names))
    if missing:
        print_info(f"not found (skipped): {missing}")
    print_detail("columns", f"{before_cols} → {after_cols}")
    snapshot(state, "drop_columns")

    narrative = narrate("Column Removal", {
        "dropped": [{"column": d["column"], "reason": d["reason"]} for d in to_drop if d["column"] in column_names],
        "before": before_cols,
        "after": after_cols,
    })
    add_section(state, "Column Removal", narrative)

    return state
