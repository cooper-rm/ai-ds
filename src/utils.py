import os
import subprocess


def save_and_show(fig, state: dict, filename: str) -> str:
    """Save a matplotlib figure to the project images dir and display with imgcat."""
    images_dir = os.path.join(state["project_dir"], "images")
    os.makedirs(images_dir, exist_ok=True)

    filepath = os.path.join(images_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    fig.clf()

    try:
        subprocess.run(["imgcat", filepath], check=False)
    except FileNotFoundError:
        pass

    return filepath


def snapshot(state: dict, node_name: str) -> None:
    """
    Save a versioned parquet snapshot of state["data"] after a mutating node.

    Diffs against the previous version to record columns_added, columns_removed,
    columns_changed (dtype changes). Appends an entry to state["data_versions"].
    """
    import pandas as pd

    df = state["data"]
    versions = state.setdefault("data_versions", [])

    version_num = len(versions)
    filename = f"v{version_num:02d}_{node_name}.parquet"
    data_dir = os.path.join(state["project_dir"], "data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)

    df.to_parquet(path, index=False)

    # Diff against previous version
    columns_added = []
    columns_removed = []
    columns_changed = []

    if versions:
        prev = versions[-1]
        prev_cols = set(prev["columns"])
        curr_cols = set(df.columns.tolist())
        columns_added = list(curr_cols - prev_cols)
        columns_removed = list(prev_cols - curr_cols)
        # Dtype changes on columns present in both
        prev_dtypes = prev["dtypes"]
        for col in curr_cols & prev_cols:
            curr_dtype = str(df[col].dtype)
            if prev_dtypes.get(col) != curr_dtype:
                columns_changed.append({
                    "column": col,
                    "from": prev_dtypes.get(col),
                    "to": curr_dtype,
                })

    entry = {
        "version": version_num,
        "node": node_name,
        "path": path,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 3),
        "columns_added": columns_added,
        "columns_removed": columns_removed,
        "columns_changed": columns_changed,
    }

    versions.append(entry)

    from src.terminal import print_info
    print_info(
        f"snapshot v{version_num:02d}  {df.shape[0]}×{df.shape[1]}  "
        f"({entry['memory_mb']} MB)  →  data/{filename}"
    )
