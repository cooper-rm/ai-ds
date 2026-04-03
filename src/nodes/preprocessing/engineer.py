import pandas as pd

from src.report import narrate, add_section
from src.terminal import print_info
from src.utils import snapshot


def engineer(state: dict) -> dict:
    """Create new features based on the synthesis plan."""
    df = state["data"]
    plan = state["nodes"]["synthesis"]
    to_engineer = plan.get("engineer", [])

    if not to_engineer:
        state["nodes"]["engineer"] = {"status": "nothing_to_engineer"}
        print_info("nothing to engineer")
        return state

    results = []

    for item in to_engineer:
        name = item["name"]
        operation = item["operation"]
        source_cols = item.get("source_columns", [])
        params = item.get("params", {})

        valid_sources = [c for c in source_cols if c in df.columns]
        if not valid_sources and operation != "indicator":
            results.append({"name": name, "status": "missing_sources", "needed": source_cols})
            print_info(f"{name}: source columns {source_cols} not found, skipping")
            continue

        if operation == "sum":
            df[name] = df[valid_sources].sum(axis=1)
            results.append({"name": name, "operation": "sum", "sources": valid_sources})

        elif operation == "ratio":
            if len(valid_sources) == 2:
                denominator = df[valid_sources[1]].replace(0, float("nan"))
                df[name] = df[valid_sources[0]] / denominator
                df[name] = df[name].fillna(0)
                results.append({"name": name, "operation": "ratio", "sources": valid_sources})

        elif operation == "indicator":
            # Check if source is a derived column (like FamilySize) or existing
            source = valid_sources[0] if valid_sources else source_cols[0]
            if source in df.columns:
                threshold = params.get("threshold", 1)
                df[name] = (df[source] == threshold).astype("int8")
                results.append({"name": name, "operation": "indicator", "source": source, "threshold": threshold})
            else:
                results.append({"name": name, "status": "source_not_found", "needed": source})
                print_info(f"{name}: source {source} not found, skipping")
                continue

        elif operation == "bin":
            source = valid_sources[0]
            n_bins = params.get("bins", 4)
            labels = params.get("labels")
            df[name] = pd.qcut(df[source], q=n_bins, labels=labels, duplicates="drop")
            results.append({"name": name, "operation": "bin", "source": source, "bins": n_bins})

        elif operation == "extract":
            source = valid_sources[0]
            pattern = params.get("pattern", r"^([A-Za-z]+)")
            df[name] = df[source].astype(str).str.extract(pattern, expand=False)
            results.append({"name": name, "operation": "extract", "source": source, "pattern": pattern})

        else:
            results.append({"name": name, "status": "unknown_operation", "operation": operation})
            print_info(f"{name}: unknown operation '{operation}', skipping")
            continue

        print_info(f"{name}: {operation}({valid_sources or source_cols})")

    state["data"] = df
    state["nodes"]["engineer"] = {
        "status": "engineered",
        "results": results,
        "new_column_count": len(df.columns),
    }

    snapshot(state, "engineer")
    narrative = narrate("Feature Engineering", {"results": results})
    add_section(state, "Feature Engineering", narrative)

    return state
