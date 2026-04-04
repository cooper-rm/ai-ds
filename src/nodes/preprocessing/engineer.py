import numpy as np
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

        elif operation == "polynomial":
            degree = min(params.get("degree", 2), 2)
            created = []
            for i, col_a in enumerate(valid_sources):
                df[f"{col_a}_sq"] = df[col_a] ** 2
                created.append(f"{col_a}_sq")
                if degree == 2:
                    for col_b in valid_sources[i + 1:]:
                        interaction = f"{col_a}_x_{col_b}"
                        df[interaction] = df[col_a] * df[col_b]
                        created.append(interaction)
            results.append({"name": name, "operation": "polynomial", "sources": valid_sources, "created": created})

        elif operation == "datetime":
            source = valid_sources[0]
            components = params.get("components", ["year", "month", "day", "dayofweek", "hour"])
            dt_col = pd.to_datetime(df[source], errors="coerce")
            created = []
            for comp in components:
                col_name = f"{source}_{comp}"
                if comp == "dayofweek":
                    df[col_name] = dt_col.dt.dayofweek
                else:
                    df[col_name] = getattr(dt_col.dt, comp, None)
                created.append(col_name)
            results.append({"name": name, "operation": "datetime", "source": source, "created": created})

        elif operation == "log":
            source = valid_sources[0]
            df[name] = np.log1p(df[source])
            results.append({"name": name, "operation": "log", "source": source})

        elif operation == "multiply":
            if len(valid_sources) == 2:
                df[name] = df[valid_sources[0]] * df[valid_sources[1]]
                results.append({"name": name, "operation": "multiply", "sources": valid_sources})

        elif operation == "groupby_agg":
            group_col = params.get("group_col")
            agg_col = params.get("agg_col")
            agg_func = params.get("agg_func", "mean")
            if group_col in df.columns and agg_col in df.columns:
                df[name] = df.groupby(group_col)[agg_col].transform(agg_func)
                results.append({"name": name, "operation": "groupby_agg", "group_col": group_col, "agg_col": agg_col, "agg_func": agg_func})
            else:
                results.append({"name": name, "status": "missing_sources", "needed": [group_col, agg_col]})
                print_info(f"{name}: groupby_agg columns not found, skipping")
                continue

        elif operation == "count":
            source = valid_sources[0]
            df[name] = df[source].map(df[source].value_counts())
            results.append({"name": name, "operation": "count", "source": source})

        elif operation == "reciprocal":
            source = valid_sources[0]
            df[name] = 1 / (df[source] + 1e-8)
            results.append({"name": name, "operation": "reciprocal", "source": source})

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
