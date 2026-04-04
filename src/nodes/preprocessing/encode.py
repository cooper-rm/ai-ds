import math

import numpy as np
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

        elif method == "target":
            target_col = item["target_col"]
            global_mean = df[target_col].mean()
            global_prior = 10
            agg = df.groupby(col)[target_col].agg(["mean", "count"])
            mapping = {}
            for cat, row in agg.iterrows():
                cat_mean = row["mean"]
                count = row["count"]
                smoothed = (count * cat_mean + global_prior * global_mean) / (count + global_prior)
                mapping[cat] = smoothed
            df[col] = df[col].map(mapping)
            mappings[col] = {str(k): float(v) for k, v in mapping.items()}
            results.append({"column": col, "method": "target", "mapping": mappings[col]})
            print_info(f"{col}: target  —  {len(mapping)} categories encoded")

        elif method == "frequency":
            total = len(df)
            freq = df[col].value_counts() / total
            mapping = freq.to_dict()
            df[col] = df[col].map(mapping)
            mappings[col] = {str(k): float(v) for k, v in mapping.items()}
            results.append({"column": col, "method": "frequency", "mapping": mappings[col]})
            print_info(f"{col}: frequency  —  {len(mapping)} categories encoded")

        elif method == "binary":
            unique_vals = sorted(df[col].dropna().unique().tolist())
            n_unique = len(unique_vals)
            n_bits = max(math.ceil(math.log2(n_unique)), 1) if n_unique > 0 else 1
            code_map = {v: i for i, v in enumerate(unique_vals)}
            codes = df[col].map(code_map)
            new_cols = []
            for bit in range(n_bits):
                bit_col = f"{col}_b{bit}"
                df[bit_col] = codes.apply(
                    lambda x, b=bit: int((x >> b) & 1) if pd.notna(x) else np.nan
                )
                new_cols.append(bit_col)
            df = df.drop(columns=[col])
            results.append({"column": col, "method": "binary", "new_columns": new_cols})
            print_info(f"{col}: binary  —  {n_bits} bit columns")

        elif method == "woe":
            target_col = item["target_col"]
            total_event = df[target_col].sum()
            total_non_event = len(df) - total_event
            woe_map = {}
            iv_total = 0.0
            for cat in df[col].dropna().unique():
                mask = df[col] == cat
                event = df.loc[mask, target_col].sum() + 0.5
                non_event = mask.sum() - df.loc[mask, target_col].sum() + 0.5
                pct_event = event / (total_event + 0.5 * df[col].nunique())
                pct_non_event = non_event / (total_non_event + 0.5 * df[col].nunique())
                woe_val = np.log(pct_event / pct_non_event)
                woe_map[cat] = woe_val
                iv_total += (pct_event - pct_non_event) * woe_val
            df[col] = df[col].map(woe_map)
            mappings[col] = {str(k): float(v) for k, v in woe_map.items()}
            results.append({
                "column": col,
                "method": "woe",
                "mapping": mappings[col],
                "iv": float(iv_total),
            })
            print_info(f"{col}: woe  —  IV={iv_total:.4f}")

        elif method == "hash":
            n_components = item.get("n_components", 8)
            new_cols = [f"{col}_h{i}" for i in range(n_components)]
            for hcol in new_cols:
                df[hcol] = 0
            for idx in df.index:
                val = df.at[idx, col]
                if pd.notna(val):
                    h = hash(str(val)) % n_components
                    df.at[idx, f"{col}_h{h}"] = 1
            df = df.drop(columns=[col])
            results.append({"column": col, "method": "hash", "new_columns": new_cols, "n_components": n_components})
            print_info(f"{col}: hash  —  {n_components} components")

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
