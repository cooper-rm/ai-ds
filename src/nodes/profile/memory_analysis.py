import numpy as np


def memory_analysis(state: dict) -> dict:
    """Deep memory analysis — per-column breakdown, dtype optimization recommendations."""
    df = state["data"]

    # Per-column memory breakdown
    mem = df.memory_usage(deep=True)
    total_memory = mem.sum()
    total_memory_mb = round(total_memory / (1024 * 1024), 4)

    # Exclude index from per-column breakdown
    col_memory = {col: round(mem[col] / (1024 * 1024), 4) for col in df.columns}

    # Top consumers ranked by memory
    top_consumers = sorted(
        [{"column": col, "memory_mb": mb, "dtype": str(df[col].dtype), "pct_of_total": round(mb / total_memory_mb * 100, 1)}
         for col, mb in col_memory.items()],
        key=lambda x: x["memory_mb"],
        reverse=True,
    )

    # Dtype optimization recommendations
    optimizations = []
    projected_memory = total_memory

    for col in df.columns:
        current_dtype = str(df[col].dtype)
        current_size = mem[col]
        suggestion = _suggest_dtype(df[col], current_dtype)

        if suggestion:
            projected_size = _estimate_size(df[col], suggestion)
            savings = current_size - projected_size
            if savings > 0:
                optimizations.append({
                    "column": col,
                    "current_dtype": current_dtype,
                    "suggested_dtype": suggestion,
                    "savings_mb": round(savings / (1024 * 1024), 4),
                })
                projected_memory -= savings

    projected_memory_mb = round(projected_memory / (1024 * 1024), 4)
    savings_mb = round((total_memory - projected_memory) / (1024 * 1024), 4)
    savings_pct = round((1 - projected_memory / total_memory) * 100, 1) if total_memory > 0 else 0

    # Compare to pre-load estimate
    estimated_mb = state["nodes"].get("analyze_file", {}).get("estimated_memory_mb", 0)
    if estimated_mb > 0:
        accuracy = round(abs(total_memory_mb - estimated_mb) / estimated_mb * 100, 1)
    else:
        accuracy = None

    state["nodes"]["memory_analysis"] = {
        "actual_memory_mb": total_memory_mb,
        "estimated_memory_mb": estimated_mb,
        "estimation_accuracy_pct_off": accuracy,
        "per_column_mb": col_memory,
        "top_consumers": top_consumers[:5],
        "optimizations": optimizations,
        "projected_memory_mb": projected_memory_mb,
        "potential_savings_mb": savings_mb,
        "potential_savings_pct": savings_pct,
    }

    # Print summary
    print(f"   Actual memory: {total_memory_mb} MB")
    if accuracy is not None:
        print(f"   Pre-load estimate: {estimated_mb} MB ({accuracy}% off)")

    print(f"   Top consumers:")
    for tc in top_consumers[:3]:
        print(f"     {tc['column']}: {tc['memory_mb']} MB ({tc['dtype']}) — {tc['pct_of_total']}%")

    if optimizations:
        print(f"   Optimizations: {len(optimizations)} columns")
        for opt in optimizations[:3]:
            print(f"     {opt['column']}: {opt['current_dtype']} → {opt['suggested_dtype']} (saves {opt['savings_mb']} MB)")
        print(f"   Projected: {projected_memory_mb} MB (saves {savings_mb} MB / {savings_pct}%)")
    else:
        print(f"   No dtype optimizations found")

    from src.report import narrate, add_section
    narrative = narrate("Memory Analysis", {
        "actual_memory_mb": total_memory_mb,
        "top_consumers": top_consumers[:5],
        "optimization_count": len(optimizations),
        "projected_memory_mb": projected_memory_mb,
        "savings_pct": savings_pct,
    })
    add_section(state, "Memory Analysis", narrative)

    return state


def _suggest_dtype(series, current_dtype: str) -> str | None:
    """Suggest a more memory-efficient dtype for a series."""
    if current_dtype == "int64":
        min_val, max_val = series.min(), series.max()
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            return "int8"
        if min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            return "int16"
        if min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            return "int32"

    if current_dtype == "float64":
        if not series.isnull().any():
            min_val, max_val = series.min(), series.max()
            if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
                return "float32"

    if current_dtype == "object":
        nunique = series.nunique()
        ratio = nunique / len(series) if len(series) > 0 else 1
        if nunique < 50 or ratio < 0.05:
            return "category"

    return None


def _estimate_size(series, target_dtype: str) -> int:
    """Estimate memory size if column were converted to target dtype."""
    n = len(series)
    if target_dtype == "int8":
        return n * 1
    if target_dtype == "int16":
        return n * 2
    if target_dtype == "int32":
        return n * 4
    if target_dtype == "float32":
        return n * 4
    if target_dtype == "category":
        return series.astype("category").memory_usage(deep=True)
    return series.memory_usage(deep=True)
