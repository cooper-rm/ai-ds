import json

from src.llm.client import ask


def synthesis(state: dict) -> dict:
    """LLM reads all profiling results and produces machine-actionable preprocessing plan."""
    nodes = state["nodes"]

    profile_summary = {
        "dataset": {
            "file": nodes.get("analyze_file", {}).get("filepath"),
            "row_count": nodes.get("load_data", {}).get("row_count"),
            "column_count": nodes.get("load_data", {}).get("column_count"),
            "memory_mb": nodes.get("optimize_dtypes", {}).get("after_mb"),
        },
        "column_classifications": nodes.get("classify", {}).get("classifications", {}),
        "missing_values": nodes.get("summarize", {}).get("missing", {}),
        "missing_pct": nodes.get("summarize", {}).get("missing_pct", {}),
        "numeric_stats": nodes.get("summarize", {}).get("numeric_stats", {}),
        "dtype_optimizations_applied": nodes.get("optimize_dtypes", {}).get("changes", []),
        "type_issues": {
            "numeric_as_string": nodes.get("types", {}).get("numeric_as_string", []),
            "dates_as_string": nodes.get("types", {}).get("dates_as_string", []),
            "suggest_categorical": nodes.get("types", {}).get("suggest_categorical", []),
        },
        "structural_issues": nodes.get("structure", {}).get("status"),
        "sentinels": nodes.get("structure", {}).get("sentinels", {}).get("values_found", {}),
        "anomalies": {
            "zero_variance": nodes.get("anomalies", {}).get("zero_variance", []),
            "extreme_skew": nodes.get("anomalies", {}).get("extreme_skew", []),
            "high_cardinality": nodes.get("anomalies", {}).get("high_cardinality", []),
        },
    }

    prompt = f"""You are producing a machine-actionable preprocessing plan from profiling results.
Do NOT assume you know this dataset. Base recommendations ONLY on the profiling data provided.

Profiling results:
{json.dumps(profile_summary, indent=2)}

Produce a preprocessing plan as structured actions that code can execute.

Rules:
- Each action must map to a specific operation with exact parameters
- Only recommend what the data supports — do not guess domain knowledge
- Be concise — no prose, just structured decisions

Respond with EXACTLY this JSON format, nothing else:
{{
  "quality_score": 1-10,
  "quality_flags": ["list of specific issues found"],
  "drop_columns": [
    {{"column": "name", "reason": "identifier|zero_variance|too_sparse"}}
  ],
  "impute": [
    {{"column": "name", "method": "median|mean|mode|grouped_median|drop_rows", "group_by": ["col1"] or null}}
  ],
  "encode": [
    {{"column": "name", "method": "label|onehot|ordinal", "categories": ["ordered list"] or null}}
  ],
  "transform": [
    {{"column": "name", "method": "log1p|sqrt|standard_scale|minmax_scale"}}
  ],
  "engineer": [
    {{"name": "new_col_name", "operation": "sum|ratio|extract|bin|indicator", "source_columns": ["col1", "col2"], "params": {{}} }}
  ],
  "preprocessing_order": [
    {{"step": 1, "action": "drop|impute|encode|transform|engineer", "targets": ["col1", "col2"]}}
  ]
}}"""

    response = ask(prompt, system="You are a data preprocessing planner. Respond only in JSON. Be concise and precise.")

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"error": "Could not parse LLM response", "raw": response[:500]}

    state["nodes"]["synthesis"] = result

    # Print summary
    print(f"\n   === PREPROCESSING PLAN ===\n")
    print(f"   Quality: {result.get('quality_score', '?')}/10")

    flags = result.get("quality_flags", [])
    if flags:
        for flag in flags:
            print(f"   ! {flag}")

    drops = result.get("drop_columns", [])
    if drops:
        print(f"\n   Drop: {len(drops)} columns")
        for d in drops:
            print(f"     {d['column']} ({d['reason']})")

    imputes = result.get("impute", [])
    if imputes:
        print(f"\n   Impute: {len(imputes)} columns")
        for imp in imputes:
            group = f" by {imp['group_by']}" if imp.get("group_by") else ""
            print(f"     {imp['column']}: {imp['method']}{group}")

    encodes = result.get("encode", [])
    if encodes:
        print(f"\n   Encode: {len(encodes)} columns")
        for enc in encodes:
            print(f"     {enc['column']}: {enc['method']}")

    transforms = result.get("transform", [])
    if transforms:
        print(f"\n   Transform: {len(transforms)} columns")
        for t in transforms:
            print(f"     {t['column']}: {t['method']}")

    engineers = result.get("engineer", [])
    if engineers:
        print(f"\n   Engineer: {len(engineers)} features")
        for e in engineers:
            print(f"     {e['name']}: {e['operation']}({e['source_columns']})")

    order = result.get("preprocessing_order", [])
    if order:
        print(f"\n   Execution order:")
        for step in order:
            print(f"     {step['step']}. {step['action']} → {step['targets']}")

    from src.report import narrate, add_section
    narrative = narrate("Preprocessing Plan", result, context="This is the LLM's recommended plan based on all profiling results.")
    add_section(state, "Preprocessing Plan", narrative)

    return state
