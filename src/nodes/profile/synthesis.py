import json

from src.llm.client import ask
from src.terminal import print_info, print_detail, llm_spinner, console


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
        "distributions": {
            col: {"skewness": v["skewness"], "shape": v["shape"]}
            for col, v in nodes.get("distributions", {}).get("per_column", {}).items()
            if "skewness" in v
        },
        "outliers": {
            col: {
                "n_outliers": r["n_outliers"],
                "outlier_pct": r["outlier_pct"],
                "iqr_bounds": [r["iqr_lower"], r["iqr_upper"]],
            }
            for col, r in nodes.get("outliers", {}).get("results", {}).items()
            if r["n_outliers"] > 0
        },
        "missing_analysis": {
            col: {
                "missing_pct":      r["missing_pct"],
                "missingness_type": r["missingness_type"],
                "winner":           r["winner"],
                "winner_distortion":r["winner_score"]["total"],
                "group_by":         r.get("group_by"),
                "regressors":       r.get("regressors", []),
                "methods_tested":   r.get("methods_tested", []),
            }
            for col, r in nodes.get("imputation", {}).get("results", {}).items()
        },
        "columns_to_drop_missing": nodes.get("missing", {}).get("columns_to_drop", []),
        "data_quality": {
            "exact_duplicates": nodes.get("data_quality", {}).get("exact_duplicates", {}).get("count", 0),
            "identifier_duplicates": nodes.get("data_quality", {}).get("identifier_duplicates", []),
            "string_issues": nodes.get("data_quality", {}).get("string_issues", []),
            "constant_columns": nodes.get("data_quality", {}).get("constant_columns", []),
        },
        "correlations": {
            "top_pairs": nodes.get("correlations", {}).get("top_pairs", [])[:5],
            "strong_pearson": nodes.get("correlations", {}).get("strong_pearson", []),
            "vif_warnings": [
                v for v in (nodes.get("correlations", {}).get("vif") or [])
                if v.get("severity") != "ok"
            ],
        },
        "bivariate": {
            "significant_anova": [
                e for e in nodes.get("bivariate", {}).get("results", {}).get("categorical_x_numeric", [])
                if e.get("significant")
            ][:10],
            "significant_associations": [
                e for e in nodes.get("bivariate", {}).get("results", {}).get("categorical_x_categorical", [])
                if e.get("significant")
            ][:10],
        },
        "rare_categories": list(nodes.get("distributions", {}).get("rare_categories", {}).keys()),
        "research_context": {
            "target_column": state.get("research_context", {}).get("target_column"),
            "research_goal": state.get("research_context", {}).get("research_goal"),
            "domain_context": state.get("research_context", {}).get("domain_context"),
            "analysis_priorities": state.get("research_context", {}).get("analysis_priorities", []),
            "user_adjustments": state.get("research_context", {}).get("user_adjustments"),
        },
        "target_analysis": {
            "imbalanced": nodes.get("target_analysis", {}).get("imbalanced"),
            "class_balance": nodes.get("target_analysis", {}).get("class_balance"),
            "top_features": nodes.get("target_analysis", {}).get("feature_importance_ranking", [])[:10],
        },
        "assumptions": {
            "levene_violations": len(nodes.get("assumptions", {}).get("levene", {}).get("violations", [])),
            "durbin_watson_flags": len(nodes.get("assumptions", {}).get("durbin_watson", {}).get("flagged", [])),
            "breusch_pagan_flags": len([
                r for r in nodes.get("assumptions", {}).get("breusch_pagan", {}).get("results", [])
                if r.get("heteroscedastic")
            ]),
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
- For imputation: use the missing_analysis winner field directly — it is already the lowest-distortion method identified by testing. Use its group_by/regressors as parameters. Any column in columns_to_drop_missing goes in drop_columns with reason "too_much_missing".

Respond with EXACTLY this JSON format, nothing else:
{{
  "quality_score": 1-10,
  "quality_flags": ["list of specific issues found"],
  "drop_columns": [
    {{"column": "name", "reason": "identifier|zero_variance|too_sparse|too_much_missing"}}
  ],
  "impute": [
    {{"column": "name", "method": "regression|grouped_median|median|mean|mode|drop_rows", "group_by": ["col1"] or null, "regressors": ["col1"] or null}}
  ],
  "encode": [
    {{"column": "name", "method": "label|onehot|ordinal|target|frequency|binary|woe|hash", "categories": ["ordered list"] or null, "target_col": "optional target column for target/woe encoding"}}
  ],
  "transform": [
    {{"column": "name", "method": "log1p|sqrt|standard_scale|minmax_scale|boxcox|yeojohnson|robust_scale|power|quantile|rank"}}
  ],
  "engineer": [
    {{"name": "new_col_name", "operation": "sum|ratio|extract|bin|indicator|polynomial|datetime|log|multiply|groupby_agg|count|reciprocal", "source_columns": ["col1", "col2"], "params": {{}} }}
  ],
  "preprocessing_order": [
    {{"step": 1, "action": "drop|impute|encode|transform|engineer", "targets": ["col1", "col2"]}}
  ]
}}"""

    with llm_spinner("Building preprocessing plan"):
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

    from rich.table import Table
    from rich import box

    score = result.get("quality_score", "?")
    score_color = "green" if isinstance(score, int) and score >= 7 else "yellow" if isinstance(score, int) and score >= 4 else "red"
    console.print(f"     [bold {score_color}]Quality {score}/10[/bold {score_color}]")

    for flag in result.get("quality_flags", []):
        print_info(f"[yellow]⚠[/yellow]  {flag}")

    plan_table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    plan_table.add_column("Action")
    plan_table.add_column("Column / Feature")
    plan_table.add_column("Method / Detail")

    for d in result.get("drop_columns", []):
        plan_table.add_row("[red]drop[/red]", d["column"], f"[dim]{d['reason']}[/dim]")
    for imp in result.get("impute", []):
        group = f" by {imp['group_by']}" if imp.get("group_by") else ""
        plan_table.add_row("[cyan]impute[/cyan]", imp["column"], f"{imp['method']}{group}")
    for enc in result.get("encode", []):
        plan_table.add_row("[magenta]encode[/magenta]", enc["column"], enc["method"])
    for t in result.get("transform", []):
        plan_table.add_row("[blue]transform[/blue]", t["column"], t["method"])
    for e in result.get("engineer", []):
        plan_table.add_row("[green]engineer[/green]", e["name"], f"{e['operation']}({', '.join(e['source_columns'])})")

    console.print(plan_table)

    from src.report import narrate, add_section
    narrative = narrate("Preprocessing Plan", result, context="This is the LLM's recommended plan based on all profiling results.")
    add_section(state, "Preprocessing Plan", narrative)

    return state
