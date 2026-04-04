"""
Interactive checkpoint node.

Pauses the pipeline after profiling is complete (before synthesis) to show
the user a summary of findings and ask for feedback. In non-interactive
mode (stdin is not a TTY), auto-continues without prompting.
"""

import sys
import json

from rich.table import Table
from rich.panel import Panel
from rich import box

from src.terminal import console, prompt_choice, print_info, print_detail, print_warning
from src.llm.client import ask
from src.terminal import llm_spinner


def _build_profile_snapshot(nodes: dict) -> dict:
    """Extract a compact snapshot of all profiling results for the LLM."""
    return {
        "dataset": {
            "row_count": nodes.get("load_data", {}).get("row_count"),
            "column_count": nodes.get("load_data", {}).get("column_count"),
            "memory_mb": nodes.get("optimize_dtypes", {}).get("after_mb")
                         or nodes.get("memory_analysis", {}).get("total_mb"),
        },
        "column_classifications": nodes.get("classify", {}).get("classifications", {}),
        "missing": nodes.get("summarize", {}).get("missing_pct", {}),
        "missing_analysis": {
            col: {
                "missing_pct": r.get("missing_pct"),
                "missingness_type": r.get("missingness_type"),
            }
            for col, r in nodes.get("imputation", {}).get("results", {}).items()
        },
        "columns_to_drop_missing": nodes.get("missing", {}).get("columns_to_drop", []),
        "type_issues": {
            "numeric_as_string": nodes.get("types", {}).get("numeric_as_string", []),
            "dates_as_string": nodes.get("types", {}).get("dates_as_string", []),
            "suggest_categorical": nodes.get("types", {}).get("suggest_categorical", []),
        },
        "anomalies": {
            "zero_variance": nodes.get("anomalies", {}).get("zero_variance", []),
            "extreme_skew": nodes.get("anomalies", {}).get("extreme_skew", []),
            "high_cardinality": nodes.get("anomalies", {}).get("high_cardinality", []),
        },
        "distributions": {
            col: {"skewness": v.get("skewness"), "shape": v.get("shape"),
                   "is_normal": v.get("normality", {}).get("is_normal")}
            for col, v in nodes.get("distributions", {}).get("per_column", {}).items()
            if "skewness" in v
        },
        "outliers": {
            col: {"n_outliers": r["n_outliers"], "outlier_pct": r["outlier_pct"]}
            for col, r in nodes.get("outliers", {}).get("results", {}).items()
            if r.get("n_outliers", 0) > 0
        },
        "correlations": {
            "top_pairs": nodes.get("correlations", {}).get("top_pairs", [])[:5],
        },
        "data_quality": {
            "exact_duplicates": nodes.get("data_quality", {}).get("exact_duplicates", {}).get("count", 0),
            "constant_columns": nodes.get("data_quality", {}).get("constant_columns", []),
        },
        "leakage": nodes.get("leakage", {}),
        "bivariate": {
            "significant_anova": [
                e for e in nodes.get("bivariate", {}).get("results", {}).get("categorical_x_numeric", [])
                if e.get("significant")
            ][:5],
        },
    }


def _get_executive_summary(snapshot: dict) -> str:
    """Ask the LLM for a 5-7 bullet executive summary of profiling results."""
    prompt = f"""You are reviewing profiling results for a dataset.

Profiling results:
{json.dumps(snapshot, indent=2)}

Produce a 5-7 bullet point executive summary of the MOST IMPORTANT findings,
surprises, and recommendations. Be specific — reference actual column names,
percentages, and values. Each bullet should be one concise sentence.

Format: one bullet per line, starting with "- ".
No preamble, no conclusion — just the bullets."""

    with llm_spinner("Generating executive summary"):
        return ask(prompt, system="You are a data science advisor. Be concise and specific.")


def _extract_key_metrics(nodes: dict) -> dict:
    """Pull key metrics from node results for the summary table."""
    metrics = {}

    # Row count, column count, memory
    metrics["row_count"] = nodes.get("load_data", {}).get("row_count", "?")
    metrics["col_count"] = nodes.get("load_data", {}).get("column_count", "?")
    memory = nodes.get("optimize_dtypes", {}).get("after_mb") \
             or nodes.get("memory_analysis", {}).get("total_mb")
    metrics["memory"] = f"{memory:.1f} MB" if memory else "?"

    # Missing columns + highest missing %
    missing_pct = nodes.get("summarize", {}).get("missing_pct", {})
    missing_cols = {k: v for k, v in missing_pct.items() if v > 0}
    metrics["missing_col_count"] = len(missing_cols)
    if missing_cols:
        worst_col = max(missing_cols, key=missing_cols.get)
        metrics["highest_missing"] = f"{worst_col} ({missing_cols[worst_col]:.1f}%)"
    else:
        metrics["highest_missing"] = "none"

    # Outlier columns + most affected
    outlier_results = nodes.get("outliers", {}).get("results", {})
    outlier_cols = {col: r["outlier_pct"] for col, r in outlier_results.items()
                    if r.get("n_outliers", 0) > 0}
    metrics["outlier_col_count"] = len(outlier_cols)
    if outlier_cols:
        worst_outlier = max(outlier_cols, key=outlier_cols.get)
        metrics["most_outliers"] = f"{worst_outlier} ({outlier_cols[worst_outlier]:.1f}%)"
    else:
        metrics["most_outliers"] = "none"

    # Normality
    dist_per_col = nodes.get("distributions", {}).get("per_column", {})
    normal_count = sum(1 for v in dist_per_col.values()
                       if v.get("normality", {}).get("is_normal"))
    non_normal_count = sum(1 for v in dist_per_col.values()
                          if "normality" in v and not v["normality"].get("is_normal"))
    metrics["normality"] = f"{normal_count} normal / {non_normal_count} non-normal"

    # Top correlation pair
    top_pairs = nodes.get("correlations", {}).get("top_pairs", [])
    if top_pairs:
        p = top_pairs[0]
        col1 = p.get("col1", p.get("column_1", "?"))
        col2 = p.get("col2", p.get("column_2", "?"))
        corr = p.get("correlation", p.get("pearson", "?"))
        if isinstance(corr, float):
            metrics["top_correlation"] = f"{col1} ↔ {col2} ({corr:+.3f})"
        else:
            metrics["top_correlation"] = f"{col1} ↔ {col2} ({corr})"
    else:
        metrics["top_correlation"] = "none computed"

    # Target balance
    leakage_node = nodes.get("leakage", {})
    target_col = leakage_node.get("target_column")
    classify_node = nodes.get("classify", {}).get("classifications", {})
    if target_col and target_col in classify_node:
        col_info = classify_node[target_col]
        if col_info.get("type") in ("binary", "categorical_nominal", "categorical_ordinal"):
            value_counts = col_info.get("value_counts", {})
            if value_counts:
                total = sum(value_counts.values())
                parts = [f"{k}: {v/total*100:.0f}%" for k, v in
                         sorted(value_counts.items(), key=lambda x: -x[1])[:3]]
                metrics["target_balance"] = f"{target_col} — {', '.join(parts)}"
            else:
                metrics["target_balance"] = f"{target_col} (details unavailable)"
        else:
            metrics["target_balance"] = f"{target_col} (continuous)"
    else:
        metrics["target_balance"] = "no target detected"

    # Leakage warnings
    metrics["leakage_warnings"] = leakage_node.get("critical_count", 0) \
                                  + leakage_node.get("warning_count", 0)

    return metrics


def _display_metrics_table(metrics: dict):
    """Render a compact Rich Table of key metrics."""
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold dim",
        padding=(0, 2),
        title="[bold white]Key Metrics[/bold white]",
        title_style="bold",
    )
    table.add_column("Metric", style="white")
    table.add_column("Value", style="cyan")

    table.add_row("Rows", str(metrics["row_count"]))
    table.add_row("Columns", str(metrics["col_count"]))
    table.add_row("Memory", metrics["memory"])
    table.add_row("Missing columns", f"{metrics['missing_col_count']}  (worst: {metrics['highest_missing']})")
    table.add_row("Outlier columns", f"{metrics['outlier_col_count']}  (worst: {metrics['most_outliers']})")
    table.add_row("Normality", metrics["normality"])
    table.add_row("Top correlation", metrics["top_correlation"])
    table.add_row("Target balance", metrics["target_balance"])

    leakage_count = metrics["leakage_warnings"]
    leakage_style = "[bold red]" if leakage_count > 0 else ""
    leakage_end = "[/bold red]" if leakage_count > 0 else ""
    table.add_row("Leakage warnings", f"{leakage_style}{leakage_count}{leakage_end}")

    console.print()
    console.print(table)
    console.print()


def _dig_deeper(snapshot: dict, user_question: str) -> str:
    """Send the user's question + profiling data to the LLM for detailed analysis."""
    prompt = f"""A user is reviewing profiling results and wants to dig deeper.

Profiling results:
{json.dumps(snapshot, indent=2)}

User's question / area of interest:
{user_question}

Provide a detailed, specific analysis addressing their question. Reference actual
column names, values, and percentages from the profiling data. Be thorough but
concise. Use markdown formatting for readability."""

    with llm_spinner("Analyzing"):
        return ask(prompt, system="You are a data science advisor. Be specific and thorough.")


def checkpoint(state: dict) -> dict:
    """Interactive checkpoint — show profiling summary and collect user feedback."""
    nodes = state["nodes"]
    snapshot = _build_profile_snapshot(nodes)
    interactions = []

    # ── Non-interactive mode: auto-continue ──────────────────────────────
    if not sys.stdin.isatty():
        print_info("Non-interactive mode — auto-continuing")
        state["nodes"]["checkpoint"] = {
            "status": "auto_continued",
            "interactive": False,
            "interactions": [],
        }
        return state

    # ── Executive summary via LLM ────────────────────────────────────────
    summary_text = _get_executive_summary(snapshot)
    console.print()
    console.print(Panel(
        summary_text,
        title="[bold white]Profiling Summary[/bold white]",
        border_style="blue",
        padding=(1, 2),
    ))

    # ── Key metrics table ────────────────────────────────────────────────
    metrics = _extract_key_metrics(nodes)
    _display_metrics_table(metrics)

    # ── Feedback loop ────────────────────────────────────────────────────
    while True:
        choice = prompt_choice(
            "Checkpoint",
            "Profiling is complete. What would you like to do?",
            [
                ("c", "Continue with preprocessing (looks good)"),
                ("d", "Dig deeper on a specific area (tell me what)"),
                ("a", "Adjust the plan (I have domain knowledge to share)"),
            ],
        )

        if choice == "c":
            print_info("Continuing to preprocessing")
            interactions.append({"action": "continue"})
            break

        elif choice == "d":
            console.print("  [bold cyan]What would you like to explore?[/bold cyan]")
            user_question = console.input("  → ")
            interactions.append({"action": "dig_deeper", "question": user_question})

            analysis = _dig_deeper(snapshot, user_question)
            console.print()
            console.print(Panel(
                analysis,
                title="[bold white]Deep Dive[/bold white]",
                border_style="cyan",
                padding=(1, 2),
            ))
            # Loop back to prompt again

        elif choice == "a":
            console.print("  [bold yellow]Share your domain knowledge or adjustments:[/bold yellow]")
            user_input = console.input("  → ")
            interactions.append({"action": "adjust", "user_input": user_input})

            # Store adjustments for synthesis to consume
            if "research_context" not in state:
                state["research_context"] = {}
            state["research_context"]["user_adjustments"] = user_input

            print_info("Adjustments stored — synthesis will incorporate your input")
            break

    # ── Store checkpoint results ─────────────────────────────────────────
    state["nodes"]["checkpoint"] = {
        "status": "done",
        "interactive": True,
        "interactions": interactions,
    }

    return state
