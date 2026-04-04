"""
Interview node — interactive intake that captures user research goals.

Runs after load_data. Asks the user about their target column, research goal,
domain context, and key questions, then asks the LLM to synthesize everything
into a structured research context for downstream nodes.
"""

import json
import sys

import pandas as pd

from src.llm.client import ask
from src.terminal import print_info, print_detail, print_warning, prompt_choice, console
from src.terminal import llm_spinner

# Column names that strongly suggest a target variable
TARGET_NAME_HINTS = {
    "target", "label", "y", "class", "outcome",
    "survived", "churn", "default", "fraud",
}


def _is_interactive() -> bool:
    """Return True if stdin is attached to a terminal (not piped)."""
    return sys.stdin.isatty()


def _detect_target_candidates(df: pd.DataFrame) -> list[dict]:
    """
    Heuristic scan for likely target columns.

    Returns a list of dicts with 'column', 'reason', and 'unique_values'.
    """
    candidates = []

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue

        nunique = series.nunique()
        reasons = []

        # Check name hints
        col_lower = col.lower().strip()
        if col_lower in TARGET_NAME_HINTS:
            reasons.append("name matches common target pattern")

        # Binary columns (0/1, yes/no, true/false)
        if nunique == 2:
            vals = set(str(v).lower() for v in series.unique())
            binary_pairs = [{"0", "1"}, {"yes", "no"}, {"true", "false"}]
            if vals in binary_pairs or nunique == 2:
                reasons.append("binary column (2 unique values)")

        # Low cardinality numeric
        if pd.api.types.is_numeric_dtype(series) and 2 <= nunique <= 10:
            if "binary column (2 unique values)" not in reasons:
                reasons.append(f"low-cardinality numeric ({nunique} unique values)")

        if reasons:
            candidates.append({
                "column": col,
                "reason": "; ".join(reasons),
                "unique_values": nunique,
            })

    # Sort: name-matched first, then by cardinality
    candidates.sort(key=lambda c: (
        "name matches" not in c["reason"],
        c["unique_values"],
    ))

    return candidates


def _build_data_summary(df: pd.DataFrame) -> dict:
    """Build a concise data summary for the LLM prompt."""
    summary = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        col_info = {
            "dtype": str(series.dtype),
            "nunique": int(series.nunique()),
            "null_count": int(series.isnull().sum()),
            "sample_values": [str(v) for v in non_null.head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
            col_info["min"] = float(series.min())
            col_info["max"] = float(series.max())
            col_info["mean"] = round(float(series.mean()), 2)

        summary["columns"][col] = col_info

    return summary


def interview(state: dict) -> dict:
    """Interview the user about research goals and synthesize with LLM."""
    df = state["data"]
    interactive = _is_interactive()

    # ── 1. Detect target column candidates ────────────────────────────────────
    candidates = _detect_target_candidates(df)

    if interactive and candidates:
        options = []
        body_lines = ["Detected possible target columns:\n"]
        for i, c in enumerate(candidates):
            key = str(i + 1)
            label = f"[bold]{c['column']}[/bold]  [dim]({c['reason']})[/dim]"
            body_lines.append(f"  {key}. {c['column']} — {c['reason']}")
            options.append((key, label))
        options.append(("o", "Other — I'll type the column name"))
        options.append(("n", "None — no target column"))

        choice = prompt_choice(
            title="Target Column",
            body="\n".join(body_lines),
            options=options,
        )

        if choice == "n":
            target_column = None
        elif choice == "o":
            target_column = console.input(
                "  [bold yellow]Column name:[/bold yellow] "
            ).strip() or None
            if target_column and target_column not in df.columns:
                print_warning(f"'{target_column}' not found in data — setting target to None")
                target_column = None
        else:
            idx = int(choice) - 1
            target_column = candidates[idx]["column"]
    elif interactive:
        print_info("no obvious target column detected")
        raw = console.input(
            "  [bold yellow]Enter target column name (or press Enter for none):[/bold yellow] "
        ).strip()
        if raw and raw in df.columns:
            target_column = raw
        else:
            if raw:
                print_warning(f"'{raw}' not found in data — setting target to None")
            target_column = None
    else:
        # Non-interactive: pick the best candidate or None
        target_column = candidates[0]["column"] if candidates else None
        print_info(
            f"non-interactive mode — auto-selected target: {target_column or 'none'}"
        )

    # ── 2. Research goal ──────────────────────────────────────────────────────
    if interactive:
        research_goal = prompt_choice(
            title="Research Goal",
            body="What are you trying to do with this data?",
            options=[
                ("predict", "Building a predictive model"),
                ("understand", "Understanding patterns and relationships"),
                ("report", "Generating a summary report for stakeholders"),
                ("explore", "Open-ended exploration, not sure yet"),
            ],
        )
    else:
        research_goal = "explore"
        print_info("non-interactive mode — defaulting goal to 'explore'")

    # ── 3. Domain context ─────────────────────────────────────────────────────
    if interactive:
        domain_context = console.input(
            "\n  [bold yellow]Any domain context that would help analysis?[/bold yellow]\n"
            "  [dim](e.g., 'medical patient data', 'financial transactions', "
            "press Enter to skip)[/dim]\n  → "
        ).strip()
    else:
        domain_context = ""

    # ── 4. Key questions ──────────────────────────────────────────────────────
    if interactive:
        key_questions = console.input(
            "\n  [bold yellow]What specific questions are you trying to answer?[/bold yellow]\n"
            "  [dim](press Enter to skip)[/dim]\n  → "
        ).strip()
    else:
        key_questions = ""

    # ── 5. LLM synthesis ──────────────────────────────────────────────────────
    data_summary = _build_data_summary(df)

    llm_prompt = f"""You are a data science advisor. A user has loaded a dataset and answered intake questions about their goals. Analyze their responses together with the data summary and produce a structured research context.

User responses:
- Target column: {target_column or "none specified"}
- Research goal: {research_goal}
- Domain context: {domain_context or "not provided"}
- Key questions: {key_questions or "not provided"}

Data summary:
{json.dumps(data_summary, indent=2)}

Return ONLY a JSON object (no markdown fences, no commentary) with these keys:
- "target_column": the confirmed target column name as a string, or null if none
- "analysis_priorities": a list of 3-6 strings describing what the analysis should focus on given the user's goals and data
- "domain_insights": a list of 1-4 strings with domain-specific analysis recommendations (or empty list if no domain context)
- "risk_flags": a list of 1-4 strings noting things to watch out for based on the domain and data (or empty list if unclear)
"""

    print_info("synthesizing research context")
    with llm_spinner("Analyzing research goals"):
        raw_response = ask(llm_prompt, system="You are a data science advisor. Respond with valid JSON only.")

    # Parse LLM response
    try:
        # Strip markdown fences if the LLM included them
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        synthesis = json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        print_warning("LLM returned invalid JSON — using defaults")
        synthesis = {
            "target_column": target_column,
            "analysis_priorities": ["general exploratory analysis"],
            "domain_insights": [],
            "risk_flags": [],
        }

    # ── 6. Store results ──────────────────────────────────────────────────────
    interview_result = {
        "status": "complete",
        "target_column": synthesis.get("target_column", target_column),
        "research_goal": research_goal,
        "domain_context": domain_context or None,
        "key_questions": key_questions or None,
        "analysis_priorities": synthesis.get("analysis_priorities", []),
        "domain_insights": synthesis.get("domain_insights", []),
        "risk_flags": synthesis.get("risk_flags", []),
        "candidates_detected": [c["column"] for c in candidates],
    }

    state["nodes"]["interview"] = interview_result

    # Top-level research_context for easy downstream access
    state["research_context"] = {
        "target_column": interview_result["target_column"],
        "research_goal": research_goal,
        "domain_context": domain_context or None,
        "key_questions": key_questions or None,
        "analysis_priorities": interview_result["analysis_priorities"],
        "domain_insights": interview_result["domain_insights"],
        "risk_flags": interview_result["risk_flags"],
    }

    # Print summary
    print_detail("target", str(interview_result["target_column"] or "none"))
    print_detail("goal", research_goal)
    if interview_result["analysis_priorities"]:
        for p in interview_result["analysis_priorities"]:
            print_detail("priority", p)
    if interview_result["risk_flags"]:
        for r in interview_result["risk_flags"]:
            print_detail("risk", r)

    return state
