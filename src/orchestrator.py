import json

from src.state import save_state
from src.llm.client import ask
from src.terminal import (
    print_step, print_skip, print_done, print_fail,
    print_phase, print_summary, llm_spinner, NODE_PHASES,
)
from src.nodes.intake.analyze_file import analyze_file
from src.nodes.intake.load_data import load_data
from src.nodes.profile.summarize import summarize
from src.nodes.profile.memory_analysis import memory_analysis
from src.nodes.profile.types import types
from src.nodes.profile.classify import classify
from src.nodes.profile.optimize_dtypes import optimize_dtypes
from src.nodes.profile.structure import structure
from src.nodes.profile.anomalies import anomalies
from src.nodes.profile.missing import missing
from src.nodes.imputation.imputation import imputation
from src.nodes.profile.distributions import distributions
from src.nodes.profile.outliers import outliers
from src.nodes.profile.synthesis import synthesis
from src.nodes.profile.finalize_report import finalize_report
from src.nodes.preprocessing.drop_columns import drop_columns
from src.nodes.preprocessing.impute import impute
from src.nodes.preprocessing.engineer import engineer
from src.nodes.preprocessing.encode import encode
from src.nodes.preprocessing.transform import transform


# Required steps that always run, with decision points marked by None
pipelines = {
    "eda": [analyze_file,
            load_data,
            summarize,
            memory_analysis,
            types,
            classify,
            optimize_dtypes,
            structure,
            anomalies,
            missing,
            imputation,
            distributions,
            outliers,
            synthesis,
            drop_columns,
            impute,
            engineer,
            encode,
            transform,
            finalize_report,
            None],
}

# Optional nodes the LLM can insert at decision points
optional_nodes = {
    # Add optional nodes here as we build them
    # "correlations": correlations,
    # "outlier_detection": outlier_detection,
    # "visualize": visualize,
}


def decide_next(state: dict) -> list:
    """Ask the LLM if any optional steps should run at this decision point."""
    if not optional_nodes:
        return []

    node_descriptions = {name: f.__doc__ or name for name, f in optional_nodes.items()}
    last_completed = state["history"][-1] if state["history"] else "none"

    prompt = f"""You are orchestrating a data science pipeline.

The pipeline just completed the "{last_completed}" step.

Here is the current state of the pipeline:
{json.dumps({k: v for k, v in state["nodes"].items()}, indent=2)}

These optional steps are available:
{json.dumps(node_descriptions, indent=2)}

Based on the current state, should any of these optional steps run before continuing?

Respond with EXACTLY this JSON format, nothing else:
{{
  "run": ["list of optional step names to run, or empty list if none"],
  "reasoning": "brief explanation"
}}"""

    with llm_spinner("Deciding optional steps"):
        response = ask(prompt, system="You are a data science pipeline orchestrator. Respond only in JSON.")

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"run": [], "reasoning": "Could not parse LLM response"}

    return [optional_nodes[name] for name in result.get("run", []) if name in optional_nodes]


def run_step(state: dict, step) -> dict:
    """Run a single step with error recovery."""
    if step.__name__ in state["history"]:
        print_skip(step.__name__)
        return state

    # Print phase divider when entering a new phase
    current_phase = NODE_PHASES.get(step.__name__, "unknown")
    prev_phase = NODE_PHASES.get(state["history"][-1], "unknown") if state["history"] else None
    if current_phase != prev_phase:
        print_phase(current_phase)

    print_step(step.__name__)
    try:
        state = step(state)
        state["history"].append(step.__name__)
        save_state(state)
        node_data = state["nodes"].get(step.__name__, {})
        detail = node_data.get("status", "")
        print_done(step.__name__, detail)
    except Exception as e:
        state["nodes"][step.__name__] = {"status": "failed", "error": str(e)}
        save_state(state)
        print_fail(step.__name__, str(e))
        raise

    return state


def orchestrator(state: dict) -> dict:
    goal = state["goal"]

    if goal not in pipelines:
        raise ValueError(f"Unknown goal: {goal}. Available: {list(pipelines.keys())}")

    if "nodes" not in state:
        state["nodes"] = {}

    pipeline = pipelines[goal]

    for step in pipeline:
        if step is None:
            extras = decide_next(state)
            for extra in extras:
                state = run_step(state, extra)
        else:
            state = run_step(state, step)

    print_summary(state)
    return state
