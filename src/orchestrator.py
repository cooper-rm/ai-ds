from src.agents.load_data import load_data
from src.agents.summarize import summarize


pipelines = {
    "eda": [load_data, summarize],
}


def orchestrator(state: dict) -> dict:
    goal = state["goal"]

    if goal not in pipelines:
        raise ValueError(f"Unknown goal: {goal}. Available: {list(pipelines.keys())}")

    pipeline = pipelines[goal]

    for step in pipeline:
        print(f">> Running: {step.__name__}")
        state = step(state)
        state["history"].append(step.__name__)

    return state
