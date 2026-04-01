import json
import os


def create_state(filepath: str, goal: str, name: str) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.join(repo_root, "projects", name)
    state_path = os.path.join(project_dir, "state.json")

    if os.path.exists(state_path):
        return load_state(state_path)

    return {
        "name": name,
        "filepath": filepath,
        "goal": goal,
        "project_dir": project_dir,
        "state_path": state_path,
        "data": None,
        "nodes": {},
        "decisions": [],
        "history": [],
    }


def load_state(state_path: str) -> dict:
    with open(state_path, "r") as f:
        state = json.load(f)
    state["data"] = None
    return state


def save_state(state: dict) -> None:
    serializable = {k: v for k, v in state.items() if k != "data" and not k.startswith("_")}
    with open(state["state_path"], "w") as f:
        json.dump(serializable, f, indent=2, default=str)
