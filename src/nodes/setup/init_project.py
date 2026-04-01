import os

from src.state import save_state


def init_project(state: dict) -> dict:
    """Create the project directory and save initial state as JSON."""
    project_dir = state["project_dir"]

    if os.path.exists(state["state_path"]):
        return state

    print(f"   Creating project: {state['name']}")
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "report"), exist_ok=True)

    state["nodes"]["init_project"] = {
        "status": "created",
    }

    state["history"].append("init_project")
    save_state(state)
    return state
