import os
import shutil
import subprocess
import sys


def _get_paths(project_dir: str) -> tuple:
    """Get venv path and python path from project directory."""
    venv_path = os.path.join(project_dir, "venv")
    venv_python = os.path.join(venv_path, "bin", "python")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    requirements = os.path.join(repo_root, "requirements.txt")
    return venv_path, venv_python, requirements


def create_env(state: dict) -> dict:
    """Create venv inside project dir and install requirements."""
    project_dir = state["project_dir"]
    venv_path, venv_python, requirements = _get_paths(project_dir)

    if os.path.exists(venv_path):
        print(f"   Venv already exists")
    else:
        print(f"   Creating virtual environment")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

    if os.path.exists(requirements):
        print(f"   Installing requirements")
        subprocess.run([venv_python, "-m", "pip", "install", "-r", requirements, "-q"], check=True)

    state["_venv_python"] = venv_python
    return state


def check_env(project_dir: str) -> bool:
    """Check if the venv exists and has a valid python."""
    venv_path, venv_python, _ = _get_paths(project_dir)
    return os.path.exists(venv_path) and os.path.exists(venv_python)


def running_in_env(project_dir: str) -> bool:
    """Check if we're currently running inside the target venv."""
    venv_path, _, _ = _get_paths(project_dir)
    return sys.prefix == venv_path


def delete_env(project_dir: str) -> None:
    """Delete the venv directory."""
    venv_path, _, _ = _get_paths(project_dir)
    if os.path.exists(venv_path):
        print(f"   Deleting virtual environment")
        shutil.rmtree(venv_path)
    else:
        print(f"   Venv not found")


def ensure_env(state: dict, script_path: str, args: list) -> None:
    """Create env if needed, re-launch inside it if not already there."""
    project_dir = state["project_dir"]

    if running_in_env(project_dir):
        return

    from src.state import save_state

    state = create_env(state)
    venv_path, venv_python, _ = _get_paths(project_dir)

    state["nodes"]["environment"] = {
        "venv_path": venv_path,
        "venv_python": venv_python,
    }

    state["history"].append("environment")
    save_state(state)

    print(f">> Re-launching inside venv", flush=True)
    result = subprocess.run([state["_venv_python"], script_path] + args)
    sys.exit(result.returncode)
