def create_state(filepath: str, goal: str) -> dict:
    return {
        "filepath": filepath,
        "goal": goal,
        "data": None,
        "summary": None,
        "decisions": [],
        "history": [],
    }
