import os
import subprocess


def save_and_show(fig, state: dict, filename: str) -> str:
    """Save a matplotlib figure to the project images dir and display with imgcat."""
    images_dir = os.path.join(state["project_dir"], "images")
    os.makedirs(images_dir, exist_ok=True)

    filepath = os.path.join(images_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    fig.clf()

    # Display inline with imgcat (iTerm2)
    try:
        subprocess.run(["imgcat", filepath], check=False)
    except FileNotFoundError:
        pass

    print(f"   Saved: {filepath}")
    return filepath
