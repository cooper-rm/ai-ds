import subprocess


def ask(prompt: str, system: str = "") -> str:
    """Send a prompt to Claude via the CLI. Uses your Claude Code subscription."""
    cmd = ["claude", "-p"]

    if system:
        prompt = f"[System: {system}]\n\n{prompt}"

    cmd.append(prompt)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    return result.stdout.strip()
