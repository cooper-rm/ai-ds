"""
Terminal UI utilities — colors, spinners, and styled output.
Uses `rich` for all formatting. Import these instead of raw print().
"""

import time
from contextlib import contextmanager

from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from rich.rule import Rule

console = Console()

# Phase colors — each phase of the pipeline gets a distinct color
PHASE_COLORS = {
    "setup":         "bright_black",
    "intake":        "cyan",
    "profile":       "blue",
    "preprocessing": "green",
    "unknown":       "white",
}

# Node → phase mapping
NODE_PHASES = {
    "init_project":   "setup",
    "ensure_env":     "setup",
    "analyze_file":   "intake",
    "load_data":      "intake",
    "validate_file":  "intake",
    "summarize":      "profile",
    "memory_analysis":"profile",
    "types":          "profile",
    "classify":       "profile",
    "optimize_dtypes":"profile",
    "structure":      "profile",
    "anomalies":      "profile",
    "missing":        "profile",
    "imputation":     "profile",
    "distributions":  "profile",
    "outliers":       "profile",
    "synthesis":      "profile",
    "drop_columns":   "preprocessing",
    "impute":         "preprocessing",
    "engineer":       "preprocessing",
    "encode":         "preprocessing",
    "transform":      "preprocessing",
    "finalize_report":"preprocessing",
}


def _phase_color(node_name: str) -> str:
    phase = NODE_PHASES.get(node_name, "unknown")
    return PHASE_COLORS[phase]


def _phase_label(node_name: str) -> str:
    return NODE_PHASES.get(node_name, "unknown").upper()


# ── Pipeline banner ──────────────────────────────────────────────────────────

def print_banner(name: str, filepath: str, goal: str):
    """Print the startup banner."""
    console.print()
    console.print(Panel.fit(
        f"[bold white]ai-ds[/bold white]  [dim]—[/dim]  [bold cyan]{name}[/bold cyan]\n"
        f"[dim]{filepath}[/dim]  [dim]·[/dim]  [dim]{goal.upper()}[/dim]",
        border_style="bright_black",
        padding=(0, 2),
    ))
    console.print()


# ── Step header ──────────────────────────────────────────────────────────────

def print_step(node_name: str):
    """Print a styled header when a node starts running."""
    color = _phase_color(node_name)
    phase = _phase_label(node_name)
    label = node_name.replace("_", " ")

    console.print(
        f"  [bold {color}]▸[/bold {color}]  "
        f"[bold white]{label}[/bold white]  "
        f"[dim]{phase}[/dim]"
    )


def print_skip(node_name: str):
    """Print a skip notice for already-completed nodes."""
    label = node_name.replace("_", " ")
    console.print(f"  [dim]↩  {label}  already done[/dim]")


def print_done(node_name: str, detail: str = ""):
    """Print a success line after a node finishes."""
    color = _phase_color(node_name)
    suffix = f"  [dim]{detail}[/dim]" if detail else ""
    console.print(f"  [bold {color}]✓[/bold {color}]  [dim]{node_name.replace('_', ' ')}[/dim]{suffix}")


def print_fail(node_name: str, error: str):
    """Print an error line when a node fails."""
    console.print(f"  [bold red]✗[/bold red]  [red]{node_name.replace('_', ' ')}[/red]")
    console.print(f"     [dim red]{error}[/dim red]")
    console.print(f"     [dim]Fix the issue and re-run to resume from this step.[/dim]")


def print_info(msg: str):
    """Print a plain indented info line."""
    console.print(f"     [dim]{msg}[/dim]")


def print_detail(key: str, value):
    """Print a key/value detail line."""
    console.print(f"     [dim]{key}:[/dim] [white]{value}[/white]")


def print_warning(msg: str):
    console.print(f"     [yellow]⚠[/yellow]  [yellow]{msg}[/yellow]")


# ── Phase dividers ───────────────────────────────────────────────────────────

def print_phase(phase: str):
    """Print a section rule between pipeline phases."""
    color = PHASE_COLORS.get(phase.lower(), "white")
    console.print()
    console.rule(f"[bold {color}]{phase.upper()}[/bold {color}]", style=color)
    console.print()


# ── LLM spinner ──────────────────────────────────────────────────────────────

@contextmanager
def llm_spinner(label: str = "Thinking"):
    """Context manager that shows a spinner while an LLM call runs."""
    with console.status(
        f"[bold cyan]{label}…[/bold cyan]",
        spinner="dots",
        spinner_style="cyan",
    ):
        yield


# ── Interactive prompt ────────────────────────────────────────────────────────

def prompt_choice(title: str, body: str, options: list[tuple[str, str]]) -> str:
    """
    Display a styled interactive prompt and return the chosen key.

    options: list of (key, label) e.g. [("y", "Yes, continue"), ("n", "Take easier path")]
    Returns the key the user selected.
    """
    console.print()
    console.print(Panel(
        f"[bold white]{title}[/bold white]\n\n{body}",
        border_style="yellow",
        padding=(1, 2),
    ))
    for key, label in options:
        console.print(f"  [bold yellow]{key}[/bold yellow]  {label}")
    console.print()

    keys = {k.lower() for k, _ in options}
    while True:
        try:
            raw = input("  → ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = options[-1][0].lower()   # default to last option on non-interactive
        if raw in keys:
            console.print()
            return raw
        console.print(f"  [dim]Please enter one of: {', '.join(k for k, _ in options)}[/dim]")


# ── Pipeline summary ─────────────────────────────────────────────────────────

def print_summary(state: dict):
    """Print a final pipeline summary table."""
    history = state.get("history", [])
    nodes = state.get("nodes", {})

    console.print()
    console.rule("[bold white]Pipeline Complete[/bold white]", style="bright_black")
    console.print()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Node", style="white")
    table.add_column("Phase", style="dim")
    table.add_column("Status", justify="right")

    for node_name in history:
        phase = _phase_label(node_name)
        color = _phase_color(node_name)
        node_data = nodes.get(node_name, {})
        status = node_data.get("status", "done")

        if "fail" in status:
            badge = "[bold red]✗ failed[/bold red]"
        else:
            badge = f"[{color}]✓ {status}[/{color}]"

        table.add_row(node_name.replace("_", " "), phase, badge)

    console.print(table)

    # Show output path if report was generated
    project_dir = state.get("project_dir", "")
    if project_dir:
        import os
        report_path = os.path.join(project_dir, "report", "report.pdf")
        if os.path.exists(report_path):
            console.print(f"  [bold green]Report →[/bold green] [dim]{report_path}[/dim]")

    console.print()
