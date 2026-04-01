import argparse
import sys

from src.state import create_state
from src.nodes.init_project import init_project
from src.nodes.environment import ensure_env
from src.orchestrator import orchestrator


def main():
    parser = argparse.ArgumentParser(description="Automated data science pipeline")
    parser.add_argument("--filepath", required=True, help="Path to the input file")
    parser.add_argument("--goal", required=True, help="Goal to accomplish (e.g. eda)")
    parser.add_argument("--name", required=True, help="Project name")
    args = parser.parse_args()

    state = create_state(filepath=args.filepath, goal=args.goal, name=args.name)
    init_project(state)
    ensure_env(state, __file__, sys.argv[1:])
    state = orchestrator(state)


if __name__ == "__main__":
    main()
