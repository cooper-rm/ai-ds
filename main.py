import argparse

from src.state import create_state
from src.orchestrator import orchestrator


def main():
    parser = argparse.ArgumentParser(description="Automated data science pipeline")
    parser.add_argument("--filepath", required=True, help="Path to the input file")
    parser.add_argument("--goal", required=True, help="Goal to accomplish (e.g. eda, preprocessing)")
    args = parser.parse_args()

    state = create_state(filepath=args.filepath, goal=args.goal)
    state = orchestrator(state)


if __name__ == "__main__":
    main()
