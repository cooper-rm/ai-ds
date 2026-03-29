# ai-ds

Automated data science pipeline that evolves from deterministic agent functions to LLM-orchestrated workflows.

## How it works

```
python main.py --filepath data.csv --goal eda
```

`main.py` parses CLI args, builds a shared state dict, and passes it to an orchestrator. The orchestrator selects a pipeline of agents based on the goal and runs them in sequence. Each agent reads from and writes to state.

## Project structure

```
main.py                    # Entry point — CLI args → state → orchestrator
src/
├── state.py               # Shared state dict creation
├── orchestrator.py        # Goal → pipeline lookup → agent loop
└── agents/
    ├── load_data.py       # Reads CSV into state["data"]
    └── summarize.py       # Computes stats into state["summary"]
docs/
├── generate_file_hierarchy.py   # File/function hierarchy diagram
├── generate_flow.py             # Runtime flow diagram
├── file_hierarchy.png
└── flow.png
```

## Architecture

![File Hierarchy](docs/file_hierarchy.png)

## Runtime Flow

![Flow](docs/flow.png)
