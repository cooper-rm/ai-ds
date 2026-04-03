# ai-ds

Automated data science pipeline. Pass a CSV, get a profiled, preprocessed dataset and a PDF report — driven by LLM analysis at key decision points.

```
python3 -u main.py --filepath data.csv --goal eda --name my_project
```

---

## What it does

1. **Profiles** your dataset — dtypes, missing values, memory, type anomalies, structural issues, distributions, outliers, cardinality
2. **Classifies** every column by analytical type using Claude (continuous, categorical, identifier, text, etc.)
3. **Analyzes missingness** — classifies each missing column as MCAR/MAR/MNAR via statistical tests, evaluates up to 16 imputation methods per column, and selects the winner by minimizing regression line distortion (Δslope + ΔR²)
4. **Plans** a full preprocessing strategy — Claude reads all profiling results and returns a structured JSON plan
5. **Executes** that plan — drops junk columns, imputes missing values, engineers features, encodes categoricals, applies numeric transforms
6. **Versions** the dataset — saves a parquet snapshot with column-level diffs after every mutating step
7. **Reports** — every step narrated by Claude and compiled into a PDF with charts

---

## Architecture

### File + module hierarchy

![File Hierarchy](docs/file_hierarchy.png)

### Runtime pipeline flow

![Flow](docs/flow.png)

---

## Pipeline phases

### Setup
| Node | What it does |
|------|-------------|
| `init_project` | Creates `projects/name/{venv/ images/ data/ report/ state.json}` |
| `ensure_env` | Creates a per-project venv, installs `requirements.txt`, re-launches the process inside it |

### Intake
| Node | What it does |
|------|-------------|
| `analyze_file` | Detects file type, size, row/column count (without full load), estimates memory, installs file-specific deps |
| `load_data` | Reads file into `state["data"]` as a pandas DataFrame |

### Profile
| Node | What it does |
|------|-------------|
| `summarize` | dtypes, missing value counts + %, numeric stats (mean/std/min/max) |
| `memory_analysis` | Per-column memory breakdown, dtype downcast savings estimate |
| `types` | Detects numeric-as-string, date strings, low-cardinality columns worth converting |
| `classify` | **LLM** — classifies each column as continuous/discrete/categorical/identifier/binary/text/datetime |
| `optimize_dtypes` | Applies numeric downcasting (int64→int8/16/32, float64→float32). Strings untouched. |
| `structure` | Garbled characters, misaligned values, sentinel values (-999, N/A, null strings) |
| `anomalies` | Zero-variance columns, extreme skew (abs > 2), high-cardinality (>50% unique) |
| `missing` | Classifies missingness as MCAR/MAR/MNAR. Flags columns for drop (>50% missing) or imputation. Finds numeric/categorical correlates. |
| `imputation` | **LLM selects 3–5 methods per column** from 16 candidates. Runs all, scores each by regression line distortion, picks winner. Interactive gate for deep learning methods. |
| `distributions` | Per-column distribution plots (histogram + KDE + Q-Q). Records skewness, kurtosis, shape tag. |
| `outliers` | IQR (1.5×) + Z-score (>3) per numeric column. Box plot grid + strip plots for notable columns. |
| `synthesis` | **LLM** — reads all profiling results, returns structured JSON preprocessing plan |

### Preprocessing
All preprocessing nodes read from `state["nodes"]["synthesis"]` — the LLM plan drives execution.

| Node | What it does |
|------|-------------|
| `drop_columns` | Drops identifier, zero-variance, and overly sparse columns |
| `impute` | Applies the winning imputation method per column (from the imputation node). Saves versioned parquet snapshot. |
| `engineer` | Creates new features: `sum` / `ratio` / `indicator` / `bin` / `extract` |
| `encode` | Encodes categoricals: `label` / `onehot` / `ordinal`. Saves mappings to state. |
| `transform` | Numeric transforms: `log1p` / `sqrt` / `standard_scale` / `minmax_scale`. Generates before/after plots. |
| `finalize_report` | Compiles all report sections + charts into a PDF via PyLaTeX |

---

## Imputation system

The most sophisticated part of the pipeline. For each column with missing data:

1. **Missingness classification** (`missing.py`) — point-biserial correlation determines MCAR/MAR/MNAR. Finds top numeric and categorical correlates.
2. **Method selection** (`imputation/select.py`) — one LLM call for all columns. Claude picks 3–5 methods from the full menu based on missingness type, dtype, and correlate structure.
3. **Method evaluation** (`imputation/methods/`) — all selected methods run against the actual data.
4. **Scoring** (`imputation/score.py`) — each result is scored by fitting OLS on complete cases, refitting after imputation, and measuring `|Δslope|/|slope_orig| + |ΔR²|`. Categorical columns use Total Variation Distance instead.
5. **Winner selection** — lowest distortion score wins. Method results and plots are recorded in state.

### Available methods (16 total)

| Tier | Methods |
|------|---------|
| Simple | `mean`, `median`, `mode`, `grouped_median`, `knn` |
| Statistical | `regression`, `stochastic_regression`, `pmm`, `hotdeck`, `mice`, `missforest`, `em`, `softimpute` |
| Deep learning | `gain` (GAN), `mida` (denoising autoencoder), `hivae` (variational autoencoder) |

Deep learning methods require PyTorch and trigger an **interactive confirmation prompt** before running — the pipeline pauses and asks whether to continue with deep learning or fall back to statistical methods only.

---

## Key design decisions

**State dict as shared memory** — every node takes `state` and returns `state`. No globals, no class instances. The state is serialized to JSON after each node, so the pipeline can resume from any failure point.

**Error recovery** — `state["history"]` tracks completed nodes. Re-running the same command skips everything already done and resumes from the failed step.

**Dataset versioning** — `snapshot()` in `utils.py` saves a versioned parquet file (`v01_impute.parquet`, etc.) after each mutating step and diffs which columns were added, removed, or changed vs. the previous version. All snapshots live in `projects/name/data/`.

**Two LLM roles** — `classify` and `synthesis` use Claude as an *analyst* (structured JSON output). `narrate()` in `report.py` uses Claude as a *storyteller* (prose for the PDF). These are intentionally separate.

**LLM plans, code executes** — `synthesis` returns a machine-actionable JSON plan. The preprocessing nodes are dumb executors — they don't make decisions, they just apply whatever the plan says. This means you can inspect/edit the plan in `state.json` before re-running.

**Interactive gates** — `prompt_choice()` in `terminal.py` pauses the pipeline at high-stakes decisions. Currently used for deep learning imputation confirmation. Designed as a reusable pattern for future checkpoints (outlier capping, column dropping, etc.).

**Per-project venv** — each project gets its own isolated Python environment. `ensure_env` handles create/check/re-launch transparently.

**Dynamic decision points** — `None` entries in the pipeline list are decision points where Claude can insert optional nodes. Required nodes always run; optional nodes are LLM-gated.

---

## Project structure

```
main.py                                # CLI args → state → init → env → orchestrator
requirements.txt
src/
├── state.py                           # State creation, persistence (JSON)
├── orchestrator.py                    # Pipeline runner + LLM decision points
├── terminal.py                        # Rich terminal UI (colors, spinners, prompts, summary)
├── report.py                          # LLM narrator + PyLaTeX PDF compiler
├── utils.py                           # snapshot() — versioned parquet + column diffs
├── llm/
│   └── client.py                      # Claude CLI wrapper (claude -p)
└── nodes/
    ├── setup/
    │   ├── init_project.py
    │   └── environment.py
    ├── intake/
    │   ├── analyze_file.py
    │   └── load_data.py
    ├── profile/
    │   ├── summarize.py
    │   ├── memory_analysis.py
    │   ├── types.py
    │   ├── classify.py                # LLM
    │   ├── optimize_dtypes.py
    │   ├── structure.py
    │   ├── anomalies.py
    │   ├── missing.py                 # MCAR/MAR/MNAR classification
    │   ├── distributions.py           # Distribution plots + shape tags
    │   ├── outliers.py                # IQR + Z-score outlier detection
    │   └── synthesis.py              # LLM
    ├── imputation/
    │   ├── imputation.py              # Orchestrator — select, run, score, pick winner
    │   ├── select.py                  # LLM method selection
    │   ├── score.py                   # Regression line distortion scoring
    │   ├── plot.py                    # 4-panel comparison plots
    │   └── methods/
    │       ├── simple.py              # mean, median, mode, grouped_median, knn
    │       ├── statistical.py         # regression, pmm, hotdeck, mice, missforest, em, softimpute
    │       └── deep.py                # GAIN, MIDA, HI-VAE (PyTorch)
    └── preprocessing/
        ├── drop_columns.py
        ├── impute.py
        ├── engineer.py
        ├── encode.py
        ├── transform.py
        └── finalize_report.py
docs/
├── generate_flow.py                   # Regenerate flow.png
├── generate_file_hierarchy.py         # Regenerate file_hierarchy.png
├── flow.png
└── file_hierarchy.png
projects/                              # Per-run project dirs (gitignored)
templates/                             # Reference analysis templates
```

---

## Output

Each run produces `projects/<name>/`:
```
projects/my_project/
├── state.json          # Full pipeline state — inspect or edit between runs
├── data/               # Versioned parquet snapshots (v01_impute.parquet, v02_encode.parquet, …)
├── images/             # All generated plots (distributions, imputation comparisons, outliers…)
├── report/
│   └── report.pdf      # Final PDF report
└── venv/               # Isolated Python environment
```

---

## Resuming after failure

If a node fails, fix the issue and re-run the same command. The pipeline reads `state.json`, sees which nodes are in `history`, skips them, and picks up from the failed step.

To start completely fresh:
```
rm -rf projects/my_project && python3 -u main.py --filepath data.csv --goal eda --name my_project
```
