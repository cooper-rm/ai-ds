import graphviz  # type: ignore

dot = graphviz.Digraph(
    "flow",
    format="png",
    graph_attr={
        "bgcolor": "#ffffff",
        "rankdir": "TB",
        "splines": "line",
        "nodesep": "0.8",
        "ranksep": "1.2",
        "pad": "1.0",
        "dpi": "200",
    },
    node_attr={
        "style": "filled,rounded",
        "shape": "box",
        "fontname": "Helvetica",
        "penwidth": "1.5",
    },
    edge_attr={
        "color": "#b0b0b0",
        "arrowsize": "0.8",
        "penwidth": "1.5",
    },
)

# --- Step 1: CLI ---
dot.node(
    "CMD",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">1. CLI Input</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">User runs the program</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">python main.py --filepath data.csv --goal eda --name my_project</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#f8d7da",
    color="#dc3545",
)

# --- Step 2: Init project ---
dot.node(
    "INIT",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">2. Init Project</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Creates project directory and state.json</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">projects/my_project/<BR ALIGN="LEFT"/>  ├── state.json<BR ALIGN="LEFT"/>  └── venv/</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#e2d9f3",
    color="#6f42c1",
)

# --- Step 3: Environment ---
dot.node(
    "ENV",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">3. Environment</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Creates venv, installs deps, re-launches inside it</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Saves state → re-launches with venv python<BR ALIGN="LEFT"/>Child process loads state from JSON</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#e2d9f3",
    color="#6f42c1",
)

# --- Step 4: Analyze file ---
dot.node(
    "ANALYZE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">4. Analyze File</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Inspect file before loading</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Checks file type, size, row/column count<BR ALIGN="LEFT"/>Estimates memory requirements<BR ALIGN="LEFT"/>Installs deps if needed (e.g. openpyxl)<BR ALIGN="LEFT"/>Writes to state["nodes"]["analyze_file"]</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#d4edda",
    color="#28a745",
)

# --- Step 5: Validate file (LLM gate) ---
dot.node(
    "VALIDATE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">5. Validate File (LLM Gate)</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Claude reviews the file analysis</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Sends analyze_file results to Claude<BR ALIGN="LEFT"/>Claude checks for issues and red flags<BR ALIGN="LEFT"/>Returns: proceed (true/false), issues, recommendations<BR ALIGN="LEFT"/>Blocks pipeline if proceed = false</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#ffe0cc",
    color="#e67e22",
)

# --- Step 6: Load data ---
dot.node(
    "LOAD",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">6. Load Data</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Read file into DataFrame</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">pd.read_csv → state["data"]<BR ALIGN="LEFT"/>Records row_count, column_count, memory_mb</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#d4edda",
    color="#28a745",
)

# --- Step 7: Summarize ---
dot.node(
    "SUMMARIZE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">7. Summarize</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Compute dataset statistics</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">dtypes, missing values, missing %<BR ALIGN="LEFT"/>Numeric stats (mean, std, min, max)<BR ALIGN="LEFT"/>Writes to state["nodes"]["summarize"]</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#d4edda",
    color="#28a745",
)

# --- Step 8: Decision point ---
dot.node(
    "DECIDE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">8. Decision Point</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">LLM decides if optional nodes should run</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Claude reads current state<BR ALIGN="LEFT"/>Picks from available optional nodes<BR ALIGN="LEFT"/>Or continues to next required step</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#fff3cd",
    color="#d4a017",
)

# --- Edges ---
dot.edge("CMD", "INIT")
dot.edge("INIT", "ENV")
dot.edge("ENV", "ANALYZE")
dot.edge("ANALYZE", "VALIDATE")
dot.edge("VALIDATE", "LOAD")
dot.edge("LOAD", "SUMMARIZE")
dot.edge("SUMMARIZE", "DECIDE")

dot.render("flow", directory=".", cleanup=True)
print("Generated docs/flow.png")
