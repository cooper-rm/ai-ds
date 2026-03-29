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

# --- Step 1: User runs command ---
dot.node(
    "CMD",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">1. CLI Input</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">User runs the program</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">python main.py --filepath data.csv --goal eda</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#f8d7da",
    color="#dc3545",
)

# --- Step 2: Parse args + build state ---
dot.node(
    "STATE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">2. Build State</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">create_state() initializes shared memory</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">state = {<BR ALIGN="LEFT"/>    "filepath": "data.csv",<BR ALIGN="LEFT"/>    "goal": "eda",<BR ALIGN="LEFT"/>    "data": None,<BR ALIGN="LEFT"/>    "summary": None,<BR ALIGN="LEFT"/>    "decisions": [],<BR ALIGN="LEFT"/>    "history": [],<BR ALIGN="LEFT"/>}</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#e2d9f3",
    color="#6f42c1",
)

# --- Step 3: Orchestrator picks pipeline ---
dot.node(
    "ORCH",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">3. Orchestrator Selects Pipeline</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Reads state["goal"] and looks up the agent sequence</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">goal = "eda"<BR ALIGN="LEFT"/>pipeline = [load_data, summarize]<BR ALIGN="LEFT"/>Begins looping through agents in order</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#cce5ff",
    color="#0066cc",
)

# --- Step 4: load_data ---
dot.node(
    "LOAD",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">4. Agent: load_data()</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">First agent in the pipeline</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Reads state["filepath"] → "data.csv"<BR ALIGN="LEFT"/>Runs pd.read_csv("data.csv")<BR ALIGN="LEFT"/>Stores DataFrame in state["data"]<BR ALIGN="LEFT"/>Prints: "Loaded 1000 rows, 12 columns"<BR ALIGN="LEFT"/>Returns state to orchestrator</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#d4edda",
    color="#28a745",
)

# --- Step 5: summarize ---
dot.node(
    "SUMMARIZE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">5. Agent: summarize()</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Second agent in the pipeline</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Reads DataFrame from state["data"]<BR ALIGN="LEFT"/>Computes shape, dtypes, missing counts<BR ALIGN="LEFT"/>Calculates missing value percentages<BR ALIGN="LEFT"/>Stores results in state["summary"]<BR ALIGN="LEFT"/>Returns state to orchestrator</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#d4edda",
    color="#28a745",
)

# --- Step 6: Done ---
dot.node(
    "DONE",
    label='''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
            <TR><TD><B><FONT POINT-SIZE="16">6. Pipeline Complete</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">All agents have run — state is fully populated</FONT></TD></TR>
            <HR/>
            <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">state["data"] → DataFrame with raw data<BR ALIGN="LEFT"/>state["summary"] → dict with shape, dtypes, missing<BR ALIGN="LEFT"/>state["history"] → ["load_data", "summarize"]<BR ALIGN="LEFT"/>Final state returned to main.py</FONT></TD></TR>
        </TABLE>
    >''',
    fillcolor="#fff3cd",
    color="#d4a017",
)

# --- Edges: runtime flow ---
dot.edge("CMD", "STATE")
dot.edge("STATE", "ORCH")
dot.edge("ORCH", "LOAD")
dot.edge("LOAD", "SUMMARIZE")
dot.edge("SUMMARIZE", "DONE")

dot.render("flow", directory=".", cleanup=True)
print("Generated docs/flow.png")
