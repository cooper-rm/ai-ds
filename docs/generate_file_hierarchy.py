import graphviz  # type: ignore

dot = graphviz.Digraph(
    "file_hierarchy",
    format="png",
    graph_attr={
        "bgcolor": "#ffffff",
        "rankdir": "TB",
        "splines": "line",
        "nodesep": "1.5",
        "ranksep": "1.5",
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

# --- Layer 1: Entry point ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "MAIN",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">main.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Entry point — where everything starts</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">main()</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Parses CLI args: --filepath and --goal<BR ALIGN="LEFT"/>2. Calls create_state() to build the state dict<BR ALIGN="LEFT"/>3. Passes state to orchestrator() to run the pipeline</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#f8d7da",
        color="#dc3545",
    )

# --- Layer 2: What main calls ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "STATE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">state.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Shared memory — the dict every agent reads and writes</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">create_state(filepath, goal) → dict</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Returns a dict with these keys:<BR ALIGN="LEFT"/>  • filepath — path to the input CSV<BR ALIGN="LEFT"/>  • goal — what pipeline to run (e.g. "eda")<BR ALIGN="LEFT"/>  • data — None until load_data fills it<BR ALIGN="LEFT"/>  • summary — None until summarize fills it<BR ALIGN="LEFT"/>  • decisions — list of choices made by agents<BR ALIGN="LEFT"/>  • history — list of agent names that have run</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#e2d9f3",
        color="#6f42c1",
    )
    s.node(
        "ORCH",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">orchestrator.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Decides which agents to run and in what order</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">orchestrator(state) → state</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Reads state["goal"] to pick a pipeline<BR ALIGN="LEFT"/>2. Pipelines are lists of agent functions:<BR ALIGN="LEFT"/>     "eda" → [load_data, summarize]<BR ALIGN="LEFT"/>3. Loops through each agent in order<BR ALIGN="LEFT"/>4. Passes state through each one<BR ALIGN="LEFT"/>5. Later: LLM will replace this with reasoning</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#cce5ff",
        color="#0066cc",
    )

# --- Layer 3: Agents ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "LOAD",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">load_data.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Agent — loads raw data into state</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">load_data(state) → state</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Reads state["filepath"]<BR ALIGN="LEFT"/>2. Uses pd.read_csv to load the file<BR ALIGN="LEFT"/>3. Stores the DataFrame in state["data"]<BR ALIGN="LEFT"/>4. Prints row and column count</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )
    s.node(
        "SUMMARIZE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">summarize.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Agent — computes dataset overview stats</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">summarize(state) → state</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Reads the DataFrame from state["data"]<BR ALIGN="LEFT"/>2. Computes shape, column types, missing values<BR ALIGN="LEFT"/>3. Calculates missing value percentages<BR ALIGN="LEFT"/>4. Stores everything in state["summary"]</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )

# --- Edges: call chain ---
dot.edge("MAIN", "STATE")
dot.edge("MAIN", "ORCH")
dot.edge("ORCH", "LOAD")
dot.edge("ORCH", "SUMMARIZE")

dot.render("file_hierarchy", directory=".", cleanup=True)
print("Generated docs/file_hierarchy.png")
