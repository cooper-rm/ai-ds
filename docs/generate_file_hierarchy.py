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
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Parses CLI args: --filepath, --goal, --name<BR ALIGN="LEFT"/>2. Calls create_state() to build the state dict<BR ALIGN="LEFT"/>3. Runs init_project() and ensure_env()<BR ALIGN="LEFT"/>4. Passes state to orchestrator()</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#f8d7da",
        color="#dc3545",
    )

# --- Layer 2: Core modules ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "STATE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">state.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Shared memory — persists to state.json</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">create_state() / load_state() / save_state()</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Creates fresh state or loads from JSON<BR ALIGN="LEFT"/>Saves after each node completes<BR ALIGN="LEFT"/>Keys: name, filepath, goal, project_dir,<BR ALIGN="LEFT"/>nodes, decisions, history</FONT></TD></TR>
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
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Dynamic pipeline with LLM decision points</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">orchestrator(state) → state</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">1. Runs required nodes in order<BR ALIGN="LEFT"/>2. At decision points (None), asks LLM<BR ALIGN="LEFT"/>3. LLM picks optional nodes to insert<BR ALIGN="LEFT"/>4. Skips completed nodes (error recovery)<BR ALIGN="LEFT"/>5. Saves state after each step</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#cce5ff",
        color="#0066cc",
    )

# --- Layer 3: LLM client ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "LLM",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="16">llm/client.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Claude CLI wrapper — uses subscription</FONT></TD></TR>
                <HR/>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" COLOR="#555555">ask(prompt, system) → str</FONT></TD></TR>
                <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#888888">Shells out to claude -p<BR ALIGN="LEFT"/>No API key needed</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#fff3cd",
        color="#d4a017",
    )

# --- Layer 4: Nodes ---
with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "INIT",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">init_project.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Creates project dir + state.json</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )
    s.node(
        "ENV",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">environment.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">Venv CRUD + ensure/re-launch</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "ANALYZE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">analyze_file.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">File type, size, memory estimate</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )
    s.node(
        "VALIDATE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">validate_file.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">LLM gate — checks for issues</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#ffe0cc",
        color="#e67e22",
    )

with dot.subgraph() as s:
    s.attr(rank="same")
    s.node(
        "LOAD",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">load_data.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">CSV/parquet/xlsx → state["data"]</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )
    s.node(
        "SUMMARIZE",
        label='''<
            <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
                <TR><TD><B><FONT POINT-SIZE="14">summarize.py</FONT></B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10" COLOR="#999999">dtypes, missing, numeric stats</FONT></TD></TR>
            </TABLE>
        >''',
        fillcolor="#d4edda",
        color="#28a745",
    )

# --- Edges ---
dot.edge("MAIN", "STATE")
dot.edge("MAIN", "ORCH")
dot.edge("MAIN", "INIT")
dot.edge("MAIN", "ENV")
dot.edge("ORCH", "ANALYZE")
dot.edge("ORCH", "VALIDATE")
dot.edge("ORCH", "LOAD")
dot.edge("ORCH", "SUMMARIZE")
dot.edge("VALIDATE", "LLM")
dot.edge("ORCH", "LLM", style="dashed")

dot.render("file_hierarchy", directory=".", cleanup=True)
print("Generated docs/file_hierarchy.png")
