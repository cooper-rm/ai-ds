"""
File + module hierarchy diagram.
Run from repo root: python3 docs/generate_file_hierarchy.py
"""
import graphviz  # type: ignore
import os

dot = graphviz.Digraph(
    "file_hierarchy",
    format="png",
    graph_attr={
        "bgcolor": "#0f1117",
        "rankdir": "LR",
        "splines": "ortho",
        "nodesep": "0.4",
        "ranksep": "2.0",
        "pad": "1.0",
        "dpi": "200",
        "fontname": "Helvetica Neue",
    },
    node_attr={
        "style": "filled,rounded",
        "shape": "box",
        "fontname": "Helvetica Neue",
        "penwidth": "0",
        "margin": "0.15,0.1",
    },
    edge_attr={
        "color": "#2e3440",
        "arrowsize": "0.6",
        "penwidth": "1.2",
        "arrowhead": "vee",
    },
)

# ── color system ──────────────────────────────────────────────────────────────
COLORS = {
    "entry":      ("#1c1c1c", "#555555", "#eeeeee"),
    "core":       ("#1a1528", "#7c6af7", "#c4bdff"),
    "llm":        ("#2b2210", "#d4a017", "#f5d98a"),
    "setup":      ("#1e1b2e", "#9d8df5", "#ccc6ff"),
    "intake":     ("#0d2137", "#1a8fe3", "#7ec8f7"),
    "profile":    ("#0f2318", "#1db954", "#6ee89b"),
    "llm_node":   ("#2b2210", "#d4a017", "#f5d98a"),
    "preprocess": ("#2a1a0e", "#e07b39", "#f5c49a"),
    "util":       ("#181c24", "#4a5568", "#9aa5b4"),
}

def node(dot_obj, nid, title, subtitle, color_key, funcs=None):
    bg, border, txt = COLORS[color_key]
    func_row = ""
    if funcs:
        func_list = "<BR ALIGN=\"LEFT\"/>".join(f"<FONT COLOR=\"{txt}\" OPACITY=\"70\">{f}</FONT>" for f in funcs)
        func_row = f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{func_list}</FONT></TD></TR>'
    label = f'''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="4">
            <TR><TD><B><FONT POINT-SIZE="12" COLOR="{txt}">{title}</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9" COLOR="{txt}" OPACITY="75">{subtitle}</FONT></TD></TR>
            {func_row}
        </TABLE>
    >'''
    dot_obj.node(nid, label=label, fillcolor=bg, color=border, penwidth="1.5")

# ── ENTRY ─────────────────────────────────────────────────────────────────────
node(dot, "MAIN", "main.py", "Entry point", "entry",
     ["main()  →  parse args → state → init → env → orchestrator"])

# ── CORE ──────────────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_core") as s:
    s.attr(label="src/  core", fontcolor="#7c6af7", fontsize="10", fontname="Helvetica Neue",
           color="#7c6af7", style="rounded,dashed", bgcolor="#110e1c", penwidth="1")
    node(s, "STATE", "state.py", "Shared state — persists to JSON", "core",
         ["create_state()  load_state()  save_state()"])
    node(s, "ORCH",  "orchestrator.py", "Dynamic pipeline runner", "core",
         ["orchestrator()  run_step()  decide_next()"])
    node(s, "REPORT","report.py", "LLM narrator + PDF compiler", "core",
         ["narrate()  add_section()  compile_pdf()"])
    node(s, "TERM",  "terminal.py", "Rich terminal UI", "util",
         ["print_step()  print_done()  print_fail()  llm_spinner()  print_summary()"])
    node(s, "UTILS", "utils.py", "Shared helpers", "util",
         ["save_and_show()  — save PNG + imgcat display"])

node(dot, "LLM", "llm/client.py", "Claude CLI wrapper", "llm",
     ["ask(prompt, system) → str", "Shells out to: claude -p"])

# ── SETUP nodes ───────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_setup") as s:
    s.attr(label="nodes/setup/", fontcolor="#9d8df5", fontsize="10", fontname="Helvetica Neue",
           color="#9d8df5", style="rounded,dashed", bgcolor="#13111e", penwidth="1")
    node(s, "INIT", "init_project.py", "Creates project dir structure", "setup",
         ["init_project(state)"])
    node(s, "ENV",  "environment.py",  "Venv CRUD + re-launch", "setup",
         ["ensure_env()  create_venv()  check_venv()  delete_venv()"])

# ── INTAKE nodes ──────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_intake") as s:
    s.attr(label="nodes/intake/", fontcolor="#1a8fe3", fontsize="10", fontname="Helvetica Neue",
           color="#1a8fe3", style="rounded,dashed", bgcolor="#091624", penwidth="1")
    node(s, "ANALYZE", "analyze_file.py", "File inspection before load", "intake",
         ["analyze_file(state)  →  type · size · rows · memory"])
    node(s, "LOADD",   "load_data.py",    "Read file → DataFrame", "intake",
         ["load_data(state)  →  state[\"data\"]"])

# ── PROFILE nodes ─────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_profile") as s:
    s.attr(label="nodes/profile/", fontcolor="#1db954", fontsize="10", fontname="Helvetica Neue",
           color="#1db954", style="rounded,dashed", bgcolor="#091a0f", penwidth="1")
    node(s, "SUMM",  "summarize.py",      "dtypes · missing · numeric stats", "profile")
    node(s, "MEM",   "memory_analysis.py","Per-column memory + savings estimate", "profile")
    node(s, "TYPES", "types.py",          "Detect type mismatches + low cardinality", "profile")
    node(s, "CLASS", "classify.py",       "⚡ LLM: column analytical type", "llm_node")
    node(s, "OPT",   "optimize_dtypes.py","Downcast numeric dtypes", "profile")
    node(s, "STRUCT","structure.py",      "Garbled chars · sentinel values", "profile")
    node(s, "ANOM",  "anomalies.py",      "Skew · zero variance · high cardinality", "profile")
    node(s, "SYNTH", "synthesis.py",      "⚡ LLM: full preprocessing plan", "llm_node")

# ── PREPROCESSING nodes ───────────────────────────────────────────────────────
with dot.subgraph(name="cluster_preprocess") as s:
    s.attr(label="nodes/preprocessing/", fontcolor="#e07b39", fontsize="10", fontname="Helvetica Neue",
           color="#e07b39", style="rounded,dashed", bgcolor="#1e1208", penwidth="1")
    node(s, "DROP",  "drop_columns.py","Drop flagged columns", "preprocess")
    node(s, "IMP",   "impute.py",      "Fill missing values + plots", "preprocess")
    node(s, "ENG",   "engineer.py",    "Create new features", "preprocess")
    node(s, "ENC",   "encode.py",      "Encode categoricals", "preprocess")
    node(s, "TRANS", "transform.py",   "Numeric transforms + plots", "preprocess")
    node(s, "FIN",   "finalize_report.py","Compile PDF report", "preprocess")

# ── EDGES ─────────────────────────────────────────────────────────────────────
dot.edge("MAIN",  "STATE")
dot.edge("MAIN",  "ORCH")
dot.edge("MAIN",  "INIT")
dot.edge("MAIN",  "ENV")
dot.edge("ORCH",  "LLM",   style="dashed", color="#d4a017")
dot.edge("ORCH",  "TERM")
dot.edge("ORCH",  "ANALYZE")
dot.edge("ORCH",  "LOADD")
dot.edge("ORCH",  "SUMM")
dot.edge("ORCH",  "MEM")
dot.edge("ORCH",  "TYPES")
dot.edge("ORCH",  "CLASS")
dot.edge("ORCH",  "OPT")
dot.edge("ORCH",  "STRUCT")
dot.edge("ORCH",  "ANOM")
dot.edge("ORCH",  "SYNTH")
dot.edge("ORCH",  "DROP")
dot.edge("ORCH",  "IMP")
dot.edge("ORCH",  "ENG")
dot.edge("ORCH",  "ENC")
dot.edge("ORCH",  "TRANS")
dot.edge("ORCH",  "FIN")
dot.edge("CLASS", "LLM",   style="dashed", color="#d4a017")
dot.edge("SYNTH", "LLM",   style="dashed", color="#d4a017")
dot.edge("FIN",   "REPORT")
dot.edge("IMP",   "UTILS")
dot.edge("TRANS", "UTILS")

out_dir = os.path.join(os.path.dirname(__file__))
dot.render("file_hierarchy", directory=out_dir, cleanup=True)
print(f"Generated {out_dir}/file_hierarchy.png")
