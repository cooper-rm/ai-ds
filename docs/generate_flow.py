"""
Runtime flow diagram — full pipeline with all phases and nodes.
Run from repo root: python3 docs/generate_flow.py
"""
import graphviz  # type: ignore
import os

dot = graphviz.Digraph(
    "flow",
    format="png",
    graph_attr={
        "bgcolor": "#0f1117",
        "rankdir": "TB",
        "splines": "ortho",
        "nodesep": "0.6",
        "ranksep": "1.0",
        "pad": "1.2",
        "dpi": "200",
        "fontname": "Helvetica Neue",
    },
    node_attr={
        "style": "filled,rounded",
        "shape": "box",
        "fontname": "Helvetica Neue",
        "penwidth": "0",
        "margin": "0.2,0.12",
    },
    edge_attr={
        "color": "#3a3f4b",
        "arrowsize": "0.7",
        "penwidth": "1.5",
        "arrowhead": "vee",
    },
)

# ── color palette ─────────────────────────────────────────────────────────────
C = {
    "setup":      ("#1e1b2e", "#7c6af7", "#b0a8ff"),   # bg, border, text
    "intake":     ("#0d2137", "#1a8fe3", "#7ec8f7"),
    "profile":    ("#0f2318", "#1db954", "#6ee89b"),
    "preprocess": ("#2a1a0e", "#e07b39", "#f5c49a"),
    "llm":        ("#2b2210", "#d4a017", "#f5d98a"),
    "decision":   ("#1e2535", "#6b7fad", "#aab6d4"),
    "cli":        ("#1c1c1c", "#555555", "#cccccc"),
}

def phase_node(dot, node_id, title, subtitle, phase, details=None):
    bg, border, txt = C[phase]
    detail_row = f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{txt}" OPACITY="70">{details}</FONT></TD></TR>' if details else ""
    label = f'''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="5">
            <TR><TD><B><FONT POINT-SIZE="13" COLOR="{txt}">{title}</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9" COLOR="{txt}" OPACITY="80">{subtitle}</FONT></TD></TR>
            {detail_row}
        </TABLE>
    >'''
    dot.node(node_id, label=label, fillcolor=bg, color=border, penwidth="1.5", fontcolor=txt)

def llm_node(dot, node_id, title, subtitle):
    bg, border, txt = C["llm"]
    label = f'''<
        <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="5">
            <TR><TD><B><FONT POINT-SIZE="11" COLOR="{txt}">⚡ {title}</FONT></B></TD></TR>
            <TR><TD><FONT POINT-SIZE="9" COLOR="{txt}" OPACITY="80">{subtitle}</FONT></TD></TR>
        </TABLE>
    >'''
    dot.node(node_id, label=label, fillcolor=bg, color=border, penwidth="1.5")

# ── CLI ───────────────────────────────────────────────────────────────────────
bg, border, txt = C["cli"]
dot.node("CLI", label=f'''<
    <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="6">
        <TR><TD><B><FONT POINT-SIZE="14" COLOR="{txt}">python main.py</FONT></B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9" COLOR="#888888">--filepath data.csv  --goal eda  --name my_project</FONT></TD></TR>
    </TABLE>
>''', fillcolor=bg, color=border, penwidth="1.5")

# ── SETUP phase ───────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_setup") as s:
    s.attr(label="SETUP", fontcolor="#7c6af7", fontsize="11", fontname="Helvetica Neue",
           color="#7c6af7", style="rounded,dashed", bgcolor="#13111e", penwidth="1")
    phase_node(s, "INIT", "init_project", "Create project dir + subdirs", "setup",
               "projects/name/{venv/ images/ data/ report/ state.json}")
    phase_node(s, "ENV", "ensure_env", "Venv CRUD + re-launch inside it", "setup",
               "Creates venv → pip install -r requirements.txt → re-exec")

# ── INTAKE phase ──────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_intake") as s:
    s.attr(label="INTAKE", fontcolor="#1a8fe3", fontsize="11", fontname="Helvetica Neue",
           color="#1a8fe3", style="rounded,dashed", bgcolor="#091624", penwidth="1")
    phase_node(s, "ANALYZE", "analyze_file", "Inspect file before loading", "intake",
               "File type · size · row/col estimate · memory · install deps")
    phase_node(s, "LOAD", "load_data", "Read file into DataFrame", "intake",
               "pd.read_csv → state[\"data\"]  ·  records shape + memory_mb")

# ── PROFILE phase ─────────────────────────────────────────────────────────────
with dot.subgraph(name="cluster_profile") as s:
    s.attr(label="PROFILE", fontcolor="#1db954", fontsize="11", fontname="Helvetica Neue",
           color="#1db954", style="rounded,dashed", bgcolor="#091a0f", penwidth="1")
    phase_node(s, "SUMMARIZE",   "summarize",      "dtypes · missing · numeric stats", "profile")
    phase_node(s, "MEMORY",      "memory_analysis","Per-column memory + dtype savings estimate", "profile")
    phase_node(s, "TYPES",       "types",          "Detect numeric-as-string · date strings · low cardinality", "profile")
    phase_node(s, "CLASSIFY",    "classify",       "LLM classifies each column by analytical type", "llm",
               "continuous / discrete / categorical / identifier / text / binary / datetime")
    phase_node(s, "OPTIMIZE",    "optimize_dtypes","Downcast numeric columns (int64→int8, float64→float32)", "profile")
    phase_node(s, "STRUCTURE",   "structure",      "Garbled chars · misalignment · sentinels (-999, N/A…)", "profile")
    phase_node(s, "ANOMALIES",   "anomalies",      "Zero variance · extreme skew · high cardinality", "profile")
    phase_node(s, "SYNTHESIS",   "synthesis",      "LLM builds full preprocessing plan from profiling results", "llm",
               "drop · impute · encode · transform · engineer  →  structured JSON")

# ── PREPROCESSING phase ───────────────────────────────────────────────────────
with dot.subgraph(name="cluster_preprocess") as s:
    s.attr(label="PREPROCESSING", fontcolor="#e07b39", fontsize="11", fontname="Helvetica Neue",
           color="#e07b39", style="rounded,dashed", bgcolor="#1e1208", penwidth="1")
    phase_node(s, "DROP",     "drop_columns","Drop identifier/zero-variance/sparse columns", "preprocess")
    phase_node(s, "IMPUTE",   "impute",      "Fill missing: median/mean/mode/grouped_median/drop_rows", "preprocess")
    phase_node(s, "ENGINEER", "engineer",    "Create features: sum/ratio/indicator/bin/extract", "preprocess")
    phase_node(s, "ENCODE",   "encode",      "Encode categoricals: label/onehot/ordinal", "preprocess")
    phase_node(s, "TRANSFORM","transform",   "Numeric transforms: log1p/sqrt/standard_scale/minmax_scale", "preprocess")
    phase_node(s, "REPORT",   "finalize_report","Compile all sections → PDF via PyLaTeX", "preprocess")

# ── DECISION point ────────────────────────────────────────────────────────────
bg, border, txt = C["decision"]
dot.node("DECIDE", label=f'''<
    <TABLE BORDER="0" CELLSPACING="0" CELLPADDING="5">
        <TR><TD><B><FONT POINT-SIZE="11" COLOR="{txt}">⬡ Decision Point</FONT></B></TD></TR>
        <TR><TD><FONT POINT-SIZE="9" COLOR="{txt}" OPACITY="80">LLM picks optional nodes to insert</FONT></TD></TR>
    </TABLE>
>''', fillcolor=bg, color=border, penwidth="1.5", shape="diamond")

# ── EDGES ─────────────────────────────────────────────────────────────────────
# Setup
dot.edge("CLI",      "INIT")
dot.edge("INIT",     "ENV")
dot.edge("ENV",      "ANALYZE")

# Intake
dot.edge("ANALYZE",  "LOAD")

# Profile
dot.edge("LOAD",     "SUMMARIZE")
dot.edge("SUMMARIZE","MEMORY")
dot.edge("MEMORY",   "TYPES")
dot.edge("TYPES",    "CLASSIFY",  color="#d4a017", style="dashed")
dot.edge("CLASSIFY", "OPTIMIZE")
dot.edge("OPTIMIZE", "STRUCTURE")
dot.edge("STRUCTURE","ANOMALIES")
dot.edge("ANOMALIES","SYNTHESIS", color="#d4a017", style="dashed")

# Preprocessing
dot.edge("SYNTHESIS","DROP")
dot.edge("DROP",     "IMPUTE")
dot.edge("IMPUTE",   "ENGINEER")
dot.edge("ENGINEER", "ENCODE")
dot.edge("ENCODE",   "TRANSFORM")
dot.edge("TRANSFORM","REPORT")
dot.edge("REPORT",   "DECIDE")

out_dir = os.path.join(os.path.dirname(__file__))
dot.render("flow", directory=out_dir, cleanup=True)
print(f"Generated {out_dir}/flow.png")
