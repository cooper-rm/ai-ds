import os
import json
from datetime import datetime

from src.llm.client import ask


def narrate(section_name: str, data: dict, context: str = "") -> str:
    """Ask the narrator LLM to write a report section."""
    prompt = f"""You are narrating a data science report for a non-technical stakeholder.

Section: {section_name}
{f"Context: {context}" if context else ""}

Data:
{json.dumps(data, indent=2, default=str)}

Write 2-4 sentences explaining what happened in this step, what was found, and why it matters.
Be clear, concise, and professional. No jargon without explanation. No markdown formatting.
Do not use any special characters like curly braces, backslashes, percent signs, ampersands, or underscores."""

    return ask(prompt, system="You are a data science report narrator. Write clear, concise prose. No markdown. No special characters.")


def add_section(state: dict, title: str, body: str, images: list = None) -> None:
    """Add a section to the report sections list in state."""
    if "report_sections" not in state:
        state["report_sections"] = []

    state["report_sections"].append({
        "title": title,
        "body": body,
        "images": images or [],
    })


def compile_pdf(state: dict) -> str:
    """Compile report sections into a polished PDF using PyLaTeX."""
    from pylatex import Document, Section, Command, NoEscape, Package
    from pylatex.utils import bold

    report_dir = os.path.join(state["project_dir"], "report")
    report_path = os.path.join(report_dir, "report")

    sections = state.get("report_sections", [])
    if not sections:
        return ""

    # Document setup
    geometry = {"margin": "1in"}
    doc = Document(geometry_options=geometry)
    doc.packages.append(Package("graphicx"))
    doc.packages.append(Package("xcolor"))
    doc.packages.append(Package("titlesec"))
    doc.packages.append(Package("parskip"))
    doc.packages.append(Package("float"))

    # Colors and styling
    doc.preamble.append(NoEscape(r"\definecolor{accent}{RGB}{52, 119, 235}"))
    doc.preamble.append(NoEscape(r"\definecolor{darktext}{RGB}{40, 40, 40}"))
    doc.preamble.append(NoEscape(r"\definecolor{lighttext}{RGB}{120, 120, 120}"))
    doc.preamble.append(NoEscape(r"\titleformat{\section}{\Large\bfseries\color{darktext}}{}{0em}{}[\color{accent}\titlerule]"))
    doc.preamble.append(NoEscape(r"\color{darktext}"))

    # Title page
    project_title = state.get("name", "Project").replace("_", " ").title()
    doc.preamble.append(Command("title", NoEscape(
        r"{\color{accent}\rule{\linewidth}{1pt}} \\\vspace{0.5cm} "
        r"{\Huge\bfseries " + _escape(project_title) + r"} \\\vspace{0.3cm} "
        r"{\Large\color{lighttext} Data Science Pipeline Report} \\\vspace{0.3cm} "
        r"{\color{accent}\rule{\linewidth}{1pt}}"
    )))
    doc.preamble.append(Command("author", ""))

    date_str = datetime.now().strftime("%B %d, %Y")
    file_path = state.get("filepath", "")
    row_count = state.get("nodes", {}).get("load_data", {}).get("row_count", "")
    col_count = state.get("nodes", {}).get("load_data", {}).get("column_count", "")

    date_line = date_str
    if file_path:
        date_line += r" \\ {\color{lighttext}\small Source: " + _escape(file_path) + "}"
    if row_count and col_count:
        date_line += r" \\ {\color{lighttext}\small " + f"{row_count} rows x {col_count} columns" + "}"

    doc.preamble.append(Command("date", NoEscape(date_line)))
    doc.append(NoEscape(r"\maketitle"))
    doc.append(NoEscape(r"\thispagestyle{empty}"))
    doc.append(NoEscape(r"\newpage"))

    # Content sections
    for sec in sections:
        with doc.create(Section(sec["title"])):
            doc.append(NoEscape(_escape(sec["body"])))

            for img_path in sec.get("images", []):
                if os.path.exists(img_path):
                    doc.append(NoEscape(r"\begin{figure}[H]"))
                    doc.append(NoEscape(r"\centering"))
                    doc.append(NoEscape(r"\includegraphics[width=\textwidth]{" + img_path + "}"))
                    doc.append(NoEscape(r"\end{figure}"))

    # Generate PDF
    doc.generate_pdf(report_path, clean_tex=True, compiler="pdflatex")

    return report_path + ".pdf"


def _escape(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text
