import subprocess

from src.report import compile_pdf


def finalize_report(state: dict) -> dict:
    """Compile the report markdown into a PDF."""
    pdf_path = compile_pdf(state)

    if pdf_path:
        state["nodes"]["finalize_report"] = {
            "status": "generated",
            "pdf_path": pdf_path,
        }
        print(f"   Report: {pdf_path}")

        try:
            subprocess.run(["imgcat", pdf_path], check=False)
        except FileNotFoundError:
            pass
    else:
        state["nodes"]["finalize_report"] = {
            "status": "no_content",
        }
        print(f"   No report content to compile")

    return state
