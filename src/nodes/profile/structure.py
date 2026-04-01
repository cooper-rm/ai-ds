import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import save_and_show


SENTINEL_STRINGS = {"N/A", "n/a", "NA", "null", "NULL", "None", "?", ".", "-", "missing", "unknown", "UNKNOWN", "not available", ""}
SENTINEL_NUMBERS = {-999, 9999, -1, 99999}


def structure(state: dict) -> dict:
    """Verify structural integrity — garbled chars, misalignment, header issues, sentinels."""
    df = state["data"]

    garbled = _check_garbled(df)
    misaligned = _check_misaligned(df)
    header_issues = _check_headers(df)
    sentinels = _check_sentinels(df)

    has_issues = garbled["found"] or misaligned["found"] or header_issues["found"] or sentinels["found"]

    state["nodes"]["structure"] = {
        "status": "issues_found" if has_issues else "clean",
        "garbled": garbled,
        "misaligned": misaligned,
        "header_issues": header_issues,
        "sentinels": sentinels,
        "images": [],
    }

    # Generate sentinel heatmap if sentinels found
    if sentinels["found"]:
        images = _plot_sentinels(df, sentinels, state)
        state["nodes"]["structure"]["images"] = images

    # Print summary
    print(f"   Encoding: {'garbled in ' + str(len(garbled['columns'])) + ' columns' if garbled['found'] else 'clean'}")
    print(f"   Alignment: {'issues found' if misaligned['found'] else 'OK'}")
    print(f"   Headers: {'issues found' if header_issues['found'] else 'OK'}")
    print(f"   Sentinels: {'found in ' + str(len(sentinels['columns'])) + ' columns' if sentinels['found'] else 'none'}")

    if sentinels["found"]:
        for col, vals in sentinels["values_found"].items():
            print(f"     {col}: {vals}")

    from src.report import narrate, add_section
    narrative = narrate("Structural Integrity", {
        "encoding": "clean" if not garbled["found"] else f"garbled in {len(garbled['columns'])} columns",
        "alignment": "OK" if not misaligned["found"] else "issues found",
        "headers": "OK" if not header_issues["found"] else "issues found",
        "sentinels": "none" if not sentinels["found"] else f"found in {len(sentinels['columns'])} columns",
    })
    add_section(state, "Structural Integrity", narrative, images if sentinels["found"] else None)

    return state


def _check_garbled(df) -> dict:
    """Check string columns for encoding artifacts."""
    garbled_patterns = re.compile(r'[Ã©Ã¼Ã¢â€™\x00-\x08\x0b\x0c\x0e-\x1f]')
    columns = []
    samples = {}

    for col in df.select_dtypes(include="object").columns:
        bad = df[col].dropna().astype(str).str.contains(garbled_patterns, regex=True)
        if bad.any():
            columns.append(col)
            samples[col] = list(df[col][bad].head(3).values)

    return {"found": bool(columns), "columns": columns, "samples": samples}


def _check_misaligned(df) -> dict:
    """Check if values appear shifted between columns."""
    details = []

    for col in df.select_dtypes(include="number").columns:
        str_vals = df[col].dropna().astype(str)
        alpha_count = str_vals.str.contains(r'[a-zA-Z]', regex=True).sum()
        if alpha_count > 0:
            details.append({"column": col, "issue": f"{alpha_count} alphabetic values in numeric column"})

    return {"found": bool(details), "details": details}


def _check_headers(df) -> dict:
    """Check if data rows contain repeated headers."""
    details = []
    col_names = set(df.columns)

    for idx in range(min(5, len(df))):
        row_vals = set(str(v) for v in df.iloc[idx].values)
        overlap = row_vals & col_names
        if len(overlap) > len(df.columns) * 0.5:
            details.append({"row": idx, "matching_columns": list(overlap)})

    return {"found": bool(details), "details": details}


def _check_sentinels(df) -> dict:
    """Scan for common sentinel/placeholder values."""
    columns = []
    values_found = {}

    for col in df.columns:
        found = {}

        if df[col].dtype == "object":
            val_counts = df[col].astype(str).str.strip().value_counts()
            for sentinel in SENTINEL_STRINGS:
                if sentinel in val_counts.index:
                    found[sentinel] = int(val_counts[sentinel])
        else:
            for sentinel in SENTINEL_NUMBERS:
                count = (df[col] == sentinel).sum()
                if count > 0:
                    found[str(sentinel)] = int(count)

        if found:
            columns.append(col)
            values_found[col] = found

    return {"found": bool(columns), "columns": columns, "values_found": values_found}


def _plot_sentinels(df, sentinels, state) -> list:
    """Plot sentinel value distribution."""
    images = []

    cols = sentinels["columns"]
    vals = sentinels["values_found"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cols) * 0.5)))

    y_labels = []
    x_values = []
    for col in cols:
        total = sum(vals[col].values())
        y_labels.append(f"{col}")
        x_values.append(total)

    colors = ["#C44E52" if v > 10 else "#CCB974" for v in x_values]
    ax.barh(y_labels, x_values, color=colors)
    ax.set_title("Sentinel Values by Column", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")

    plt.tight_layout()
    path = save_and_show(fig, state, "sentinels.png")
    images.append(path)
    plt.close()

    return images
