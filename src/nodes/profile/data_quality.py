"""
Data quality checks node.

Checks for: exact duplicates, identifier column duplicates, string
inconsistencies (mixed casing, whitespace), range validation, and
constant columns.  Produces a summary bar chart of issue counts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.report import narrate, add_section
from src.utils import save_and_show
from src.terminal import print_info, print_detail, print_warning


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

def data_quality(state: dict) -> dict:
    """Run data quality checks and report findings."""
    df = state["data"]
    images = []

    # 1. Exact duplicate rows
    exact_dupes = _check_exact_duplicates(df)

    # 2. Subset duplicates on identifier columns
    classifications = (
        state.get("nodes", {})
        .get("classify", {})
        .get("classifications", {})
    )
    identifier_dupes = _check_identifier_duplicates(df, classifications)

    # 3. String inconsistencies
    string_issues = _check_string_inconsistencies(df)

    # 4. Range validation
    range_issues = _check_range_validation(df)

    # 5. Constant columns
    constant_cols = _check_constant_columns(df)

    # ---- terminal output ----
    _print_findings(
        exact_dupes, identifier_dupes, string_issues, range_issues, constant_cols,
    )

    # ---- summary visualization ----
    images += _plot_summary(
        exact_dupes, identifier_dupes, string_issues, range_issues, constant_cols,
        state,
    )

    # ---- persist to state ----
    state["nodes"]["data_quality"] = {
        "status": "issues_found" if any([
            exact_dupes["count"],
            identifier_dupes,
            string_issues,
            range_issues,
            constant_cols,
        ]) else "clean",
        "exact_duplicates": exact_dupes,
        "identifier_duplicates": identifier_dupes,
        "string_issues": string_issues,
        "range_issues": range_issues,
        "constant_columns": constant_cols,
        "images": images,
    }

    # ---- report ----
    narrative = narrate("Data Quality", {
        "exact_duplicate_count": exact_dupes["count"],
        "exact_duplicate_pct": exact_dupes["pct"],
        "identifier_duplicates": identifier_dupes,
        "string_issues_count": len(string_issues),
        "range_issues_count": len(range_issues),
        "constant_columns_count": len(constant_cols),
    })
    add_section(state, "Data Quality", narrative, images)

    return state


# ---------------------------------------------------------------------------
# 1. Exact duplicate rows
# ---------------------------------------------------------------------------

def _check_exact_duplicates(df: pd.DataFrame) -> dict:
    mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated(keep="first").sum())
    dup_pct = round(dup_count / len(df) * 100, 2) if len(df) > 0 else 0.0

    examples = []
    if dup_count > 0:
        dup_rows = df[mask].head(10)
        examples = dup_rows.to_dict(orient="records")[:5]

    return {"count": dup_count, "pct": dup_pct, "examples": examples}


# ---------------------------------------------------------------------------
# 2. Identifier column duplicates
# ---------------------------------------------------------------------------

def _check_identifier_duplicates(
    df: pd.DataFrame, classifications: dict,
) -> list:
    results = []
    id_cols = [
        col for col, info in classifications.items()
        if info.get("type") == "identifier" and col in df.columns
    ]
    for col in id_cols:
        n_dupes = int(df[col].dropna().duplicated().sum())
        if n_dupes > 0:
            results.append({
                "column": col,
                "duplicate_count": n_dupes,
                "sample_duplicates": (
                    df[col][df[col].duplicated(keep=False)]
                    .value_counts()
                    .head(5)
                    .to_dict()
                ),
            })
    return results


# ---------------------------------------------------------------------------
# 3. String inconsistencies
# ---------------------------------------------------------------------------

def _check_string_inconsistencies(df: pd.DataFrame) -> list:
    results = []
    str_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        issues: dict = {}
        series = df[col].dropna()
        if series.empty:
            continue

        # Mixed casing -------------------------------------------------------
        lower = series.str.lower()
        grouped = series.groupby(lower).apply(
            lambda g: list(g.unique())
        )
        mixed = {k: v for k, v in grouped.items() if len(v) > 1}
        if mixed:
            issues["mixed_casing"] = {
                str(k): v for k, v in list(mixed.items())[:5]
            }

        # Leading / trailing whitespace --------------------------------------
        ws_mask = series != series.str.strip()
        ws_count = int(ws_mask.sum())
        if ws_count > 0:
            issues["whitespace"] = {
                "count": ws_count,
                "examples": series[ws_mask].head(5).tolist(),
            }

        if issues:
            issues["column"] = col
            results.append(issues)

    return results


# ---------------------------------------------------------------------------
# 4. Range validation
# ---------------------------------------------------------------------------

_PCT_KEYWORDS = ("pct", "percent", "rate", "ratio")
_AGE_KEYWORDS = ("age",)


def _check_range_validation(df: pd.DataFrame) -> list:
    results = []
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        col_lower = col.lower()
        cmin = float(series.min())
        cmax = float(series.max())

        # Percentage-like columns
        if any(kw in col_lower for kw in _PCT_KEYWORDS):
            if cmax > 100 or cmin < 0:
                if not (0 <= cmin and cmax <= 1):
                    results.append({
                        "column": col,
                        "issue": "possible percentage outside expected range",
                        "min": round(cmin, 4),
                        "max": round(cmax, 4),
                    })

        # Age-like columns
        if any(kw in col_lower for kw in _AGE_KEYWORDS):
            flagged = False
            if cmin < 0:
                flagged = True
            if cmax > 120:
                flagged = True
            if flagged:
                results.append({
                    "column": col,
                    "issue": "possible age outside expected range [0, 120]",
                    "min": round(cmin, 4),
                    "max": round(cmax, 4),
                })

    return results


# ---------------------------------------------------------------------------
# 5. Constant columns
# ---------------------------------------------------------------------------

def _check_constant_columns(df: pd.DataFrame) -> list:
    results = []
    for col in df.columns:
        if df[col].nunique() == 1:
            val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            results.append({"column": col, "constant_value": str(val)})
    return results


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _print_findings(exact_dupes, identifier_dupes, string_issues,
                    range_issues, constant_cols):
    # Exact duplicates
    print_info(
        f"exact duplicate rows: {exact_dupes['count']} "
        f"({exact_dupes['pct']}% of dataset)"
    )
    if exact_dupes["count"] > 0:
        print_warning(f"{exact_dupes['count']} exact duplicate rows detected")

    # Identifier duplicates
    if identifier_dupes:
        for item in identifier_dupes:
            print_warning(
                f"identifier column '{item['column']}' has "
                f"{item['duplicate_count']} duplicate values"
            )
    else:
        print_info("no identifier column duplicates")

    # String issues
    if string_issues:
        for item in string_issues:
            col = item["column"]
            parts = []
            if "mixed_casing" in item:
                parts.append(f"{len(item['mixed_casing'])} mixed-casing groups")
            if "whitespace" in item:
                parts.append(f"{item['whitespace']['count']} whitespace issues")
            print_warning(f"'{col}': {', '.join(parts)}")
    else:
        print_info("no string inconsistencies")

    # Range issues
    if range_issues:
        for item in range_issues:
            print_warning(
                f"'{item['column']}': {item['issue']} "
                f"(min={item['min']}, max={item['max']})"
            )
    else:
        print_info("no range validation issues")

    # Constant columns
    if constant_cols:
        for item in constant_cols:
            print_warning(
                f"constant column '{item['column']}' = {item['constant_value']}"
            )
    else:
        print_info("no constant columns")

    # Summary counts
    total = (
        exact_dupes["count"]
        + len(identifier_dupes)
        + len(string_issues)
        + len(range_issues)
        + len(constant_cols)
    )
    print_detail("total issue categories flagged", total)


# ---------------------------------------------------------------------------
# Summary visualization
# ---------------------------------------------------------------------------

def _plot_summary(exact_dupes, identifier_dupes, string_issues,
                  range_issues, constant_cols, state) -> list:
    categories = []
    counts = []

    if exact_dupes["count"]:
        categories.append("Exact Duplicates")
        counts.append(exact_dupes["count"])
    if identifier_dupes:
        categories.append("Identifier Duplicates")
        counts.append(sum(i["duplicate_count"] for i in identifier_dupes))
    if string_issues:
        categories.append("String Inconsistencies")
        counts.append(len(string_issues))
    if range_issues:
        categories.append("Range Validation")
        counts.append(len(range_issues))
    if constant_cols:
        categories.append("Constant Columns")
        counts.append(len(constant_cols))

    if not categories:
        return []

    fig, ax = plt.subplots(figsize=(8, max(3, len(categories) * 0.8)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(categories, counts, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Issue Count")
    ax.set_title("Data Quality Issues Summary", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = save_and_show(fig, state, "data_quality_summary.png")
    plt.close(fig)
    return [path]
