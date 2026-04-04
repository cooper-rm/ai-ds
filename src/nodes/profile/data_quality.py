"""
Data quality checks node.

Checks for: exact duplicates, identifier column duplicates, string
inconsistencies (mixed casing, whitespace), range validation, and
constant columns.  Produces a summary bar chart of issue counts.
"""

import itertools
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

    # 6. Near-duplicate rows
    near_dupes = _check_near_duplicates(df)

    # 7. Cross-column consistency
    cross_column = _check_cross_column_consistency(df, classifications)

    # 8. String normalization issues
    string_norm = _check_string_normalization(df)

    # 9. Uniqueness validation
    uniqueness = _check_uniqueness_validation(df, classifications)

    # ---- terminal output ----
    _print_findings(
        exact_dupes, identifier_dupes, string_issues, range_issues, constant_cols,
        near_dupes, cross_column, string_norm, uniqueness,
    )

    # ---- summary visualization ----
    images += _plot_summary(
        exact_dupes, identifier_dupes, string_issues, range_issues, constant_cols,
        near_dupes, cross_column, string_norm, uniqueness, state,
    )

    # ---- persist to state ----
    state["nodes"]["data_quality"] = {
        "status": "issues_found" if any([
            exact_dupes["count"],
            identifier_dupes,
            string_issues,
            range_issues,
            constant_cols,
            near_dupes["count"],
            cross_column["date_order_violations"] or cross_column["derived_columns"],
            string_norm,
            uniqueness["likely_identifiers"] or uniqueness["low_uniqueness"],
        ]) else "clean",
        "exact_duplicates": exact_dupes,
        "identifier_duplicates": identifier_dupes,
        "string_issues": string_issues,
        "range_issues": range_issues,
        "constant_columns": constant_cols,
        "near_duplicates": near_dupes,
        "cross_column_consistency": cross_column,
        "string_normalization": string_norm,
        "uniqueness_validation": uniqueness,
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
        "near_duplicate_count": near_dupes["count"],
        "cross_column_date_violations": len(cross_column["date_order_violations"]),
        "cross_column_derived": len(cross_column["derived_columns"]),
        "string_normalization_count": len(string_norm),
        "likely_identifiers": len(uniqueness["likely_identifiers"]),
        "low_uniqueness_columns": len(uniqueness["low_uniqueness"]),
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
# 6. Near-duplicate rows
# ---------------------------------------------------------------------------

_NEAR_DUPE_SAMPLE_CAP = 5000


def _check_near_duplicates(df: pd.DataFrame) -> dict:
    """Find rows identical on all columns except 1-2."""
    n_rows, n_cols = df.shape
    if n_rows < 2 or n_cols < 3:
        return {"count": 0, "examples": []}

    # Sample down if dataset is large
    if n_rows > _NEAR_DUPE_SAMPLE_CAP:
        sample = df.sample(n=_NEAR_DUPE_SAMPLE_CAP, random_state=42).reset_index(
            drop=True
        )
    else:
        sample = df.reset_index(drop=True)

    threshold = 0.8
    min_matching = int(n_cols * threshold)

    # Convert to object array for fast row-wise comparison
    arr = sample.astype(str).values
    near_pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for i in range(len(arr)):
        if len(near_pairs) >= 50:
            break
        for j in range(i + 1, len(arr)):
            if (i, j) in seen:
                continue
            matches = int((arr[i] == arr[j]).sum())
            if matches >= min_matching and matches < n_cols:
                diff_cols = [
                    sample.columns[k]
                    for k in range(n_cols)
                    if arr[i][k] != arr[j][k]
                ]
                near_pairs.append({
                    "row_a": int(i),
                    "row_b": int(j),
                    "matching_cols": matches,
                    "differing_cols": diff_cols,
                })
                seen.add((i, j))
            if len(near_pairs) >= 50:
                break

    return {
        "count": len(near_pairs),
        "examples": near_pairs[:10],
        "sampled": n_rows > _NEAR_DUPE_SAMPLE_CAP,
    }


# ---------------------------------------------------------------------------
# 7. Cross-column consistency
# ---------------------------------------------------------------------------

def _check_cross_column_consistency(
    df: pd.DataFrame, classifications: dict,
) -> dict:
    """Check date ordering and derived-column redundancy."""
    date_violations: list[dict] = []
    derived_columns: list[dict] = []

    # --- Date ordering checks ---
    date_cols = [
        col for col, info in classifications.items()
        if info.get("type") == "datetime" and col in df.columns
    ]
    if len(date_cols) >= 2:
        # Parse date columns
        parsed: dict[str, pd.Series] = {}
        for col in date_cols:
            try:
                parsed[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                continue

        # Check all pairs for logical ordering based on column name hints
        _ORDER_HINTS = [
            ("start", "end"), ("begin", "end"), ("created", "updated"),
            ("created", "closed"), ("open", "close"), ("from", "to"),
            ("departure", "arrival"), ("hire", "termination"),
            ("birth", "death"), ("issue", "resolve"),
        ]
        parsed_names = list(parsed.keys())
        for ca, cb in itertools.combinations(parsed_names, 2):
            ca_low, cb_low = ca.lower(), cb.lower()
            for hint_a, hint_b in _ORDER_HINTS:
                if hint_a in ca_low and hint_b in cb_low:
                    mask = (parsed[ca].notna()) & (parsed[cb].notna())
                    violations = int((parsed[ca][mask] > parsed[cb][mask]).sum())
                    if violations > 0:
                        date_violations.append({
                            "column_early": ca,
                            "column_late": cb,
                            "violation_count": violations,
                        })
                elif hint_a in cb_low and hint_b in ca_low:
                    mask = (parsed[cb].notna()) & (parsed[ca].notna())
                    violations = int((parsed[cb][mask] > parsed[ca][mask]).sum())
                    if violations > 0:
                        date_violations.append({
                            "column_early": cb,
                            "column_late": ca,
                            "violation_count": violations,
                        })

    # --- Derived column detection ---
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if df[c].notna().sum() > 10
    ]
    if len(num_cols) >= 3:
        for target in num_cols:
            if derived_columns and len(derived_columns) >= 20:
                break
            for ca, cb in itertools.combinations(
                [c for c in num_cols if c != target], 2
            ):
                sa = df[ca].astype(float)
                sb = df[cb].astype(float)
                st = df[target].astype(float)
                mask = sa.notna() & sb.notna() & st.notna()
                if mask.sum() < 10:
                    continue
                sa_m, sb_m, st_m = sa[mask], sb[mask], st[mask]
                # Sum
                if np.allclose(sa_m + sb_m, st_m, equal_nan=True, rtol=1e-6):
                    derived_columns.append({
                        "column": target,
                        "derived_from": [ca, cb],
                        "operation": "sum",
                    })
                    break
                # Difference (a - b)
                if np.allclose(sa_m - sb_m, st_m, equal_nan=True, rtol=1e-6):
                    derived_columns.append({
                        "column": target,
                        "derived_from": [ca, cb],
                        "operation": "difference",
                    })
                    break
                # Product
                if np.allclose(sa_m * sb_m, st_m, equal_nan=True, rtol=1e-6):
                    derived_columns.append({
                        "column": target,
                        "derived_from": [ca, cb],
                        "operation": "product",
                    })
                    break
                # Ratio (a / b, guard zero-division)
                safe = mask & (sb != 0)
                if safe.sum() >= 10:
                    if np.allclose(
                        sa[safe] / sb[safe], st[safe],
                        equal_nan=True, rtol=1e-6,
                    ):
                        derived_columns.append({
                            "column": target,
                            "derived_from": [ca, cb],
                            "operation": "ratio",
                        })
                        break

    return {
        "date_order_violations": date_violations,
        "derived_columns": derived_columns,
    }


# ---------------------------------------------------------------------------
# 8. String normalization issues
# ---------------------------------------------------------------------------

_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _check_string_normalization(df: pd.DataFrame) -> list:
    """Detect embedded special chars and inconsistent delimiters."""
    results = []
    str_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        issues: dict = {}
        series = df[col].dropna().astype(str)
        if series.empty:
            continue

        # Embedded non-printable characters / tabs within values
        special_mask = series.str.contains(_NON_PRINTABLE_RE, na=False)
        tab_mask = series.str.contains(r"\t", na=False)
        combined_mask = special_mask | tab_mask
        special_count = int(combined_mask.sum())
        if special_count > 0:
            issues["special_chars"] = {
                "count": special_count,
                "examples": series[combined_mask].head(5).tolist(),
            }

        # Inconsistent delimiters within a column
        comma_count = int(series.str.contains(",", na=False).sum())
        semicolon_count = int(series.str.contains(";", na=False).sum())
        if comma_count > 0 and semicolon_count > 0:
            total = len(series)
            # Flag only when both delimiters are used by a meaningful fraction
            if (
                min(comma_count, semicolon_count) / max(comma_count, semicolon_count)
                > 0.05
            ):
                issues["inconsistent_delimiters"] = {
                    "comma_values": comma_count,
                    "semicolon_values": semicolon_count,
                    "total_values": total,
                }

        if issues:
            issues["column"] = col
            results.append(issues)

    return results


# ---------------------------------------------------------------------------
# 9. Uniqueness validation
# ---------------------------------------------------------------------------

def _check_uniqueness_validation(
    df: pd.DataFrame, classifications: dict,
) -> dict:
    """Flag columns with extreme uniqueness ratios."""
    likely_identifiers: list[dict] = []
    low_uniqueness: list[dict] = []

    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        nunique = series.nunique()
        count = len(series)
        ratio = nunique / count

        # 100% unique -> likely identifier
        if ratio == 1.0 and count > 1:
            classified_type = (
                classifications.get(col, {}).get("type", "unknown")
            )
            likely_identifiers.append({
                "column": col,
                "nunique": nunique,
                "count": count,
                "classified_as": classified_type,
            })

        # Very low uniqueness on non-binary column
        if ratio < 0.001 and nunique > 2:
            likely_identifiers_cols = {
                col for col, info in classifications.items()
                if info.get("type") == "identifier"
            }
            if col not in likely_identifiers_cols:
                low_uniqueness.append({
                    "column": col,
                    "nunique": nunique,
                    "count": count,
                    "uniqueness_ratio": round(ratio, 6),
                })

    return {
        "likely_identifiers": likely_identifiers,
        "low_uniqueness": low_uniqueness,
    }


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _print_findings(exact_dupes, identifier_dupes, string_issues,
                    range_issues, constant_cols, near_dupes, cross_column,
                    string_norm, uniqueness):
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

    # Near-duplicate rows
    if near_dupes["count"] > 0:
        suffix = " (sampled)" if near_dupes.get("sampled") else ""
        print_warning(
            f"{near_dupes['count']} near-duplicate row pairs detected{suffix}"
        )
        for pair in near_dupes["examples"][:3]:
            print_detail(
                f"  rows {pair['row_a']} & {pair['row_b']}: "
                f"differ on {pair['differing_cols']}",
                "",
            )
    else:
        print_info("no near-duplicate rows")

    # Cross-column consistency
    if cross_column["date_order_violations"]:
        for v in cross_column["date_order_violations"]:
            print_warning(
                f"date ordering violation: '{v['column_early']}' > "
                f"'{v['column_late']}' in {v['violation_count']} rows"
            )
    else:
        print_info("no date ordering violations")

    if cross_column["derived_columns"]:
        for d in cross_column["derived_columns"]:
            print_warning(
                f"'{d['column']}' appears derived ({d['operation']}) "
                f"from {d['derived_from']}"
            )
    else:
        print_info("no derived-column redundancies detected")

    # String normalization
    if string_norm:
        for item in string_norm:
            col = item["column"]
            parts = []
            if "special_chars" in item:
                parts.append(
                    f"{item['special_chars']['count']} values with special chars"
                )
            if "inconsistent_delimiters" in item:
                d = item["inconsistent_delimiters"]
                parts.append(
                    f"mixed delimiters (comma={d['comma_values']}, "
                    f"semicolon={d['semicolon_values']})"
                )
            print_warning(f"'{col}': {', '.join(parts)}")
    else:
        print_info("no string normalization issues")

    # Uniqueness validation
    if uniqueness["likely_identifiers"]:
        for item in uniqueness["likely_identifiers"]:
            print_info(
                f"'{item['column']}' is 100% unique "
                f"(classified as {item['classified_as']})"
            )
    if uniqueness["low_uniqueness"]:
        for item in uniqueness["low_uniqueness"]:
            print_warning(
                f"'{item['column']}' has very low uniqueness "
                f"({item['uniqueness_ratio']:.4%}, {item['nunique']} unique "
                f"of {item['count']})"
            )
    if not uniqueness["likely_identifiers"] and not uniqueness["low_uniqueness"]:
        print_info("no uniqueness anomalies")

    # Summary counts
    total = (
        exact_dupes["count"]
        + len(identifier_dupes)
        + len(string_issues)
        + len(range_issues)
        + len(constant_cols)
        + near_dupes["count"]
        + len(cross_column["date_order_violations"])
        + len(cross_column["derived_columns"])
        + len(string_norm)
        + len(uniqueness["likely_identifiers"])
        + len(uniqueness["low_uniqueness"])
    )
    print_detail("total issue categories flagged", total)


# ---------------------------------------------------------------------------
# Summary visualization
# ---------------------------------------------------------------------------

def _plot_summary(exact_dupes, identifier_dupes, string_issues,
                  range_issues, constant_cols, near_dupes, cross_column,
                  string_norm, uniqueness, state) -> list:
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
    if near_dupes["count"]:
        categories.append("Near-Duplicate Rows")
        counts.append(near_dupes["count"])
    cross_col_total = (
        len(cross_column["date_order_violations"])
        + len(cross_column["derived_columns"])
    )
    if cross_col_total:
        categories.append("Cross-Column Consistency")
        counts.append(cross_col_total)
    if string_norm:
        categories.append("String Normalization")
        counts.append(len(string_norm))
    uniqueness_total = (
        len(uniqueness["likely_identifiers"]) + len(uniqueness["low_uniqueness"])
    )
    if uniqueness_total:
        categories.append("Uniqueness Validation")
        counts.append(uniqueness_total)

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
