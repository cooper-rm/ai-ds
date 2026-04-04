"""
Leakage detection node.

Scans the dataset for potential data leakage patterns that could
invalidate model performance estimates:

  - Target leakage (features with suspiciously high correlation to target)
  - Identifier leakage (ID columns still present in the feature set)
  - Temporal leakage (future-looking columns or misordered time data)
  - High-cardinality string leakage (free text that may encode the target)
  - Near-constant post-target columns (features derived from the target)

No visualizations — terminal warnings and report narrative only.
"""
import numpy as np
import pandas as pd

from src.report import narrate, add_section
from src.terminal import print_info, print_detail, print_warning


# Column names that commonly represent a target variable
TARGET_NAMES = {"target", "label", "survived", "y", "class", "outcome"}

# Correlation threshold for flagging target leakage
CORRELATION_THRESHOLD = 0.95

# Cardinality ratio threshold for high-cardinality string leakage
HIGH_CARDINALITY_RATIO = 0.50

# Keywords suggesting future/forecast information
TEMPORAL_LEAK_KEYWORDS = {"future", "next", "predicted", "forecast"}


def leakage(state: dict) -> dict:
    """Detect potential data leakage across multiple dimensions."""
    df = state["data"]
    findings = []

    # ── Identify target column ───────────────────────────────────────────
    target_col = _find_target_column(state)

    # ── 1. Target leakage ────────────────────────────────────────────────
    if target_col is not None and target_col in df.columns:
        findings += _check_target_leakage(df, target_col)
    else:
        print_detail("leakage", "No clear target column found — skipping target-dependent checks.")

    # ── 2. Identifier leakage ────────────────────────────────────────────
    findings += _check_identifier_leakage(df, state)

    # ── 3. Temporal leakage ──────────────────────────────────────────────
    findings += _check_temporal_leakage(df, state)

    # ── 4. High-cardinality string leakage ───────────────────────────────
    findings += _check_high_cardinality_leakage(df, state, target_col)

    # ── 5. Near-constant post-target columns ─────────────────────────────
    if target_col is not None and target_col in df.columns:
        findings += _check_near_constant_within_target(df, target_col)

    # ── Terminal output ──────────────────────────────────────────────────
    critical = [f for f in findings if f["severity"] == "critical"]
    warnings = [f for f in findings if f["severity"] == "warning"]

    if not findings:
        print_info("No data leakage indicators detected.")
    else:
        if critical:
            print_warning(f"CRITICAL leakage risks: {len(critical)}")
            for f in critical:
                print_warning(f"  [{f['check']}] {f['column']}: {f['detail']}")
        if warnings:
            print_info(f"Leakage warnings: {len(warnings)}")
            for f in warnings:
                print_info(f"  [{f['check']}] {f['column']}: {f['detail']}")

    # ── Store results ────────────────────────────────────────────────────
    state["nodes"]["leakage"] = {
        "status": "risks_found" if findings else "clean",
        "target_column": target_col,
        "findings": findings,
        "critical_count": len(critical),
        "warning_count": len(warnings),
    }

    # ── Narrative + report section ───────────────────────────────────────
    narrative = narrate("Data Leakage Detection", {
        "target_column": target_col,
        "findings": findings,
        "critical_count": len(critical),
        "warning_count": len(warnings),
    })
    add_section(state, "Data Leakage Detection", narrative)

    return state


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_target_column(state: dict):
    """Identify the target column from classify node results, if available."""
    classifications = (
        state.get("nodes", {})
        .get("classify", {})
        .get("classifications", {})
    )
    if not classifications:
        return None

    for col, info in classifications.items():
        col_type = info.get("type", "")
        if col_type == "binary" and col.lower() in TARGET_NAMES:
            return col

    # Fallback: any binary column whose name matches target patterns
    for col, info in classifications.items():
        if col.lower() in TARGET_NAMES:
            return col

    return None


def _check_target_leakage(df, target_col):
    """Check for features with suspiciously high correlation or mutual
    information with the target, and for trivial 1:1 mappings."""
    findings = []
    target = df[target_col]

    # Only proceed if target is numeric or can be coerced
    try:
        target_numeric = pd.to_numeric(target, errors="coerce")
    except Exception:
        return findings

    if target_numeric.isna().all():
        return findings

    numeric_cols = [
        col for col in df.columns
        if col != target_col and str(df[col].dtype) in (
            "int8", "int16", "int32", "int64", "float16", "float32", "float64"
        )
    ]

    # Pearson correlation check
    for col in numeric_cols:
        try:
            series = df[col].dropna()
            common = target_numeric.dropna().index.intersection(series.index)
            if len(common) < 10:
                continue
            r = target_numeric.loc[common].corr(df[col].loc[common])
            if pd.notna(r) and abs(r) > CORRELATION_THRESHOLD:
                findings.append({
                    "check": "target_correlation",
                    "column": col,
                    "severity": "critical",
                    "detail": f"|r| = {abs(r):.4f} with target '{target_col}'",
                    "value": round(float(r), 4),
                })
        except Exception:
            continue

    # 1:1 mapping check (exact bijection with target)
    for col in df.columns:
        if col == target_col:
            continue
        try:
            paired = df[[target_col, col]].dropna()
            if len(paired) < 10:
                continue
            # Group by target and count unique feature values
            mapping = paired.groupby(target_col)[col].nunique()
            # If every target value maps to exactly one feature value
            if (mapping == 1).all():
                reverse = paired.groupby(col)[target_col].nunique()
                if (reverse == 1).all():
                    findings.append({
                        "check": "target_1to1_mapping",
                        "column": col,
                        "severity": "critical",
                        "detail": f"1:1 bijection with target '{target_col}' — likely derived from target",
                    })
        except Exception:
            continue

    return findings


def _check_identifier_leakage(df, state):
    """Flag identifier columns that are still in the dataset."""
    findings = []
    classifications = (
        state.get("nodes", {})
        .get("classify", {})
        .get("classifications", {})
    )
    if not classifications:
        return findings

    for col, info in classifications.items():
        if info.get("type") == "identifier" and col in df.columns:
            findings.append({
                "check": "identifier_present",
                "column": col,
                "severity": "warning",
                "detail": "Identifier column still present — will leak row-level information if used as feature.",
            })

    return findings


def _check_temporal_leakage(df, state):
    """Check for columns that suggest future information or temporal ordering issues."""
    findings = []
    classifications = (
        state.get("nodes", {})
        .get("classify", {})
        .get("classifications", {})
    )

    # Check column names for future-looking keywords
    for col in df.columns:
        col_lower = col.lower()
        for keyword in TEMPORAL_LEAK_KEYWORDS:
            if keyword in col_lower:
                findings.append({
                    "check": "temporal_keyword",
                    "column": col,
                    "severity": "warning",
                    "detail": f"Column name contains '{keyword}' — may represent future information.",
                })
                break

    # If classifications available, find datetime columns
    if not classifications:
        return findings

    datetime_cols = [
        col for col, info in classifications.items()
        if info.get("type") == "datetime" and col in df.columns
    ]

    if len(datetime_cols) < 2:
        return findings

    # Try to parse datetime columns and check if any are after the primary one
    parsed = {}
    for col in datetime_cols:
        try:
            parsed[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue

    if len(parsed) < 2:
        return findings

    # Use the first datetime column (by position) as the primary timestamp
    primary_col = datetime_cols[0]
    primary_ts = parsed.get(primary_col)
    if primary_ts is None or primary_ts.isna().all():
        return findings

    for col, ts in parsed.items():
        if col == primary_col:
            continue
        if ts.isna().all():
            continue
        # Check if this datetime is consistently after the primary
        valid = primary_ts.notna() & ts.notna()
        if valid.sum() < 10:
            continue
        pct_after = (ts[valid] > primary_ts[valid]).mean()
        if pct_after > 0.9:
            findings.append({
                "check": "temporal_future_datetime",
                "column": col,
                "severity": "critical",
                "detail": (
                    f"{pct_after:.0%} of values are after primary datetime "
                    f"'{primary_col}' — potential future leakage."
                ),
            })

    return findings


def _check_high_cardinality_leakage(df, state, target_col):
    """Flag high-cardinality non-identifier columns that might encode the target."""
    findings = []
    classifications = (
        state.get("nodes", {})
        .get("classify", {})
        .get("classifications", {})
    )

    for col in df.columns:
        if col == target_col:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Only check object/string columns
        if str(df[col].dtype) not in ("object", "string", "category"):
            continue

        # Skip if already classified as identifier
        if classifications:
            col_info = classifications.get(col, {})
            if col_info.get("type") == "identifier":
                continue

        nunique = series.nunique()
        ratio = nunique / len(series)

        if ratio > HIGH_CARDINALITY_RATIO:
            findings.append({
                "check": "high_cardinality_string",
                "column": col,
                "severity": "warning",
                "detail": (
                    f"{nunique} unique values ({ratio:.0%} of rows) — "
                    f"high-cardinality text may encode target information."
                ),
            })

    return findings


def _check_near_constant_within_target(df, target_col):
    """Find columns that are nearly constant within each target class,
    suggesting they were derived from the target."""
    findings = []
    target = df[target_col]

    numeric_cols = [
        col for col in df.columns
        if col != target_col and str(df[col].dtype) in (
            "int8", "int16", "int32", "int64", "float16", "float32", "float64"
        )
    ]

    for col in numeric_cols:
        try:
            overall_std = df[col].std()
            if pd.isna(overall_std) or overall_std == 0:
                continue

            group_stds = df.groupby(target_col)[col].std()
            # Check if std within every group is very small relative to overall
            if group_stds.isna().all():
                continue
            max_group_std = group_stds.max()
            if pd.isna(max_group_std):
                continue

            if max_group_std < 0.01 * overall_std:
                findings.append({
                    "check": "near_constant_within_target",
                    "column": col,
                    "severity": "critical",
                    "detail": (
                        f"Within-class std ({max_group_std:.6f}) < 1% of overall std "
                        f"({overall_std:.4f}) — column may be derived from target."
                    ),
                })
        except Exception:
            continue

    return findings
