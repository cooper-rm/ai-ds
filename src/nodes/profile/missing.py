"""
Missing value classification node.

For each column with missing values:
  1. Classify missingness type: MCAR / MAR / MNAR_suspected
  2. Apply drop threshold (>50% missing → recommend drop)
  3. Gather correlates and column statistics

Output: state["nodes"]["missing"] with evidence per column.
Consumed by: nodes/imputation/imputation.py and synthesis.py
"""
import numpy as np
from scipy import stats

from src.report import narrate, add_section
from src.terminal import print_info, print_detail

DROP_THRESHOLD     = 0.50
WARN_THRESHOLD     = 0.30
MAR_CORR_THRESHOLD = 0.15
NUMERIC_DTYPES     = ("int8","int16","int32","int64","float32","float64")


def missing(state: dict) -> dict:
    """Classify missingness type and gather imputation evidence per column."""
    df = state["data"]

    missing_cols = [c for c in df.columns if df[c].isnull().any()]

    if not missing_cols:
        state["nodes"]["missing"] = {"status": "no_missing", "results": {}}
        print_info("no missing values found")
        return state

    results = {}

    for col in missing_cols:
        missing_count = int(df[col].isnull().sum())
        missing_pct   = round(missing_count / len(df), 4)
        is_numeric    = str(df[col].dtype) in NUMERIC_DTYPES

        if missing_pct > DROP_THRESHOLD:
            results[col] = {
                "missing_count":     missing_count,
                "missing_pct":       missing_pct,
                "missingness_type":  "unknown",
                "is_numeric":        is_numeric,
                "dtype":             str(df[col].dtype),
                "nunique":           int(df[col].nunique()),
                "recommendation":    "drop",
                "reason":            f"{missing_pct:.0%} missing exceeds {DROP_THRESHOLD:.0%} threshold",
                "mar_correlates":    [],
                "top_categorical_correlate": None,
            }
            print_info(f"{col}: {missing_pct:.1%} missing → [red]recommend drop[/red]")
            continue

        ev = _gather_evidence(df, col, missing_count, missing_pct)
        results[col] = ev

        flag = "  [yellow]⚠ high missingness[/yellow]" if missing_pct > WARN_THRESHOLD else ""
        print_info(
            f"{col}: {missing_pct:.1%} missing  ·  {ev['missingness_type']}"
            + (f"  ·  correlates: {ev['mar_correlates'][:2]}" if ev["mar_correlates"] else "")
            + flag
        )

    to_drop   = [c for c, r in results.items() if r.get("recommendation") == "drop"]
    to_impute = [c for c, r in results.items() if r.get("recommendation") != "drop"]

    print_detail("recommend drop",   len(to_drop))
    print_detail("recommend impute", len(to_impute))

    state["nodes"]["missing"] = {
        "status":            "classified",
        "results":           results,
        "columns_to_drop":   to_drop,
        "columns_to_impute": to_impute,
    }

    narrative = narrate("Missing Value Classification", {
        "summary": {
            col: {"missing_pct": r["missing_pct"], "type": r["missingness_type"]}
            for col, r in results.items()
        }
    })
    add_section(state, "Missing Value Classification", narrative)
    return state


def _gather_evidence(df, col, missing_count, missing_pct):
    is_numeric = str(df[col].dtype) in NUMERIC_DTYPES
    indicator  = df[col].isnull().astype(int)

    mar_correlates       = []
    top_categorical_corr = None

    for other in df.columns:
        if other == col:
            continue
        try:
            if str(df[other].dtype) in NUMERIC_DTYPES and df[other].notnull().sum() > 10:
                r, _ = stats.pointbiserialr(indicator, df[other].fillna(df[other].median()))
                if abs(r) >= MAR_CORR_THRESHOLD:
                    mar_correlates.append({"column": other, "correlation": round(float(r), 4)})
            else:
                group_rates = df.groupby(df[other].fillna("__missing__"))[col].apply(
                    lambda s: s.isnull().mean()
                )
                spread = float(group_rates.max() - group_rates.min())
                if spread >= MAR_CORR_THRESHOLD:
                    if top_categorical_corr is None or spread > top_categorical_corr[1]:
                        top_categorical_corr = (other, spread)
        except Exception:
            continue

    mar_correlates.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    if mar_correlates or top_categorical_corr:
        missingness_type = "MAR"
    elif is_numeric:
        missingness_type = "MNAR_suspected"
    else:
        missingness_type = "MCAR"

    ev = {
        "missing_count":             missing_count,
        "missing_pct":               missing_pct,
        "missingness_type":          missingness_type,
        "is_numeric":                is_numeric,
        "dtype":                     str(df[col].dtype),
        "nunique":                   int(df[col].nunique()),
        "mar_correlates":            [c["column"] for c in mar_correlates[:3]],
        "mar_correlate_strengths":   mar_correlates[:3],
        "top_categorical_correlate": top_categorical_corr[0] if top_categorical_corr else None,
        "recommendation":            None,  # filled by imputation node
    }

    if is_numeric:
        complete = df[col].dropna().astype(float)
        ev["mean"]     = round(float(complete.mean()), 4)
        ev["std"]      = round(float(complete.std()), 4)
        ev["skewness"] = round(float(complete.skew()), 4)

    return ev
