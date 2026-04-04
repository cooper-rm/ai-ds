"""
Split-data node — splits the loaded dataset into train/val/test sets.

Runs after load_data but before profiling or interview nodes.
All downstream nodes operate on the training split only, ensuring
no data leakage from validation or test sets into profiling/preprocessing.

Split strategy is chosen automatically based on data characteristics:
  - Temporal data (datetime columns): chronological split, NO shuffle
  - Classification target: stratified random split
  - Other: random split
The user can override via interactive prompt.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.terminal import print_info, print_detail, print_warning, prompt_choice, console, llm_spinner
from src.llm.client import ask
from src.utils import snapshot

# Column names that strongly suggest a target variable
_TARGET_NAME_HINTS = {
    "target", "label", "y", "class", "outcome",
    "survived", "churn", "default", "fraud",
}

# Column names / dtypes that suggest temporal data
_TEMPORAL_NAME_HINTS = {
    "date", "time", "timestamp", "datetime", "created", "updated",
    "created_at", "updated_at", "event_date", "order_date", "ts",
}


def _is_interactive() -> bool:
    """Return True if stdin is attached to a terminal."""
    return sys.stdin.isatty()


def _detect_temporal_column(df: pd.DataFrame) -> str | None:
    """
    Detect if the dataset has a datetime/temporal column that suggests
    time-ordered data. Returns the column name or None.
    """
    # Check for datetime dtype columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # Check column names against temporal hints
    for col in df.columns:
        if col.lower().strip().replace("_", "") in {
            h.replace("_", "") for h in _TEMPORAL_NAME_HINTS
        }:
            # Try to parse as datetime
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.8:
                    return col
            except Exception:
                pass

    return None


def _detect_target(df: pd.DataFrame) -> str | None:
    """
    Auto-detect a likely target column for stratified splitting.

    Checks for common target names first, then falls back to binary columns.
    Returns the column name or None.
    """
    # Priority 1: column name matches a known target pattern
    for col in df.columns:
        if col.lower().strip() in _TARGET_NAME_HINTS:
            return col

    # Priority 2: binary columns (exactly 2 unique non-null values)
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        if series.nunique() == 2:
            return col

    return None


def _parse_ratios(raw: str) -> tuple[float, float, float] | None:
    """
    Parse a ratio string like '70/15/15' or '80/10/10'.

    Returns (train, val, test) as fractions summing to 1.0, or None on failure.
    """
    parts = raw.strip().replace(" ", "").split("/")
    if len(parts) != 3:
        return None
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None

    total = sum(values)
    if total <= 0:
        return None

    fracs = tuple(v / total for v in values)
    # Sanity check: no fraction should be zero or negative
    if any(f <= 0 for f in fracs):
        return None

    return fracs  # type: ignore[return-value]


def _get_llm_recommendation(df: pd.DataFrame, temporal_col: str | None, target_col: str | None) -> dict:
    """
    Ask the LLM to recommend a split strategy based on data characteristics.
    Returns a dict with 'method', 'ratios', and 'reasoning'.
    """
    import json

    n_rows = len(df)
    n_cols = len(df.columns)
    dtypes = {col: str(df[col].dtype) for col in df.columns}
    missing_pct = {col: round(df[col].isnull().mean() * 100, 1)
                   for col in df.columns if df[col].isnull().any()}

    target_info = "none detected"
    if target_col and target_col in df.columns:
        nunique = df[target_col].nunique()
        balance = df[target_col].value_counts(normalize=True).to_dict()
        target_info = f"'{target_col}' — {nunique} classes, balance: {balance}"

    temporal_info = f"'{temporal_col}'" if temporal_col else "none detected"

    prompt = f"""You are advising on how to split a dataset for analysis and modeling.

Dataset characteristics:
- {n_rows} rows, {n_cols} columns
- Column dtypes: {json.dumps(dtypes)}
- Missing values: {json.dumps(missing_pct) if missing_pct else "none"}
- Target column: {target_info}
- Temporal column: {temporal_info}

Recommend a split strategy. Consider:
1. If temporal data exists, shuffling leaks future info — use chronological split
2. If classification target exists with imbalanced classes, use stratified split
3. If dataset is small (<1000 rows), consider larger train ratio (80/10/10)
4. If dataset is large (>100k), standard 70/15/15 is fine

Respond with EXACTLY this JSON, nothing else:
{{
  "method": "chronological|stratified|random",
  "ratios": "70/15/15",
  "reasoning": "One sentence explaining why this strategy is best for this data."
}}"""

    try:
        with llm_spinner("Analyzing split strategy"):
            response = ask(prompt, system="You are a data science advisor. Respond only in JSON.")

        result = json.loads(response)
        # Validate expected keys
        if "method" in result and "ratios" in result:
            return result
    except Exception:
        pass

    # Fallback: heuristic recommendation
    if temporal_col:
        return {
            "method": "chronological",
            "ratios": "70/15/15",
            "reasoning": f"Temporal column '{temporal_col}' detected — chronological split prevents future leakage.",
        }
    if target_col:
        return {
            "method": "stratified",
            "ratios": "80/10/10" if n_rows < 1000 else "70/15/15",
            "reasoning": f"Target column '{target_col}' detected — stratified split preserves class balance across splits.",
        }
    return {
        "method": "random",
        "ratios": "70/15/15",
        "reasoning": "No temporal or target column detected — random shuffle split is appropriate.",
    }


def _print_split_distribution(label: str, df_split: pd.DataFrame, target_col: str | None):
    """Print row count and target distribution for a split."""
    count = len(df_split)
    if target_col and target_col in df_split.columns:
        dist = df_split[target_col].value_counts(normalize=True)
        dist_str = ", ".join(f"{k}={v:.1%}" for k, v in dist.items())
        print_detail(label, f"{count} rows  ({dist_str})")
    else:
        print_detail(label, f"{count} rows")


def split_data(state: dict) -> dict:
    """Split the dataset into train/val/test sets."""
    df = state["data"]
    interactive = _is_interactive()

    # ── 1. Detect data characteristics ─────────────────────────────────────
    temporal_col = _detect_temporal_column(df)
    target_col = _detect_target(df)

    if temporal_col:
        print_warning(f"temporal column detected: '{temporal_col}' — shuffling would cause leakage")

    # ── 2. LLM recommendation ───────────────────────────────────────────────
    recommendation = _get_llm_recommendation(df, temporal_col, target_col)
    rec_method = recommendation.get("method", "stratified" if target_col else "random")
    rec_ratios = recommendation.get("ratios", "70/15/15")
    rec_reasoning = recommendation.get("reasoning", "")

    # ── 3. Ask for split strategy ────────────────────────────────────────────
    if interactive:
        rec_label = {
            "chronological": "Chronological (no shuffle — preserves time order)",
            "stratified": f"Stratified on '{target_col}' (preserves class balance)",
            "random": "Random shuffle",
        }.get(rec_method, rec_method)

        body = (
            f"[bold]Our agent recommends:[/bold]  {rec_label}, {rec_ratios}\n\n"
            f"[dim]{rec_reasoning}[/dim]"
        )

        options = [("y", f"Accept recommendation ({rec_method}, {rec_ratios})")]
        if temporal_col and rec_method != "chronological":
            options.append(("t", "Chronological split (temporal data detected)"))
        if target_col and rec_method != "stratified":
            options.append(("s", f"Stratified split on '{target_col}'"))
        if rec_method != "random":
            options.append(("r", "Random shuffle split"))
        options.append(("c", "Custom (specify your own ratios and method)"))
        options.append(("n", "No split (use all data — not recommended for modeling)"))

        choice = prompt_choice(
            title="Data Splitting Strategy",
            body=body,
            options=options,
        )
    else:
        choice = "y"
        print_info(f"agent recommends: {rec_method} split, {rec_ratios}")
        if rec_reasoning:
            print_info(f"  reasoning: {rec_reasoning}")

    # ── 4. No-split early exit ───────────────────────────────────────────────
    if choice == "n":
        snapshot(state, "split_data_full")
        state["splits"] = None
        state["nodes"]["split_data"] = {
            "status": "skipped",
            "reason": "user chose no split",
            "total_rows": len(df),
            "recommendation": recommendation,
        }
        print_info("no split — all data will be used as-is")
        return state

    # ── 5. Resolve method and ratios from choice ─────────────────────────────
    if choice == "y":
        # Accept LLM recommendation
        split_method = rec_method
        ratios = _parse_ratios(rec_ratios) or (0.70, 0.15, 0.15)
    elif choice == "t":
        split_method = "chronological"
        ratios = _parse_ratios(rec_ratios) or (0.70, 0.15, 0.15)
    elif choice == "s":
        split_method = "stratified"
        ratios = _parse_ratios(rec_ratios) or (0.70, 0.15, 0.15)
    elif choice == "r":
        split_method = "random"
        ratios = _parse_ratios(rec_ratios) or (0.70, 0.15, 0.15)
    elif choice == "c":
        raw = console.input(
            "  [bold yellow]Enter ratios (e.g. 70/15/15):[/bold yellow] "
        ).strip()
        ratios = _parse_ratios(raw)
        if ratios is None:
            print_info("could not parse ratios — falling back to 70/15/15")
            ratios = (0.70, 0.15, 0.15)
        method_raw = console.input(
            "  [bold yellow]Method (random/stratified/chronological):[/bold yellow] "
        ).strip().lower()
        split_method = method_raw if method_raw in ("random", "stratified", "chronological") else rec_method
    else:
        split_method = rec_method
        ratios = (0.70, 0.15, 0.15)

    train_frac, val_frac, test_frac = ratios
    temporal_split = (split_method == "chronological")

    # ── 5. Snapshot the full dataset before splitting ────────────────────────
    snapshot(state, "split_data_full")

    # ── 6. Perform the split ─────────────────────────────────────────────────
    if temporal_split:
        # Chronological split — NO shuffle, preserve row order
        # Sort by temporal column first
        df_sorted = df.sort_values(temporal_col).reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        df_train = df_sorted.iloc[:train_end]
        df_val = df_sorted.iloc[train_end:val_end]
        df_test = df_sorted.iloc[val_end:]

        stratified = False
        print_info(f"chronological split on '{temporal_col}' — no shuffling")
        if temporal_col in df_train.columns:
            try:
                train_range = f"{df_train[temporal_col].min()} → {df_train[temporal_col].max()}"
                test_range = f"{df_test[temporal_col].min()} → {df_test[temporal_col].max()}"
                print_detail("train period", train_range)
                print_detail("test period", test_range)
            except Exception:
                pass
    else:
        # Shuffled split — stratify if method is stratified and target exists
        stratified = False
        if split_method == "stratified" and target_col:
            min_class_count = df[target_col].value_counts().min()
            if min_class_count >= 2:
                stratified = True
                print_info(f"stratifying on '{target_col}'")
            else:
                print_info(
                    f"target '{target_col}' has too few samples per class "
                    f"for stratification — falling back to random split"
                )
        elif split_method == "random":
            print_info("random shuffle split (no stratification)")
        else:
            print_info("random split")

        # Two-stage split
        test_size = test_frac
        remainder_frac = train_frac + val_frac
        val_relative = val_frac / remainder_frac

        stratify_col = df[target_col] if stratified else None
        df_remainder, df_test = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_col,
        )

        stratify_remainder = df_remainder[target_col] if stratified else None
        df_train, df_val = train_test_split(
            df_remainder, test_size=val_relative, random_state=42,
            stratify=stratify_remainder,
        )

    # ── 6. Store results ─────────────────────────────────────────────────────
    state["splits"] = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }

    state["split_indices"] = {
        "train": list(df_train.index),
        "val": list(df_val.index),
        "test": list(df_test.index),
    }

    # Downstream nodes operate on train only
    state["data"] = df_train

    state["nodes"]["split_data"] = {
        "status": "done",
        "method": "chronological" if temporal_split else ("stratified" if stratified else "random"),
        "temporal_column": temporal_col if temporal_split else None,
        "ratios": {
            "train": round(train_frac, 4),
            "val": round(val_frac, 4),
            "test": round(test_frac, 4),
        },
        "row_counts": {
            "total": len(df),
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
        },
        "stratified": stratified,
        "target_column": target_col,
        "recommendation": recommendation,
    }

    # ── 7. Terminal output ───────────────────────────────────────────────────
    print_info(
        f"split {len(df)} rows → "
        f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )
    print_detail("stratified", str(stratified))
    if stratified:
        print_detail("target column", target_col)

    _print_split_distribution("train", df_train, target_col)
    _print_split_distribution("val", df_val, target_col)
    _print_split_distribution("test", df_test, target_col)

    # ── 8. Snapshot the training set ─────────────────────────────────────────
    snapshot(state, "split_data_train")

    return state
