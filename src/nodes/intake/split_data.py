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

from src.terminal import print_info, print_detail, print_warning, prompt_choice, console
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

    # ── 2. Ask for split strategy ────────────────────────────────────────────
    if interactive:
        if temporal_col:
            choice = prompt_choice(
                title="Data Splitting",
                body=(
                    f"Temporal column [bold]'{temporal_col}'[/bold] detected. "
                    "Shuffling time series data causes future information to leak into training.\n\n"
                    "Recommended: chronological split (first 70% = train, next 15% = val, last 15% = test)."
                ),
                options=[
                    ("t", "Chronological split (recommended for temporal data)"),
                    ("s", "Standard shuffled split (70/15/15 — ignores time order)"),
                    ("c", "Custom split (specify ratios)"),
                    ("n", "No split (use all data for EDA)"),
                ],
            )
        else:
            choice = prompt_choice(
                title="Data Splitting",
                body="How would you like to split the data?",
                options=[
                    ("s", "Standard split (70/15/15 train/val/test)"),
                    ("c", "Custom split (specify ratios)"),
                    ("n", "No split (use all data for EDA — not recommended for modeling)"),
                ],
            )
    else:
        if temporal_col:
            choice = "t"
            print_info("non-interactive mode — temporal data detected, using chronological split")
        else:
            choice = "s"
            print_info("non-interactive mode — defaulting to standard 70/15/15 split")

    # ── 3. No-split early exit ───────────────────────────────────────────────
    if choice == "n":
        snapshot(state, "split_data_full")
        state["splits"] = None
        state["nodes"]["split_data"] = {
            "status": "skipped",
            "reason": "user chose no split",
            "total_rows": len(df),
        }
        print_info("no split — all data will be used as-is")
        return state

    # ── 4. Determine ratios ──────────────────────────────────────────────────
    if choice == "c":
        raw = console.input(
            "  [bold yellow]Enter ratios (e.g. 70/15/15):[/bold yellow] "
        ).strip()
        ratios = _parse_ratios(raw)
        if ratios is None:
            print_info("could not parse ratios — falling back to 70/15/15")
            ratios = (0.70, 0.15, 0.15)
    else:
        ratios = (0.70, 0.15, 0.15)

    train_frac, val_frac, test_frac = ratios
    temporal_split = (choice == "t")

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
        # Shuffled split — stratify if target exists
        stratified = False
        if target_col:
            min_class_count = df[target_col].value_counts().min()
            if min_class_count >= 2:
                stratified = True
                print_info(f"stratifying on '{target_col}'")
            else:
                print_info(
                    f"target '{target_col}' detected but too few samples per class "
                    f"for stratification — using random split"
                )
                target_col = None
        else:
            print_info("no target column detected — using random split")

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
