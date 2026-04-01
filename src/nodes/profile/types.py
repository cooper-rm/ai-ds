import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import save_and_show


def types(state: dict) -> dict:
    """Validate dtype inference — find numbers as strings, dates as strings, suggest categoricals."""
    df = state["data"]

    numeric_as_string = []
    dates_as_string = []
    suggest_categorical = []
    type_table = {}

    for col in df.columns:
        current = str(df[col].dtype)

        if current == "object":
            # Check: numbers stored as strings
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if len(non_null) > 0:
                conversion_rate = round(numeric_converted.notna().sum() / len(non_null), 2)
                if conversion_rate > 0.8:
                    blockers = list(df[col][numeric_converted.isna() & df[col].notna()].unique()[:5])
                    numeric_as_string.append({
                        "column": col,
                        "conversion_rate": conversion_rate,
                        "blockers": [str(b) for b in blockers],
                    })
                    type_table[col] = {"current": current, "suggested": "numeric", "reason": f"{int(conversion_rate*100)}% convertible"}
                    continue

            # Check: dates stored as strings
            date_converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if len(non_null) > 0:
                date_rate = round(date_converted.notna().sum() / len(non_null), 2)
                if date_rate > 0.8:
                    dates_as_string.append({
                        "column": col,
                        "conversion_rate": date_rate,
                    })
                    type_table[col] = {"current": current, "suggested": "datetime64", "reason": f"{int(date_rate*100)}% convertible"}
                    continue

            # Check: low cardinality → categorical
            nunique = df[col].nunique()
            ratio = nunique / len(df) if len(df) > 0 else 1
            if nunique < 50 or ratio < 0.05:
                top_values = list(df[col].value_counts().head(5).index)
                suggest_categorical.append({
                    "column": col,
                    "nunique": nunique,
                    "cardinality_ratio": round(ratio, 4),
                    "top_values": [str(v) for v in top_values],
                })
                type_table[col] = {"current": current, "suggested": "category", "reason": f"{nunique} unique values"}
            else:
                type_table[col] = {"current": current, "suggested": None, "reason": "high cardinality string"}

        else:
            type_table[col] = {"current": current, "suggested": None, "reason": "ok"}

    # Determine status
    has_suggestions = bool(numeric_as_string or dates_as_string or suggest_categorical)
    status = "fixes_suggested" if has_suggestions else "all_valid"

    state["nodes"]["types"] = {
        "status": status,
        "numeric_as_string": numeric_as_string,
        "dates_as_string": dates_as_string,
        "suggest_categorical": suggest_categorical,
        "type_table": type_table,
        "images": [],
    }

    # Generate dtype distribution chart
    images = _plot_dtype_distribution(df, state)
    state["nodes"]["types"]["images"] = images

    # Print summary
    print(f"   Status: {status}")
    if numeric_as_string:
        print(f"   Numbers as strings: {len(numeric_as_string)}")
        for item in numeric_as_string:
            print(f"     {item['column']}: {int(item['conversion_rate']*100)}% convertible, blockers: {item['blockers']}")
    if dates_as_string:
        print(f"   Dates as strings: {len(dates_as_string)}")
        for item in dates_as_string:
            print(f"     {item['column']}: {int(item['conversion_rate']*100)}% convertible")
    if suggest_categorical:
        print(f"   Suggest categorical: {len(suggest_categorical)}")
        for item in suggest_categorical:
            print(f"     {item['column']}: {item['nunique']} unique values")

    from src.report import narrate, add_section
    narrative = narrate("Type Validation", {
        "status": status,
        "numeric_as_string": len(numeric_as_string),
        "dates_as_string": len(dates_as_string),
        "suggest_categorical": len(suggest_categorical),
    })
    add_section(state, "Type Validation", narrative, images)

    return state


def _plot_dtype_distribution(df, state: dict) -> list:
    """Plot dtype distribution and type suggestions."""
    images = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dtype counts
    dtype_counts = df.dtypes.astype(str).value_counts()
    colors = {"int64": "#4C72B0", "float64": "#55A868", "object": "#C44E52", "bool": "#8172B2", "datetime64[ns]": "#CCB974"}
    bar_colors = [colors.get(str(d), "#999999") for d in dtype_counts.index]
    axes[0].barh(dtype_counts.index, dtype_counts.values, color=bar_colors)
    axes[0].set_title("Column Types", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Count")

    # Cardinality for object columns
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        cardinality = {col: df[col].nunique() for col in obj_cols}
        sorted_card = dict(sorted(cardinality.items(), key=lambda x: x[1], reverse=True))
        bar_colors_card = ["#C44E52" if v >= 50 else "#55A868" for v in sorted_card.values()]
        axes[1].barh(list(sorted_card.keys()), list(sorted_card.values()), color=bar_colors_card)
        axes[1].set_title("String Column Cardinality", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Unique Values")
        axes[1].axvline(x=50, color="#999999", linestyle="--", alpha=0.5, label="Category threshold")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No string columns", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("String Column Cardinality", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = save_and_show(fig, state, "types.png")
    images.append(path)
    plt.close()

    return images
