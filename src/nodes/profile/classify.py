import json

from src.llm.client import ask
from src.terminal import print_info, print_detail, llm_spinner


def classify(state: dict) -> dict:
    """Classify columns by analytical type using LLM reasoning."""
    df = state["data"]

    # Gather evidence per column
    evidence = {}
    for col in df.columns:
        series = df[col]
        nunique = series.nunique()
        non_null = len(series.dropna())

        col_info = {
            "dtype": str(series.dtype),
            "nunique": nunique,
            "cardinality_ratio": round(nunique / len(df), 4) if len(df) > 0 else 0,
            "null_count": int(series.isnull().sum()),
            "sample_values": [str(v) for v in series.dropna().sample(min(5, non_null), random_state=42).tolist()],
        }

        if series.dtype in ("int64", "float64"):
            col_info["min"] = float(series.min())
            col_info["max"] = float(series.max())
            col_info["mean"] = round(float(series.mean()), 2)

        evidence[col] = col_info

    # Ask LLM to classify
    prompt = f"""You are classifying columns in a dataset for exploratory data analysis.

For each column, determine its analytical type based on the column name, dtype, unique values, and sample values.

Column evidence:
{json.dumps(evidence, indent=2)}

Classify each column as exactly one of:
- "continuous" — numeric, many possible values (e.g. age, price, temperature)
- "discrete" — numeric, limited set of values (e.g. count of siblings, number of rooms)
- "categorical_nominal" — unordered categories (e.g. sex, color, city)
- "categorical_ordinal" — ordered categories (e.g. class rank, education level, rating)
- "identifier" — unique or near-unique ID, not useful for analysis (e.g. passenger ID, ticket number)
- "text" — free-form text, names, descriptions
- "datetime" — dates or timestamps
- "binary" — exactly 2 values (e.g. yes/no, 0/1, true/false)

Respond with EXACTLY this JSON format, nothing else:
{{
  "classifications": {{
    "column_name": {{
      "type": "one of the types above",
      "reason": "brief explanation"
    }}
  }}
}}"""

    with llm_spinner("Classifying columns"):
        response = ask(prompt, system="You are a data science assistant. Respond only in JSON.")

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        import re
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"classifications": {}}

    classifications = result.get("classifications", {})

    # Build summary counts
    type_counts = {}
    for col, info in classifications.items():
        col_type = info.get("type", "unknown")
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

    state["nodes"]["classify"] = {
        "classifications": classifications,
        "type_counts": type_counts,
    }

    for col, info in classifications.items():
        print_info(f"{col}: {info['type']}  —  {info['reason']}")
    print_detail("summary", "  ".join(f"{k} ×{v}" for k, v in type_counts.items()))

    from src.report import narrate, add_section
    narrative = narrate("Column Classification", {
        "type_counts": type_counts,
        "classifications": {col: info["type"] for col, info in classifications.items()},
    })
    add_section(state, "Column Classification", narrative)

    return state
