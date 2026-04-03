"""LLM-based imputation method selection."""
import json
import re

from src.llm.client import ask
from src.terminal import print_info, llm_spinner

# Full method menu with tier and dtype compatibility
METHOD_MENU = {
    # fast
    "mean":                  {"tier": "fast",     "dtype": "numeric", "desc": "Column mean. Baseline for MCAR symmetric numeric."},
    "median":                {"tier": "fast",     "dtype": "numeric", "desc": "Column median. Robust to outliers and skew."},
    "mode":                  {"tier": "fast",     "dtype": "any",     "desc": "Most frequent value. Best for categorical."},
    "grouped_median":        {"tier": "fast",     "dtype": "numeric", "desc": "Median within groups of a correlated categorical column."},
    "knn":                   {"tier": "fast",     "dtype": "any",     "desc": "K-nearest neighbors using feature similarity."},
    # standard
    "regression":            {"tier": "standard", "dtype": "numeric", "desc": "OLS regression prediction from correlated columns."},
    "stochastic_regression": {"tier": "standard", "dtype": "numeric", "desc": "Regression + noise to preserve variance."},
    "pmm":                   {"tier": "standard", "dtype": "numeric", "desc": "Predictive mean matching — prediction mapped to closest observed donor."},
    "hotdeck":               {"tier": "standard", "dtype": "any",     "desc": "Fill from most similar complete row (Euclidean distance)."},
    "mice":                  {"tier": "standard", "dtype": "numeric", "desc": "Iterative multivariate imputation (sklearn IterativeImputer)."},
    "missforest":            {"tier": "standard", "dtype": "any",     "desc": "Random forest iterative imputation for mixed types."},
    "em":                    {"tier": "standard", "dtype": "numeric", "desc": "Expectation-maximization under multivariate normal."},
    "softimpute":            {"tier": "standard", "dtype": "numeric", "desc": "Matrix completion via nuclear norm / SVD shrinkage."},
    # full
    "gain":                  {"tier": "full",     "dtype": "numeric", "desc": "GAIN — GAN-based: generator fills missing, discriminator verifies."},
    "mida":                  {"tier": "full",     "dtype": "numeric", "desc": "MIDA — denoising autoencoder reconstructs full row from corrupted input."},
    "hivae":                 {"tier": "full",     "dtype": "any",     "desc": "HI-VAE — variational autoencoder for heterogeneous incomplete data."},
}

TIER_ORDER = {"fast": 0, "standard": 1, "full": 2}


def select_methods(evidence: dict, intensity: str) -> dict:
    """
    Ask the LLM to pick 3-5 imputation methods per column.
    Returns {col: [method1, method2, ...]}
    """
    available = {
        k: v for k, v in METHOD_MENU.items()
        if TIER_ORDER[v["tier"]] <= TIER_ORDER[intensity]
    }

    menu_desc = {
        name: f"[{info['tier']}] [{info['dtype']}] {info['desc']}"
        for name, info in available.items()
    }

    # Strip imputed columns to just the evidence fields (no large objects)
    evidence_clean = {
        col: {k: v for k, v in ev.items() if k not in ("recommendation",)}
        for col, ev in evidence.items()
        if ev.get("recommendation") != "drop"
    }

    if not evidence_clean:
        return {}

    prompt = f"""You are selecting imputation methods to test for each missing column.

Intensity level: {intensity}

Available methods:
{json.dumps(menu_desc, indent=2)}

Column evidence:
{json.dumps(evidence_clean, indent=2)}

Selection rules:
- Pick 3-5 methods per column
- dtype "numeric" methods only for numeric columns; dtype "any" works for all
- MAR with numeric correlates → prefer multivariate methods (regression, mice, knn, missforest, pmm, gain, mida)
- MAR with categorical correlate → include grouped_median
- MCAR → simple baselines are appropriate (mean/median/mode)
- MNAR_suspected → prefer model-based (missforest, mice, gain, hivae)
- At intensity=full: include at least one deep learning method (gain/mida/hivae) for numeric MAR/MNAR columns
- Always include at least one simple baseline for comparison
- For categorical columns: mode, knn, hotdeck, missforest, hivae are compatible

Respond with EXACTLY this JSON, nothing else:
{{
  "selections": {{
    "column_name": ["method1", "method2", "method3"],
    ...
  }},
  "reasoning": {{
    "column_name": "one sentence",
    ...
  }}
}}"""

    with llm_spinner("Selecting imputation methods"):
        response = ask(prompt, system="You are a data science assistant. Respond only in JSON.")

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        result = json.loads(match.group(1)) if match else {"selections": {}}

    for col, reason in result.get("reasoning", {}).items():
        print_info(f"  {col}: {reason}")

    # Validate — only keep methods that are in the available set
    return {
        col: [m for m in methods if m in available]
        for col, methods in result.get("selections", {}).items()
    }
