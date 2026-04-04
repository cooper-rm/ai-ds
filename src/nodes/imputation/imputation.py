"""
Imputation analysis node (profile phase).

Reads evidence from missing.py, asks LLM to select methods,
runs all selected methods, scores each by regression line distortion,
and records the winner per column.

Output stored at state["nodes"]["imputation"] — consumed by synthesis
and preprocessing/impute.py.
"""
from src.report import narrate, add_section
from src.terminal import print_info, print_detail, print_warning, prompt_choice

from src.nodes.imputation.select import select_methods
from src.nodes.imputation.score  import score_distortion
from src.nodes.imputation.plot   import plot_comparison
import src.nodes.imputation.methods.simple      as simple_methods
import src.nodes.imputation.methods.statistical as stat_methods
import src.nodes.imputation.methods.deep        as deep_methods

# Maps method name → (module, function_name)
METHOD_DISPATCH = {
    "mean":                  (simple_methods, "mean"),
    "median":                (simple_methods, "median"),
    "mode":                  (simple_methods, "mode"),
    "grouped_median":        (simple_methods, "grouped_median"),
    "knn":                   (simple_methods, "knn"),
    "regression":            (stat_methods,   "regression"),
    "stochastic_regression": (stat_methods,   "stochastic_regression"),
    "pmm":                   (stat_methods,   "pmm"),
    "hotdeck":               (stat_methods,   "hotdeck"),
    "mice":                  (stat_methods,   "mice"),
    "missforest":            (stat_methods,   "missforest"),
    "em":                    (stat_methods,   "em"),
    "softimpute":            (stat_methods,   "softimpute"),
    "gain":                  (deep_methods,   "gain"),
    "mida":                  (deep_methods,   "mida"),
    "hivae":                 (deep_methods,   "hivae"),
}


def imputation(state: dict) -> dict:
    """Test candidate imputation methods and pick the winner per column."""
    df        = state["data"]
    missing   = state["nodes"].get("missing", {})
    evidence  = missing.get("results", {})

    # Only process columns flagged for imputation (not drop)
    to_impute = {
        col: ev for col, ev in evidence.items()
        if ev.get("recommendation") != "drop"
    }

    if not to_impute:
        state["nodes"]["imputation"] = {"status": "nothing_to_impute", "results": {}}
        print_info("no columns to impute")
        return state

    # ── LLM selects methods ───────────────────────────────────────────────────
    selections = select_methods(to_impute, intensity="full")

    # ── Deep learning gate ────────────────────────────────────────────────────
    DEEP_METHODS = {"gain", "mida", "hivae"}
    deep_cols = {
        col: [m for m in methods if m in DEEP_METHODS]
        for col, methods in selections.items()
        if any(m in DEEP_METHODS for m in methods)
    }

    if deep_cols:
        col_list = "\n".join(
            f"  [dim]•[/dim] [white]{col}[/white]: {', '.join(ms)}"
            for col, ms in deep_cols.items()
        )
        choice = prompt_choice(
            title="Deep Learning Imputation Selected",
            body=(
                "The imputation agent has selected methods that include deep learning "
                f"([bold]GAIN, MIDA, HI-VAE[/bold]) for the following columns:\n\n"
                f"{col_list}\n\n"
                "These methods train neural networks and can take significantly longer, "
                "but may produce higher-quality imputations for complex missing data patterns."
            ),
            options=[
                ("y", "Continue with deep learning (higher quality, slower)"),
                ("n", "Use statistical methods only (faster, still good quality)"),
            ],
        )
        if choice == "n":
            # Strip deep learning methods from all selections, keep at least one fallback
            STAT_FALLBACK = "mice"
            for col in selections:
                selections[col] = [m for m in selections[col] if m not in DEEP_METHODS]
                if not selections[col]:
                    selections[col] = [STAT_FALLBACK]
            print_info("deep learning methods removed — using statistical methods only")

    results = {}
    images  = []

    for col, ev in to_impute.items():
        methods = selections.get(col, ["median"])  # fallback if LLM missed it
        print_info(f"{col}: testing {methods}")

        method_results = {}

        for method in methods:
            try:
                mod, fn = METHOD_DISPATCH[method]
                imputed = getattr(mod, fn)(df, col, ev)
                if imputed is None:
                    print_warning(f"  {method}: returned None, skipping")
                    continue
                score = score_distortion(df, col, imputed, ev)
                method_results[method] = {"imputed": imputed, "score": score}
                ks = score.get('ks_statistic', 0)
                shape = score.get('shape_penalty', 0)
                print_info(
                    f"  {method}: Δslope={score['delta_slope']:.4f}"
                    f"  ΔR²={score['delta_r2']:.4f}"
                    f"  KS={ks:.4f}"
                    f"  shape={shape:.4f}"
                    f"  total={score['total']:.4f}"
                )
            except Exception as e:
                print_warning(f"  {method}: failed — {e}")

        if not method_results:
            # All methods failed — use median/mode as last resort
            fallback = "median" if ev.get("is_numeric") else "mode"
            mod, fn  = METHOD_DISPATCH[fallback]
            imputed  = getattr(mod, fn)(df, col, ev)
            if imputed is not None:
                score = score_distortion(df, col, imputed, ev)
                method_results[fallback] = {"imputed": imputed, "score": score}
                print_warning(f"  all methods failed — fell back to {fallback}")

        if not method_results:
            print_warning(f"  {col}: could not impute, skipping")
            continue

        best_method = min(method_results, key=lambda m: method_results[m]["score"]["total"])
        best        = method_results[best_method]
        print_info(f"  [green]winner: {best_method}[/green]  (distortion={best['score']['total']:.4f})")

        img = plot_comparison(df, col, method_results, best_method, ev, state)
        if img:
            images.append(img)

        results[col] = {
            "missing_pct":          ev["missing_pct"],
            "missingness_type":     ev["missingness_type"],
            "methods_tested":       list(method_results.keys()),
            "method_scores":        {m: r["score"] for m, r in method_results.items()},
            "winner":               best_method,
            "winner_score":         best["score"],
            "group_by":             ev.get("top_categorical_correlate"),
            "regressors":           ev.get("mar_correlates", [])[:2],
        }

    print_detail("columns analyzed", len(results))

    state["nodes"]["imputation"] = {
        "status":  "analyzed",
        "results": results,
        "images":  [str(p) for p in images],
    }

    narrative = narrate("Imputation Method Selection", {
        "summary": {
            col: {
                "winner":          r["winner"],
                "methods_tested":  r["methods_tested"],
                "distortion":      r["winner_score"]["total"],
            }
            for col, r in results.items()
        }
    })
    add_section(state, "Imputation Method Selection", narrative, images)
    return state
