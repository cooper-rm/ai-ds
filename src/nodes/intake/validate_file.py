import json

from src.llm.client import ask


def validate_file(state: dict) -> dict:
    """LLM gate after analyze_file. Checks for issues before loading."""
    file_info = state["nodes"]["analyze_file"]

    prompt = f"""You are validating a file before loading it into a data science pipeline.

Here is the file analysis:
{json.dumps(file_info, indent=2)}

Check for potential issues:
- Is the file type supported and common for data science?
- Is the file size concerning for memory?
- Does the row/column count look reasonable?
- Any red flags in the column names?

Respond with EXACTLY this JSON format, nothing else:
{{
  "proceed": true or false,
  "issues": ["list of issues if any, empty if none"],
  "recommendations": ["list of recommendations if any, empty if none"]
}}"""

    response = ask(prompt, system="You are a data validation assistant. Respond only in JSON.")

    # Parse the LLM response
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"proceed": True, "issues": [], "recommendations": ["LLM response was not valid JSON, proceeding anyway"]}

    state["nodes"]["validate_file"] = result

    if result["proceed"]:
        print("   Validation passed")
    else:
        print("   Validation FAILED")
        for issue in result["issues"]:
            print(f"   - {issue}")

    if result["recommendations"]:
        for rec in result["recommendations"]:
            print(f"   Recommendation: {rec}")

    if not result["proceed"]:
        raise RuntimeError("File validation failed. Check state['nodes']['validate_file'] for details.")

    return state
