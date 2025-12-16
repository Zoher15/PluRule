#!/usr/bin/env python3
"""
Generate LaTeX results table from evaluation performance files.

Table layout:
- Rows: Different contexts
- Columns: 3 model sizes (4B, 8B, 30B) × 2 approaches (Baseline, CoT)
- Values: Overall accuracy (no % signs)

Usage:
    python generate_results_table.py

Modify the CONFIG section to select split and metric.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# SELECT SPLIT
SPLIT = "test"

# Available contexts (rows)
# Format: (context_name, display_label)
CONTEXTS = [
    (None, "No Moderation"),  # Baseline row - 50%
    ("none", "Final Comment"),
    ("discussion", "\\quad +Discussion"),
    ("submission-discussion", "\\quad\\quad +Submission"),
    ("submission-discussion-user", "\\quad\\quad\\quad +User"),
    ("submission-media-discussion-user", "\\quad\\quad\\quad\\quad +Media"),
]

# Models and phrases (columns)
# Format: (model_name, display_name)
MODELS = [
    ("qwen3-vl-4b", "Qwen3-4B"),
    ("qwen3-vl-8b", "Qwen3-8B"),
    ("qwen3-vl-30b", "Qwen3-30B"),
]

PHRASES = [
    ("baseline", "Direct"),
    ("cot_prefill", "CoT"),
]

# Output file
OUTPUT_FILE = Path(__file__).parent / f"results_table_{SPLIT}.tex"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def find_latest_performance_file(model: str, split: str, context: str, phrase: str) -> Optional[Path]:
    """Find the latest performance_*.json file for given parameters."""
    base_dir = Path("/data3/zkachwal/reddit-mod-collection-pipeline/output/eval")
    target_dir = base_dir / model / split / context / phrase

    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return None

    perf_files = sorted(target_dir.glob("performance_*.json"))
    if not perf_files:
        print(f"No performance files found in {target_dir}")
        return None

    return perf_files[-1]  # Latest file


def load_performance_data(file_path: Path) -> Dict:
    """Load performance JSON data."""
    with open(file_path) as f:
        return json.load(f)


def format_accuracy(value: float, decimals: int = 1) -> str:
    """Format accuracy as percentage without % sign."""
    return f"{value * 100:.{decimals}f}"


def format_accuracy_with_delta(value: float, prev_value: float, decimals: int = 1, bold: bool = False) -> str:
    """Format accuracy with delta from previous row in tiny font."""
    acc = value * 100
    prev_acc = prev_value * 100
    delta = acc - prev_acc
    sign = "+" if delta >= 0 else ""
    acc_str = f"{acc:.{decimals}f}"
    if bold:
        acc_str = f"\\textbf{{{acc_str}}}"
    return f"{acc_str} {{\\tiny ({sign}{delta:.{decimals}f})}}"


def generate_results_table() -> str:
    """Generate LaTeX table with contexts as rows and models/phrases as columns."""

    # Collect all data first: {(context, model, phrase): (acc_value, prev_value)}
    data_grid = {}
    prev_row_values = {(m, p): 0.5 for m, _ in MODELS for p, _ in PHRASES}
    max_per_col = {(m, p): -1 for m, _ in MODELS for p, _ in PHRASES}

    for context_name, _ in CONTEXTS:
        if context_name is None:
            continue
        current_row = {}
        for model_name, _ in MODELS:
            for phrase_name, _ in PHRASES:
                perf_file = find_latest_performance_file(model_name, SPLIT, context_name, phrase_name)
                if perf_file:
                    perf_data = load_performance_data(perf_file)
                    acc = perf_data.get("metrics", {}).get("overall", {}).get("overall_accuracy", 0)
                else:
                    acc = None
                prev = prev_row_values[(model_name, phrase_name)]
                data_grid[(context_name, model_name, phrase_name)] = (acc, prev)
                current_row[(model_name, phrase_name)] = acc if acc else prev
                if acc and acc > max_per_col[(model_name, phrase_name)]:
                    max_per_col[(model_name, phrase_name)] = acc
        prev_row_values = current_row

    # Build table
    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append(f"\\begin{{tabular}}{{l{'cc' * len(MODELS)}}}")
    table.append("\\toprule")

    # Header
    header = " & ".join([""] + [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{d}}}}}" for _, d in MODELS])
    table.append(header + " \\\\")
    table.append(" ".join([f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(len(MODELS))]))
    subheader = " & ".join(["\\textbf{Context}"] + ["Direct & CoT"] * len(MODELS))
    table.append(subheader + " \\\\")
    table.append("\\midrule")

    # Baseline row
    table.append(f"No Moderation & \\multicolumn{{{len(MODELS)*2}}}{{c}}{{50.0}} \\\\")
    table.append("\\midrule")

    # Data rows
    for context_name, context_label in CONTEXTS:
        if context_name is None:
            continue
        cells = [context_label]
        for model_name, _ in MODELS:
            for phrase_name, _ in PHRASES:
                acc, prev = data_grid[(context_name, model_name, phrase_name)]
                if acc is not None:
                    is_best = abs(acc - max_per_col[(model_name, phrase_name)]) < 0.0001
                    cell = format_accuracy_with_delta(acc, prev, bold=is_best)
                else:
                    cell = "—"
                cells.append(cell)
        table.append(" & ".join(cells) + " \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append(f"\\caption{{Overall accuracy (\\%) across different contexts and model sizes on the {SPLIT} split.}}")
    table.append("\\label{tab:results-across-contexts}")
    table.append("\\end{table*}")

    return "\n".join(table)


def main():
    print(f"Generating results table for {SPLIT} split")
    print(f"Contexts: {len(CONTEXTS)} rows")
    print(f"Models: {len(MODELS)} × {len(PHRASES)} = {len(MODELS) * len(PHRASES)} columns")
    print()

    # Generate table
    latex_table = generate_results_table()

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(latex_table)

    print(f"✓ Saved to: {OUTPUT_FILE}")
    print("\nLatex table:")
    print(latex_table)

    return 0


if __name__ == "__main__":
    exit(main())
