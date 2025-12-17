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
# Format: (context_name, display_label, show_delta)
CONTEXTS = [
    ("none", "Comment Only", False),  # First row - no delta
    ("discussion", "\\quad +Discussion", True),
    ("submission-discussion", "\\quad\\quad +Submission", True),
    ("submission-discussion-user", "\\quad\\quad\\quad +User", True),
    ("submission-media-discussion-user", "\\quad\\quad\\quad\\quad +Images", True),
]

# Model bases (for column grouping)
# Format: (base_name, display_name)
MODEL_BASES = [
    ("qwen3-vl-4b", "Qwen3-VL-4B"),
    ("qwen3-vl-8b", "Qwen3-VL-8B"),
    ("qwen3-vl-30b", "Qwen3-VL-30B"),
]

# Model variants (sub-columns within each model)
# Format: (variant_suffix, display_name)
VARIANTS = [
    ("instruct", "Instruct"),
    ("thinking", "Thinking"),
]

# Output file
OUTPUT_FILE = Path(__file__).parent / f"results_table_{SPLIT}.tex"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def find_latest_performance_file(model_base: str, variant: str, split: str, context: str) -> Optional[Path]:
    """Find the latest performance_*.json file for given parameters."""
    base_dir = Path("/data3/zkachwal/reddit-mod-collection-pipeline/output/eval")
    model_name = f"{model_base}-{variant}"
    target_dir = base_dir / model_name / split / context / "baseline"

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
    """Generate LaTeX table with contexts as rows and models/variants as columns."""

    # Collect all data first: {(context, model_base, variant): (acc_value, prev_value)}
    data_grid = {}
    prev_row_values = {(m, v): None for m, _ in MODEL_BASES for v, _ in VARIANTS}
    max_per_col = {(m, v): -1 for m, _ in MODEL_BASES for v, _ in VARIANTS}

    for context_name, _, _ in CONTEXTS:
        current_row = {}
        for model_base, _ in MODEL_BASES:
            for variant_name, _ in VARIANTS:
                perf_file = find_latest_performance_file(model_base, variant_name, SPLIT, context_name)
                if perf_file:
                    perf_data = load_performance_data(perf_file)
                    acc = perf_data.get("metrics", {}).get("overall", {}).get("overall_accuracy", 0)
                else:
                    acc = None
                prev = prev_row_values[(model_base, variant_name)]
                data_grid[(context_name, model_base, variant_name)] = (acc, prev)
                current_row[(model_base, variant_name)] = acc if acc else prev
                if acc and acc > max_per_col[(model_base, variant_name)]:
                    max_per_col[(model_base, variant_name)] = acc
        prev_row_values = current_row

    # Build table
    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append(f"\\begin{{tabular}}{{l{'ll' * len(MODEL_BASES)}}}")
    table.append("\\toprule")

    # Header
    header = " & ".join([""] + [f"\\multicolumn{{2}}{{c}}{{\\textbf{{{d}}}}}" for _, d in MODEL_BASES])
    table.append(header + " \\\\")
    table.append(" ".join([f"\\cmidrule(lr){{{2+i*2}-{3+i*2}}}" for i in range(len(MODEL_BASES))]))
    variant_labels = " & ".join([d for _, d in VARIANTS])
    subheader = " & ".join(["\\textbf{Context}"] + [variant_labels] * len(MODEL_BASES))
    table.append(subheader + " \\\\")
    table.append("\\midrule")

    # Data rows
    for context_name, context_label, show_delta in CONTEXTS:
        cells = [context_label]
        for model_base, _ in MODEL_BASES:
            for variant_name, _ in VARIANTS:
                acc, prev = data_grid[(context_name, model_base, variant_name)]
                if acc is not None:
                    is_best = abs(acc - max_per_col[(model_base, variant_name)]) < 0.0001
                    if show_delta and prev is not None:
                        cell = format_accuracy_with_delta(acc, prev, bold=is_best)
                    else:
                        # First row - no delta
                        acc_str = format_accuracy(acc)
                        if is_best:
                            cell = f"\\textbf{{{acc_str}}}"
                        else:
                            cell = acc_str
                else:
                    cell = "—"
                cells.append(cell)
        table.append(" & ".join(cells) + " \\\\")

    # Baseline row at bottom
    table.append("\\midrule")
    table.append(f"No Moderation & \\multicolumn{{{len(MODEL_BASES)*len(VARIANTS)}}}{{c}}{{50.0}} \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append(f"\\caption{{Overall accuracy (\\%) across different contexts and model sizes on the {SPLIT} split.}}")
    table.append("\\label{tab:results-across-contexts}")
    table.append("\\end{table*}")

    return "\n".join(table)


def main():
    print(f"Generating results table for {SPLIT} split")
    print(f"Contexts: {len(CONTEXTS)} rows")
    print(f"Models: {len(MODEL_BASES)} × {len(VARIANTS)} = {len(MODEL_BASES) * len(VARIANTS)} columns")
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
