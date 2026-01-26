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
from typing import Dict, Optional

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

# Models with their variants
# Format: (model_base, display_name, [(variant_suffix, variant_display), ...])
MODELS = [
    ("qwen3-vl-4b", "Qwen3-VL-4B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("qwen3-vl-8b", "Qwen3-VL-8B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("qwen3-vl-30b", "Qwen3-VL-30B", [("instruct", "Instruct"), ("thinking", "Thinking")]),
    ("gpt5.2", "GPT-5.2", [("low", "Low"), ("high", "High")]),
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
    # Round first, then compute delta (so 57.54 vs 57.55 shows as +0.1 not +0.0)
    acc_rounded = round(acc, decimals)
    prev_rounded = round(prev_acc, decimals)
    delta = acc_rounded - prev_rounded
    sign = "+" if delta >= 0 else ""
    acc_str = f"{acc_rounded:.{decimals}f}"
    if bold:
        acc_str = f"\\textbf{{{acc_str}}}"
    return f"{acc_str} {{\\tiny ({sign}{delta:.{decimals}f})}}"


def generate_results_table() -> str:
    """Generate LaTeX table with contexts as rows and models/variants as columns."""

    # Build flat list of (model_base, variant) for iteration
    all_columns = []
    for model_base, _, variants in MODELS:
        for variant_name, _ in variants:
            all_columns.append((model_base, variant_name))

    # Collect all data first: {(context, model_base, variant): (acc_value, prev_value)}
    data_grid = {}
    prev_row_values = {col: None for col in all_columns}
    max_per_col = {col: -1 for col in all_columns}
    max_ci = 0.0  # Track max CI across all data

    for context_name, _, _ in CONTEXTS:
        current_row = {}
        for model_base, variant_name in all_columns:
            perf_file = find_latest_performance_file(model_base, variant_name, SPLIT, context_name)
            if perf_file:
                perf_data = load_performance_data(perf_file)
                overall = perf_data.get("metrics", {}).get("overall", {})
                acc = overall.get("overall_accuracy", 0)
                # Extract CI: [lower, upper] -> ± (upper - lower) / 2
                ci = overall.get("overall_accuracy_ci")
                if ci and len(ci) == 2:
                    ci_half_width = (ci[1] - ci[0]) / 2 * 100  # Convert to percentage
                    max_ci = max(max_ci, ci_half_width)
            else:
                acc = None
            prev = prev_row_values[(model_base, variant_name)]
            data_grid[(context_name, model_base, variant_name)] = (acc, prev)
            current_row[(model_base, variant_name)] = acc if acc else prev
            if acc and acc > max_per_col[(model_base, variant_name)]:
                max_per_col[(model_base, variant_name)] = acc
        prev_row_values = current_row

    # Count total columns
    total_cols = sum(len(variants) for _, _, variants in MODELS)

    # Build table
    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append("\\setlength{\\tabcolsep}{3.75pt}")
    table.append(f"\\begin{{tabular}}{{l{'l' * total_cols}}}")  # Left-align data columns
    table.append("\\toprule")

    # Header row 1: Model names with column spans (centered)
    header_parts = ["\\textbf{Models}"]
    col_idx = 2  # Start at column 2 (column 1 is Context)
    cmidrules = []
    for model_base, display_name, variants in MODELS:
        num_variants = len(variants)
        header_parts.append(f"\\multicolumn{{{num_variants}}}{{c}}{{\\textbf{{{display_name}}}}}")
        cmidrules.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + num_variants - 1}}}")
        col_idx += num_variants
    table.append(" & ".join(header_parts) + " \\\\")
    table.append(" ".join(cmidrules))

    # Header row 2: Variant names (centered via multicolumn)
    subheader_parts = ["\\textbf{Variants}"]
    for model_base, _, variants in MODELS:
        for _, variant_display in variants:
            subheader_parts.append(f"\\multicolumn{{1}}{{c}}{{{variant_display}}}")
    table.append(" & ".join(subheader_parts) + " \\\\")
    table.append("\\midrule")

    # Data rows
    for context_name, context_label, show_delta in CONTEXTS:
        cells = [context_label]
        for model_base, variant_name in all_columns:
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
    table.append(f"No Moderation & \\multicolumn{{{total_cols}}}{{c}}{{50.0}} \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    ci_str = f"{max_ci:.1f}" if max_ci > 0 else "1.1"
    table.append(f"\\caption{{Overall accuracy (\\%) across different models and contexts on the {SPLIT} set. Numbers in parentheses indicate differences compared to accuracy values in the previous row. All values have 95\\% CI of $\\pm {ci_str}\\%$. See the Appendix for a breakdown of moderated and unmoderated accuracy.}}")
    table.append("\\label{tab:results-across-contexts}")
    table.append("\\end{table*}")

    return "\n".join(table)


def generate_appendix_table() -> str:
    """Generate LaTeX table with violating accuracy/compliant accuracy/overall accuracy breakdown for appendix (transposed: contexts as columns)."""

    # Build flat list of (model_base, variant, display_name, variant_display) for rows
    all_rows = []
    for model_base, display_name, variants in MODELS:
        for variant_name, variant_display in variants:
            all_rows.append((model_base, variant_name, display_name, variant_display))

    # Columns are contexts (5 columns)
    num_context_cols = len(CONTEXTS)

    # Collect all data: {(model_base, variant, context): (mod_acc, unmod_acc, overall_acc)}
    data_grid = {}
    max_ci = 0.0  # Track max CI across all data
    # Track max per row (model/variant) for mod, unmod, and overall
    max_mod_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}
    max_unmod_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}
    max_overall_per_row = {(mb, vn): -1 for mb, vn, _, _ in all_rows}

    for model_base, variant_name, _, _ in all_rows:
        for context_name, _, _ in CONTEXTS:
            perf_file = find_latest_performance_file(model_base, variant_name, SPLIT, context_name)
            if perf_file:
                perf_data = load_performance_data(perf_file)
                overall_metrics = perf_data.get("metrics", {}).get("overall", {})
                mod_acc = overall_metrics.get("violating_accuracy")
                unmod_acc = overall_metrics.get("compliant_accuracy")
                overall_acc = overall_metrics.get("overall_accuracy")
                # Extract CI
                mod_ci = overall_metrics.get("violating_accuracy_ci")
                if mod_ci and len(mod_ci) == 2:
                    ci_half_width = (mod_ci[1] - mod_ci[0]) / 2 * 100
                    max_ci = max(max_ci, ci_half_width)
                unmod_ci = overall_metrics.get("compliant_accuracy_ci")
                if unmod_ci and len(unmod_ci) == 2:
                    ci_half_width = (unmod_ci[1] - unmod_ci[0]) / 2 * 100
                    max_ci = max(max_ci, ci_half_width)
                overall_ci = overall_metrics.get("overall_accuracy_ci")
                if overall_ci and len(overall_ci) == 2:
                    ci_half_width = (overall_ci[1] - overall_ci[0]) / 2 * 100
                    max_ci = max(max_ci, ci_half_width)
            else:
                mod_acc, unmod_acc, overall_acc = None, None, None
            data_grid[(model_base, variant_name, context_name)] = (mod_acc, unmod_acc, overall_acc)
            if mod_acc is not None and mod_acc > max_mod_per_row[(model_base, variant_name)]:
                max_mod_per_row[(model_base, variant_name)] = mod_acc
            if unmod_acc is not None and unmod_acc > max_unmod_per_row[(model_base, variant_name)]:
                max_unmod_per_row[(model_base, variant_name)] = unmod_acc
            if overall_acc is not None and overall_acc > max_overall_per_row[(model_base, variant_name)]:
                max_overall_per_row[(model_base, variant_name)] = overall_acc

    # Build table
    table = []
    table.append("\\begin{table*}[t]")
    table.append("\\centering")
    table.append("\\setlength{\\tabcolsep}{4.5pt}")
    # Columns: Model | Variant | Metric | Context1 | Context2 | ... | Context5
    table.append(f"\\begin{{tabular}}{{lll{'l' * num_context_cols}}}")
    table.append("\\toprule")

    # Header row: Model, Variant, Metric, then context names (strip \quad formatting)
    context_headers = []
    for _, ctx_label, _ in CONTEXTS:
        clean = ctx_label.replace("\\quad", "").strip()
        context_headers.append(clean)
    header_parts = ["\\textbf{Model}", "\\textbf{Variant}", "\\textbf{Metric}"] + [f"\\textbf{{{h}}}" for h in context_headers]
    table.append(" & ".join(header_parts) + " \\\\")
    table.append("\\midrule")

    # Data rows: grouped by model, then variant, with Vio Rec, Com Rec, and Acc rows
    # Track which model we're in to use multirow for model names
    current_model = None
    variant_idx_in_model = 0

    for i, (model_base, variant_name, display_name, variant_display) in enumerate(all_rows):
        # Check if this is a new model
        if display_name != current_model:
            current_model = display_name
            variant_idx_in_model = 0
            # Count how many variants this model has (for multirow span)
            num_variants = len([r for r in all_rows if r[2] == display_name])
            model_row_span = num_variants * 3  # 3 rows per variant (Vio Rec + Com Rec + Acc)
        else:
            variant_idx_in_model += 1

        # Violating Recall row
        mod_cells = []
        prev_mod = None
        for context_name, _, show_delta in CONTEXTS:
            mod_acc, _, _ = data_grid[(model_base, variant_name, context_name)]
            if mod_acc is not None:
                is_best = abs(mod_acc - max_mod_per_row[(model_base, variant_name)]) < 0.0001
                if show_delta and prev_mod is not None:
                    cell = format_accuracy_with_delta(mod_acc, prev_mod, bold=is_best)
                else:
                    acc_str = format_accuracy(mod_acc)
                    cell = f"\\textbf{{{acc_str}}}" if is_best else acc_str
                prev_mod = mod_acc
            else:
                cell = "—"
            mod_cells.append(cell)

        # Compliant Recall row
        unmod_cells = []
        prev_unmod = None
        for context_name, _, show_delta in CONTEXTS:
            _, unmod_acc, _ = data_grid[(model_base, variant_name, context_name)]
            if unmod_acc is not None:
                is_best = abs(unmod_acc - max_unmod_per_row[(model_base, variant_name)]) < 0.0001
                if show_delta and prev_unmod is not None:
                    cell = format_accuracy_with_delta(unmod_acc, prev_unmod, bold=is_best)
                else:
                    acc_str = format_accuracy(unmod_acc)
                    cell = f"\\textbf{{{acc_str}}}" if is_best else acc_str
                prev_unmod = unmod_acc
            else:
                cell = "—"
            unmod_cells.append(cell)

        # Accuracy row
        overall_cells = []
        prev_overall = None
        for context_name, _, show_delta in CONTEXTS:
            _, _, overall_acc = data_grid[(model_base, variant_name, context_name)]
            if overall_acc is not None:
                is_best = abs(overall_acc - max_overall_per_row[(model_base, variant_name)]) < 0.0001
                if show_delta and prev_overall is not None:
                    cell = format_accuracy_with_delta(overall_acc, prev_overall, bold=is_best)
                else:
                    acc_str = format_accuracy(overall_acc)
                    cell = f"\\textbf{{{acc_str}}}" if is_best else acc_str
                prev_overall = overall_acc
            else:
                cell = "—"
            overall_cells.append(cell)

        # Build the rows with proper multirow nesting
        if variant_idx_in_model == 0:
            # First variant of this model - include model multirow
            model_cell = f"\\multirow{{{model_row_span}}}{{*}}{{{display_name}}}"
        else:
            # Continuation of model - empty cell
            model_cell = ""

        # Variant always uses multirow spanning 3 rows (Vio Rec + Com Rec + Acc)
        variant_cell = f"\\multirow{{3}}{{*}}{{{variant_display}}}"

        table.append(f"{model_cell} & {variant_cell} & Vio Rec & " + " & ".join(mod_cells) + " \\\\")
        table.append(" & & Com Rec & " + " & ".join(unmod_cells) + " \\\\")
        table.append(" & & Acc & " + " & ".join(overall_cells) + " \\\\")

        # Add separators
        if i < len(all_rows) - 1:
            next_is_new_model = all_rows[i + 1][2] != display_name
            if next_is_new_model:
                table.append("\\midrule")
            else:
                # Separator between variants of same model
                table.append(f"\\cmidrule(lr){{2-{3 + num_context_cols}}}")

    # Baseline row: 0% for Vio Rec, 100% for Com Rec, 50% for Acc (spanning all context columns)
    table.append("\\midrule")
    table.append(f"\\multirow{{3}}{{*}}{{No Moderation}} & & Vio Rec & \\multicolumn{{{num_context_cols}}}{{c}}{{0.0}} \\\\")
    table.append(f" & & Com Rec & \\multicolumn{{{num_context_cols}}}{{c}}{{100.0}} \\\\")
    table.append(f" & & Acc & \\multicolumn{{{num_context_cols}}}{{c}}{{50.0}} \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    ci_str = f"{max_ci:.1f}" if max_ci > 0 else "1.5"
    table.append(f"\\caption{{Violating recall, compliant recall, and accuracy (\\%) across different models and contexts on the {SPLIT} set. Numbers in parentheses indicate differences compared to accuracy values in the previous column. All values have 95\\% CI of $\\pm {ci_str}\\%$.}}")
    table.append("\\label{tab:results-mod-unmod}")
    table.append("\\end{table*}")

    return "\n".join(table)


def main():
    total_cols = sum(len(variants) for _, _, variants in MODELS)
    print(f"Generating results tables for {SPLIT} split")
    print(f"Contexts: {len(CONTEXTS)} rows")
    print(f"Models: {len(MODELS)} models, {total_cols} total columns")
    print()

    # Generate main table
    latex_table = generate_results_table()
    with open(OUTPUT_FILE, 'w') as f:
        f.write(latex_table)
    print(f"Saved main table to: {OUTPUT_FILE}")
    print("\nMain table:")
    print(latex_table)

    # Generate appendix table with mod/unmod breakdown
    appendix_file = Path(__file__).parent / f"results_table_appendix_{SPLIT}.tex"
    appendix_table = generate_appendix_table()
    with open(appendix_file, 'w') as f:
        f.write(appendix_table)
    print(f"\nSaved appendix table to: {appendix_file}")
    print("\nAppendix table:")
    print(appendix_table)

    return 0


if __name__ == "__main__":
    exit(main())
