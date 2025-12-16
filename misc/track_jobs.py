#!/usr/bin/env python3
"""
Job Tracker for Reddit Moderation Evaluation Pipeline

Scans logs and output directories to track job status:
- COMPLETED: performance_*.json exists in output/eval/
- IN_PROGRESS: evaluation_*.log exists without corresponding performance file
- PENDING: Neither exists for the combination

Usage:
    python track_jobs.py [--json] [--csv] [--output OUTPUT_FILE]
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Track job status across evaluations')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--csv', action='store_true', help='Output results as CSV')
    parser.add_argument('--latex', action='store_true', help='Output publication-ready LaTeX table')
    parser.add_argument('--output-dir', type=str,
                       default='/data3/zkachwal/reddit-mod-collection-pipeline/output/eval',
                       help='Path to output directory')
    parser.add_argument('--logs-dir', type=str,
                       default='/data3/zkachwal/reddit-mod-collection-pipeline/logs/eval',
                       help='Path to logs directory')
    parser.add_argument('--output', type=str,
                       default='/data3/zkachwal/reddit-mod-collection-pipeline/eval/job_status.txt',
                       help='Output file path')
    return parser.parse_args()


def get_time_ago(timestamp):
    """Convert timestamp to human-readable 'X hours/mins ago' format."""
    now = datetime.now()
    diff = now - timestamp

    total_seconds = int(diff.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h ago"
    elif hours > 0:
        return f"{hours}h {minutes}m ago"
    elif minutes > 0:
        return f"{minutes}m ago"
    else:
        return "just now"


def parse_directory_path(path, base_dir):
    """
    Parse directory path to extract job parameters.

    Expected structure:
    {model}/{split}/{context}/{phrase_mode}/

    Where phrase_mode is either:
    - 'baseline' (for baseline phrase)
    - '{phrase}_{mode}' (for other phrases like 'cot_prefill')

    Returns:
        tuple: (model, split, context, phrase, mode) or None if invalid
    """
    try:
        rel_path = path.relative_to(base_dir)
        parts = rel_path.parts

        if len(parts) < 4:
            return None

        model = parts[0]
        split = parts[1]
        context = parts[2]
        phrase_mode = parts[3]

        # Parse phrase_mode
        if phrase_mode == 'baseline':
            phrase = 'baseline'
            mode = 'prefill'  # baseline always uses prefill
        elif '_' in phrase_mode:
            phrase, mode = phrase_mode.rsplit('_', 1)
        else:
            return None

        return (model, split, context, phrase, mode)

    except (ValueError, IndexError):
        return None


def is_debug_performance(perf_file):
    """Check if a performance file is from a debug run (only 5 pairs)."""
    try:
        with open(perf_file, 'r') as f:
            data = json.load(f)
            total_pairs = data.get('metrics', {}).get('overall', {}).get('total_pairs')
            # Debug mode uses only 5 thread pairs
            return total_pairs == 5
    except Exception:
        pass
    return False


def scan_completed_jobs(output_dir):
    """Scan output directory for completed jobs (performance_*.json files)."""
    completed = {}

    for perf_file in Path(output_dir).rglob('performance_*.json'):
        parsed = parse_directory_path(perf_file.parent, output_dir)
        if not parsed:
            continue

        model, split, context, phrase, mode = parsed

        # Skip if this is a debug performance file
        if is_debug_performance(perf_file):
            continue

        # Extract timestamp from filename (performance_YYYYMMDD_HHMMSS.json)
        filename = perf_file.name
        try:
            # Extract date and time from filename
            parts = filename.replace('performance_', '').replace('.json', '').split('_')
            if len(parts) >= 2:
                date_str = parts[0]
                time_str = parts[1]
                file_timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            else:
                # Fallback to file modification time
                file_timestamp = datetime.fromtimestamp(perf_file.stat().st_mtime)
        except (ValueError, IndexError):
            # Fallback to file modification time
            file_timestamp = datetime.fromtimestamp(perf_file.stat().st_mtime)

        # Get file modification time
        mtime = datetime.fromtimestamp(perf_file.stat().st_mtime)

        # Read overall accuracy from the performance file
        overall_accuracy = None
        try:
            with open(perf_file, 'r') as f:
                data = json.load(f)
                overall_accuracy = data.get('metrics', {}).get('overall', {}).get('overall_accuracy')
        except Exception:
            pass

        key = (model, split, context, phrase, mode)

        # Keep only the most recent file based on filename timestamp
        if key not in completed or file_timestamp > completed[key]['file_timestamp']:
            completed[key] = {
                'status': 'COMPLETED',
                'file': str(perf_file),
                'file_timestamp': file_timestamp,
                'timestamp': mtime,
                'time_ago': get_time_ago(mtime),
                'overall_accuracy': overall_accuracy
            }

    return completed


def is_debug_log(log_file):
    """Check if a log file is from a debug run."""
    try:
        with open(log_file, 'r') as f:
            # Read first 50 lines to find debug mode indicator
            for i, line in enumerate(f):
                if i >= 50:
                    break
                if 'Debug mode: True' in line:
                    return True
                if 'Debug mode: False' in line:
                    return False
    except Exception:
        pass
    return False


def has_completion_marker(log_file):
    """Check if a log file contains completion markers ('EVALUATION COMPLETE', 'completed successfully', or 'finished')."""
    try:
        with open(log_file, 'r') as f:
            # Read file looking for completion markers
            content = f.read()
            # Look for completion indicators
            if 'EVALUATION COMPLETE' in content or 'completed successfully' in content.lower() or 'finished' in content.lower():
                return True
    except Exception:
        pass
    return False


def scan_in_progress_jobs(logs_dir, completed_jobs):
    """Scan logs directory for in-progress jobs (evaluation_*.log files).

    Note: Scans ALL log files, regardless of completion status. The merge_results()
    function will determine if a job is IN_PROGRESS by comparing log timestamp
    with performance file timestamp (for rerun detection).
    """
    in_progress = {}

    for log_file in Path(logs_dir).rglob('evaluation_*.log'):
        parsed = parse_directory_path(log_file.parent, logs_dir)
        if not parsed:
            continue

        model, split, context, phrase, mode = parsed
        key = (model, split, context, phrase, mode)

        # Skip if this is a debug log
        if is_debug_log(log_file):
            continue

        # Get file modification time (last updated)
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)

        # Keep the most recent log file for this combination
        if key not in in_progress or mtime > in_progress[key]['timestamp']:
            in_progress[key] = {
                'status': 'IN_PROGRESS',
                'file': str(log_file),
                'timestamp': mtime,
                'time_ago': get_time_ago(mtime),
                'overall_accuracy': None
            }

    return in_progress


def get_all_combinations(output_dir, logs_dir):
    """
    Get all unique combinations from directory structure.

    This discovers jobs by scanning the actual directory structure.
    """
    combinations = set()

    for base_dir in [output_dir, logs_dir]:
        if not Path(base_dir).exists():
            continue

        for path in Path(base_dir).rglob('*'):
            if not path.is_dir():
                continue

            parsed = parse_directory_path(path, base_dir)
            if parsed:
                combinations.add(parsed)

    return sorted(combinations)


def merge_results(completed, in_progress, all_combinations):
    """Merge completed and in-progress jobs, mark remaining as pending.

    Detects reruns by checking for completion markers in log files:
    - If log file contains 'completed successfully' or 'finished': COMPLETED
    - If log file exists but no completion marker: IN_PROGRESS (still running)
    - If only performance file exists: COMPLETED
    - If only log file exists: Check for completion marker
    """
    results = {}

    for key in all_combinations:
        if key in completed and key in in_progress:
            # Both exist - check log file for completion markers
            log_file = in_progress[key]['file']

            if has_completion_marker(log_file):
                # Log has completion message - job is completed
                results[key] = completed[key]
            else:
                # Log has no completion message - job is still running (rerun)
                results[key] = in_progress[key]
        elif key in completed:
            results[key] = completed[key]
        elif key in in_progress:
            # Only log exists - check if it has completion marker
            log_file = in_progress[key]['file']

            if has_completion_marker(log_file):
                # Log has completion message but no performance file yet
                # Mark as IN_PROGRESS since results aren't generated yet
                results[key] = in_progress[key]
            else:
                # Log has no completion marker - job is still running
                results[key] = in_progress[key]
        else:
            results[key] = {
                'status': 'PENDING',
                'file': None,
                'timestamp': None,
                'time_ago': None,
                'overall_accuracy': None
            }

    return results


def display_table(results, output_file):
    """Display results in matrix table format and save to file."""
    # Group by model/phrase_mode (rows) and context (columns)
    # Structure: results[(model, split, context, phrase, mode)] = info

    # Count statistics
    total = len(results)
    completed = sum(1 for r in results.values() if r['status'] == 'COMPLETED')
    in_progress = sum(1 for r in results.values() if r['status'] == 'IN_PROGRESS')
    pending = sum(1 for r in results.values() if r['status'] == 'PENDING')

    # Extract unique values
    splits = sorted(set(k[1] for k in results.keys()))

    # For each split, create a matrix table
    output_lines = []

    # Add timestamp
    now = datetime.now()
    output_lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")
    output_lines.append("="*200)
    output_lines.append("REDDIT MODERATION EVALUATION - JOB TRACKER")
    output_lines.append("="*200)
    output_lines.append(f"Total jobs: {total}  |  ✅ Completed: {completed}  |  🔄 In Progress: {in_progress}  |  ⏳ Pending: {pending}")
    output_lines.append("="*200)

    for split in splits:
        # Filter results for this split
        split_results = {k: v for k, v in results.items() if k[1] == split}

        # Get unique models and contexts
        models_set = set(k[0] for k in split_results.keys())
        contexts_set = set(k[2] for k in split_results.keys())

        # Sort models by size (extract number from model name)
        def extract_model_size(model_name):
            """Extract size number from model name for sorting."""
            import re
            match = re.search(r'(\d+)b', model_name)
            return int(match.group(1)) if match else 999

        models = sorted(models_set, key=extract_model_size)

        # Sort contexts by number of components (most to least)
        contexts = sorted(contexts_set, key=lambda x: len(x.split('-')), reverse=True)

        # Build row labels (model + phrase_mode)
        row_labels = []
        for model in models:
            # Get unique phrase/mode combinations for this model
            phrase_modes = sorted(set((k[3], k[4]) for k in split_results.keys() if k[0] == model))
            for phrase, mode in phrase_modes:
                if phrase == 'baseline':
                    label = f"{model} | baseline"
                else:
                    label = f"{model} | {phrase}_{mode}"
                row_labels.append((model, phrase, mode, label))

        # Prepare table header
        output_lines.append(f"\n{'='*200}")
        output_lines.append(f"SPLIT: {split}")
        output_lines.append(f"{'='*200}\n")

        # Create context abbreviations and legend
        context_abbrev = {}
        context_legend = []

        # Abbreviation mapping for common words
        word_abbrev = {
            'subreddit': 'SR',
            'submission': 'SUB',
            'media': 'M',
            'discussion': 'D',
            'user': 'U'
        }

        for i, ctx in enumerate(contexts, 1):
            parts = ctx.split('-')
            # Create readable abbreviation using predefined mappings
            abbrev_parts = [word_abbrev.get(p, p[:3].upper()) for p in parts]
            abbrev = '-'.join(abbrev_parts)

            # If still too long, use numbers
            if len(abbrev) > 18:
                abbrev = f"CTX{i}"

            context_abbrev[ctx] = abbrev
            context_legend.append(f"  {abbrev:<18} = {ctx}")

        # Create column header
        col_width = 20  # Width for each cell
        header_line1 = f"{'Model | Mode':<30}"

        for ctx in contexts:
            abbrev = context_abbrev[ctx]
            header_line1 += f"{abbrev:^{col_width}}"

        output_lines.append(header_line1)
        output_lines.append("-" * (30 + col_width * len(contexts)))

        # Build table rows
        for model, phrase, mode, label in row_labels:
            row = f"{label:<30}"

            for ctx in contexts:
                key = (model, split, ctx, phrase, mode)

                if key in split_results:
                    info = split_results[key]
                    status = info['status']

                    # Format cell content
                    if status == 'COMPLETED':
                        status_icon = '✅'
                        acc = f"{info['overall_accuracy']:.3f}" if info['overall_accuracy'] is not None else '-'
                        time = info['time_ago'] if info['time_ago'] else '-'
                        # Shorten time format
                        time = time.replace(' ago', '').replace('just now', 'now')
                        cell = f"{status_icon} {acc} {time}"
                    elif status == 'IN_PROGRESS':
                        status_icon = '🔄'
                        time = info['time_ago'] if info['time_ago'] else '-'
                        time = time.replace(' ago', '').replace('just now', 'now')
                        cell = f"{status_icon} {time}"
                    else:  # PENDING
                        cell = "⏳ PENDING"

                    row += f"{cell:^{col_width}}"
                else:
                    row += f"{'-':^{col_width}}"

            output_lines.append(row)

        output_lines.append("")
        output_lines.append("Context Legend:")
        for legend_line in context_legend:
            output_lines.append(legend_line)
        output_lines.append("")

    output_lines.append("="*200)
    output_lines.append("\nStatus Legend:")
    output_lines.append("  ✅ COMPLETED - Shows: ✅ accuracy last_updated")
    output_lines.append("  🔄 IN_PROGRESS - Shows: 🔄 last_updated")
    output_lines.append("  ⏳ PENDING - No logs or outputs found")
    output_lines.append("="*200)

    # Print to console
    for line in output_lines:
        print(line)

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"\n📄 Output saved to: {output_file}")


def output_json(results, output_file):
    """Output results as JSON."""
    json_data = []

    for (model, split, context, phrase, mode), info in results.items():
        json_data.append({
            'model': model,
            'split': split,
            'context': context,
            'phrase': phrase,
            'mode': mode,
            'status': info['status'],
            'file': info['file'],
            'timestamp': info['timestamp'].isoformat() if info['timestamp'] else None,
            'time_ago': info['time_ago'],
            'overall_accuracy': info.get('overall_accuracy')
        })

    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"📄 JSON output saved to: {output_file}")


def output_csv(results, output_file):
    """Output results as CSV."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Split', 'Context', 'Phrase', 'Mode', 'Status', 'Overall Accuracy', 'File', 'Timestamp', 'Time Ago'])

        for (model, split, context, phrase, mode), info in sorted(results.items()):
            writer.writerow([
                model,
                split,
                context,
                phrase,
                mode,
                info['status'],
                info.get('overall_accuracy', ''),
                info['file'] or '',
                info['timestamp'].isoformat() if info['timestamp'] else '',
                info['time_ago'] or ''
            ])

    print(f"📄 CSV output saved to: {output_file}")


def output_latex(results, output_file):
    """Output publication-ready LaTeX table."""
    # Filter for test split and completed jobs only
    test_results = {
        k: v for k, v in results.items()
        if k[1] == 'test' and v['status'] == 'COMPLETED' and v['overall_accuracy'] is not None
    }

    if not test_results:
        print("❌ No completed test results found for LaTeX table")
        return

    # Extract unique contexts and models
    contexts = sorted(set(k[2] for k in test_results.keys()))
    models_set = set(k[0] for k in test_results.keys())

    # Sort models by size (extract number from model name)
    def extract_model_size(model_name):
        """Extract size number from model name for sorting."""
        import re
        match = re.search(r'(\d+)b', model_name)
        return int(match.group(1)) if match else 999

    models = sorted(models_set, key=extract_model_size)

    # Get all phrase/mode combinations
    phrase_modes = sorted(set((k[3], k[4]) for k in test_results.keys()))

    # Find best score per model
    best_per_model = {}
    for model in models:
        model_results = {
            k: v for k, v in test_results.items()
            if k[0] == model
        }
        if model_results:
            best_key = max(model_results.items(), key=lambda x: x[1]['overall_accuracy'])[0]
            best_per_model[model] = best_key

    # Start building LaTeX table
    latex_lines = []
    latex_lines.append("% Publication-ready results table")
    latex_lines.append("% Requires \\usepackage{booktabs}")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Across Context Configurations}")
    latex_lines.append("\\label{tab:results}")

    # Create column specification
    num_cols = len(contexts)
    col_spec = "l" + "c" * num_cols
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\toprule")

    # Create header row with context abbreviations
    word_abbrev = {
        'subreddit': 'SR',
        'submission': 'Sub',
        'media': 'Med',
        'discussion': 'Disc',
        'user': 'User'
    }

    context_abbrevs = []
    for ctx in contexts:
        parts = ctx.split('-')
        abbrev_parts = [word_abbrev.get(p, p.capitalize()[:4]) for p in parts]
        abbrev = '-'.join(abbrev_parts)
        context_abbrevs.append(abbrev)

    header = "Model & " + " & ".join(context_abbrevs) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Create rows for each model and phrase/mode combination
    for i, model in enumerate(models):
        # Extract parameter size for model label
        import re
        match = re.search(r'(\d+)b', model)
        model_size = match.group(1) + "B" if match else model

        # Get phrase/mode combinations for this model
        model_phrase_modes = sorted(set(
            (k[3], k[4]) for k in test_results.keys() if k[0] == model
        ))

        for j, (phrase, mode) in enumerate(model_phrase_modes):
            # Create row label
            if phrase == 'baseline':
                phrase_label = "Baseline"
            else:
                phrase_label = f"{phrase.upper()}-{mode.capitalize()}"

            if j == 0:
                # First row for this model - include model size with rowspan effect
                row_label = f"{model_size} & {phrase_label}"
            else:
                # Subsequent rows - indent phrase label
                row_label = f"& {phrase_label}"

            row_values = []
            for ctx in contexts:
                key = (model, 'test', ctx, phrase, mode)

                if key in test_results:
                    acc = test_results[key]['overall_accuracy'] * 100
                    acc_str = f"{acc:.1f}"

                    # Check if this is the best for this model
                    if key == best_per_model.get(model):
                        acc_str = f"\\textbf{{{acc_str}}}"

                    row_values.append(acc_str)
                else:
                    row_values.append("--")

            row = row_label + " & " + " & ".join(row_values) + " \\\\"
            latex_lines.append(row)

        # Add midrule between models (but not after the last one)
        if i < len(models) - 1:
            latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"📄 LaTeX table saved to: {output_file}")
    print(f"   Models included: {', '.join(models)}")
    print(f"   Contexts: {len(contexts)}")
    print(f"   Phrase/mode combinations: {len(phrase_modes)}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)

    # Scan directories
    print("🔍 Scanning directories...")
    completed_jobs = scan_completed_jobs(output_dir)
    print(f"   Found {len(completed_jobs)} completed jobs")

    in_progress_jobs = scan_in_progress_jobs(logs_dir, completed_jobs)
    print(f"   Found {len(in_progress_jobs)} in-progress jobs")

    all_combinations = get_all_combinations(output_dir, logs_dir)
    print(f"   Found {len(all_combinations)} total job combinations")
    print()

    # Merge results
    results = merge_results(completed_jobs, in_progress_jobs, all_combinations)

    # Output results
    if args.json:
        output_file = args.output.replace('.txt', '.json')
        output_json(results, output_file)
    elif args.csv:
        output_file = args.output.replace('.txt', '.csv')
        output_csv(results, output_file)
    elif args.latex:
        output_file = args.output.replace('.txt', '.tex')
        output_latex(results, output_file)
    else:
        display_table(results, args.output)


if __name__ == '__main__':
    main()
