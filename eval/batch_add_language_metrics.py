#!/usr/bin/env python3
"""
Batch add language metrics to all performance JSONs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.add_language_metrics import (
    load_language_mapping,
    find_latest_reasoning_file,
    find_latest_performance_file,
    calculate_language_metrics,
    update_performance_json
)
from eval.add_bootstrap_ci import find_all_result_dirs, parse_result_dir

def main():
    eval_dir = Path(__file__).parent.parent / 'output' / 'eval'

    # Find all result directories
    result_dirs = find_all_result_dirs(eval_dir)
    print(f"Found {len(result_dirs)} result directories\n")

    # Load language mapping once (all use test split)
    print("Loading language mapping...")
    language_mapping = load_language_mapping('test')
    print()

    processed = 0
    skipped = 0
    errors = 0

    for result_dir in result_dirs:
        config = parse_result_dir(result_dir)
        config_str = f"{config['model']}/{config['context']}/{config['phrase']}"

        try:
            reasoning_file = find_latest_reasoning_file(
                config['model'], config['split'], config['context'],
                config['phrase'], 'prefill'
            )
            perf_file = find_latest_performance_file(
                config['model'], config['split'], config['context'],
                config['phrase'], 'prefill'
            )

            # Check if already has per_language
            import json
            with open(perf_file) as f:
                perf_data = json.load(f)

            if 'per_language' in perf_data.get('metrics', {}):
                print(f"  ⏭️  {config_str} - already has per_language")
                skipped += 1
                continue

            # Load reasoning results
            with open(reasoning_file) as f:
                results = json.load(f)

            # Calculate language metrics
            language_metrics = calculate_language_metrics(results, language_mapping)

            # Update performance JSON
            update_performance_json(perf_file, language_metrics)

            print(f"  ✅ {config_str} - added {len(language_metrics)} languages")
            processed += 1

        except Exception as e:
            print(f"  ❌ {config_str} - {e}")
            errors += 1

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Total:     {len(result_dirs)}")

if __name__ == '__main__':
    main()
