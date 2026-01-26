#!/usr/bin/env python3
"""
Add Per-Language Metrics to Performance JSONs

Reads existing reasoning JSONs, joins with language data from test set,
and adds 'per_language' metrics to performance JSONs without re-running evaluations.

Usage:
    python add_language_metrics.py --model gpt5.2-high --split test --context submission-media-discussion-user
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS


def normalize_language(lang_code: str) -> str:
    """Normalize language code by taking root (e.g., en-au → en, pt_BR → pt)."""
    return lang_code.replace('_', '-').split('-')[0]


def load_language_mapping(split: str) -> Dict[str, str]:
    """
    Load subreddit → normalized_language mapping from hydrated dataset.

    Args:
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        Dictionary mapping subreddit name to normalized language code
    """
    # Try clustered version first, then regular hydrated
    data_dir = Path(PATHS['data'])
    possible_files = [
        data_dir / f'{split}_hydrated_clustered.json',
        data_dir / f'{split}_hydrated.json'
    ]

    dataset_file = None
    for f in possible_files:
        if f.exists():
            dataset_file = f
            break

    if not dataset_file:
        raise FileNotFoundError(f"Could not find hydrated dataset for split '{split}'")

    print(f"Loading language mapping from: {dataset_file}")
    with open(dataset_file) as f:
        dataset = json.load(f)

    # Build mapping: subreddit → normalized_language
    mapping = {}
    for sub_data in dataset['subreddits']:
        subreddit = sub_data['subreddit']
        language = sub_data.get('language', 'unknown')
        normalized_lang = normalize_language(language)
        mapping[subreddit] = normalized_lang

    print(f"  Loaded {len(mapping)} subreddits with language info")
    unique_langs = set(mapping.values())
    print(f"  Unique normalized languages: {sorted(unique_langs)}")

    return mapping


def calculate_language_metrics(results: List[Dict[str, Any]],
                               language_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate per-language accuracy statistics.

    Args:
        results: List of evaluation results (from reasoning JSON)
        language_mapping: Dict mapping subreddit → normalized_language

    Returns:
        Dictionary mapping language codes to accuracy stats
    """
    language_stats = defaultdict(lambda: {
        'violating_correct': 0,
        'compliant_correct': 0,
        'total_correct': 0,
        'count': 0
    })

    skipped = 0
    for result in results:
        subreddit = result['subreddit']

        # Get language for this subreddit
        language = language_mapping.get(subreddit)
        if not language:
            skipped += 1
            continue

        # Accumulate stats
        language_stats[language]['violating_correct'] += result['violating']['score']
        language_stats[language]['compliant_correct'] += result['compliant']['score']
        language_stats[language]['total_correct'] += result['violating']['score'] + result['compliant']['score']
        language_stats[language]['count'] += 1

    if skipped > 0:
        print(f"  Warning: Skipped {skipped} results with unknown language")

    # Calculate accuracies
    final_stats = {}
    for language, stats in language_stats.items():
        count = stats['count']
        total_threads = count * 2

        final_stats[language] = {
            'overall_accuracy': stats['total_correct'] / total_threads if total_threads > 0 else 0,
            'violating_accuracy': stats['violating_correct'] / count if count > 0 else 0,
            'compliant_accuracy': stats['compliant_correct'] / count if count > 0 else 0,
            'count': count,
            'total_threads': total_threads
        }

    return final_stats


def update_performance_json(perf_file: Path, language_metrics: Dict[str, Dict[str, Any]]):
    """
    Load performance JSON, add per_language section, and save.

    Args:
        perf_file: Path to performance JSON file
        language_metrics: Language metrics to add
    """
    print(f"Updating performance JSON: {perf_file}")

    with open(perf_file) as f:
        perf_data = json.load(f)

    # Add per_language metrics
    perf_data['metrics']['per_language'] = language_metrics

    # Save updated JSON
    with open(perf_file, 'w') as f:
        json.dump(perf_data, f, indent=2)

    print(f"  Added metrics for {len(language_metrics)} languages")
    for lang, stats in sorted(language_metrics.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"    {lang}: {stats['count']} pairs, {stats['overall_accuracy']*100:.1f}% accuracy")


def find_latest_reasoning_file(model: str, split: str, context: str, phrase: str = 'baseline', mode: str = 'prefill') -> Path:
    """Find the latest reasoning_*.json file for given parameters."""
    base_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    target_dir = base_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    reasoning_files = sorted(target_dir.glob("reasoning_*.json"))
    if not reasoning_files:
        raise FileNotFoundError(f"No reasoning files found in {target_dir}")

    return reasoning_files[-1]  # Latest file


def find_latest_performance_file(model: str, split: str, context: str, phrase: str = 'baseline', mode: str = 'prefill') -> Path:
    """Find the latest performance_*.json file for given parameters."""
    base_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    target_dir = base_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {target_dir}")

    perf_files = sorted(target_dir.glob("performance_*.json"))
    if not perf_files:
        raise FileNotFoundError(f"No performance files found in {target_dir}")

    return perf_files[-1]  # Latest file


def main():
    parser = argparse.ArgumentParser(description='Add per-language metrics to performance JSONs')
    parser.add_argument('--model', default='gpt5.2-high', help='Model name')
    parser.add_argument('--split', default='test', help='Dataset split')
    parser.add_argument('--context', default='submission-media-discussion-user', help='Context')
    parser.add_argument('--phrase', default='baseline', help='Phrase type')
    parser.add_argument('--mode', default='prefill', help='Mode')
    args = parser.parse_args()

    print("="*80)
    print("ADD PER-LANGUAGE METRICS TO PERFORMANCE JSON")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Context: {args.context}")
    print()

    # Load language mapping
    print("Step 1: Loading language mapping...")
    language_mapping = load_language_mapping(args.split)
    print()

    # Load reasoning results
    print("Step 2: Loading reasoning results...")
    reasoning_file = find_latest_reasoning_file(args.model, args.split, args.context, args.phrase, args.mode)
    print(f"  Reading: {reasoning_file}")
    with open(reasoning_file) as f:
        results = json.load(f)
    print(f"  Loaded {len(results)} evaluation results")
    print()

    # Calculate language metrics
    print("Step 3: Calculating per-language metrics...")
    language_metrics = calculate_language_metrics(results, language_mapping)
    print()

    # Update performance JSON
    print("Step 4: Updating performance JSON...")
    perf_file = find_latest_performance_file(args.model, args.split, args.context, args.phrase, args.mode)
    update_performance_json(perf_file, language_metrics)
    print()

    print("="*80)
    print("✅ COMPLETE")
    print("="*80)
    print(f"Performance JSON updated: {perf_file}")

    return 0


if __name__ == '__main__':
    exit(main())
