#!/usr/bin/env python3
"""
Shared utilities for evaluation scripts.

Provides common functions for loading performance data, extracting cluster
metrics with confidence intervals, and resolving performance file paths.
"""

import json
from pathlib import Path
from typing import Optional


def get_eval_dir() -> Path:
    """Get the base evaluation output directory."""
    return Path(__file__).resolve().parent.parent / 'output' / 'eval'


def get_perf_dir(model: str, split: str, context: str,
                 phrase: str = 'baseline', mode: str = 'prefill') -> Path:
    """Build the performance directory path for a given configuration.

    Args:
        model: Full model name (e.g. 'gpt5.2-high')
        split: Dataset split (train/val/test)
        context: Context type (e.g. 'submission-media-discussion-user')
        phrase: Phrase type ('baseline' or other)
        mode: Mode type (e.g. 'prefill')

    Returns:
        Path to the performance directory
    """
    subdir = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    return get_eval_dir() / model / split / context / subdir


def get_latest_performance_file(perf_dir: Path) -> Path:
    """Get latest performance file, preferring _ci version if available.

    Args:
        perf_dir: Directory containing performance JSON files

    Returns:
        Path to the latest performance file

    Raises:
        FileNotFoundError: If no performance files exist in the directory
    """
    perf_files_ci = sorted(perf_dir.glob('performance_*_ci.json'))
    perf_files = sorted(perf_dir.glob('performance_*.json'))

    if perf_files_ci:
        return perf_files_ci[-1]
    elif perf_files:
        return perf_files[-1]
    else:
        raise FileNotFoundError(f"No performance files in {perf_dir}")


def load_performance(model: str, split: str, context: str,
                     phrase: str = 'baseline', mode: str = 'prefill') -> dict:
    """Load performance data for a given configuration.

    Args:
        model: Full model name
        split: Dataset split
        context: Context type
        phrase: Phrase type
        mode: Mode type

    Returns:
        Parsed performance JSON data

    Raises:
        FileNotFoundError: If no performance files exist
    """
    perf_dir = get_perf_dir(model, split, context, phrase, mode)
    perf_file = get_latest_performance_file(perf_dir)
    with open(perf_file) as f:
        return json.load(f)


def extract_cluster_metrics(perf_data: dict, cluster_type: str,
                            metric: str = 'overall_accuracy') -> list:
    """Extract per-cluster accuracy and confidence intervals from performance data.

    Args:
        perf_data: Loaded performance JSON
        cluster_type: 'subreddit' or 'rule'
        metric: Metric key (e.g. 'overall_accuracy', 'violating_accuracy')

    Returns:
        List of (name, accuracy%, ci_low%, ci_high%) tuples sorted by accuracy descending.
        'Other' cluster is renamed to lowercase 'other'.
    """
    per_cluster = perf_data['metrics'].get(f'per_{cluster_type}_cluster', {})
    results = []

    for name, info in per_cluster.items():
        if metric not in info:
            continue
        acc = info[metric] * 100
        ci_key = f'{metric}_ci'
        if ci_key in info:
            ci_low, ci_high = info[ci_key]
            ci_low *= 100
            ci_high *= 100
        else:
            ci_low, ci_high = acc, acc
        display_name = 'other' if name.lower() == 'other' else name
        results.append((display_name, acc, ci_low, ci_high))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def extract_three_metrics(perf_data: dict, cluster_type: str) -> list:
    """Extract violating, overall, and compliant accuracy per cluster.

    Args:
        perf_data: Loaded performance JSON
        cluster_type: 'subreddit' or 'rule'

    Returns:
        List of (name, violating%, overall%, compliant%) tuples sorted by overall descending.
    """
    per_cluster = perf_data['metrics'].get(f'per_{cluster_type}_cluster', {})
    results = []

    for name, info in per_cluster.items():
        if 'overall_accuracy' not in info:
            continue
        violating = info.get('violating_accuracy', 0) * 100
        overall = info['overall_accuracy'] * 100
        compliant = info.get('compliant_accuracy', 0) * 100
        display_name = 'other' if name.lower() == 'other' else name
        results.append((display_name, violating, overall, compliant))

    results.sort(key=lambda x: x[2], reverse=True)
    return results


def load_cluster_size_stats(split: str = 'all', cluster_type: str = 'rule') -> dict:
    """Load cached cluster size statistics.

    Args:
        split: Dataset split (train/val/test/all)
        cluster_type: 'rule' or 'subreddit'

    Returns:
        Dictionary with cluster stats

    Raises:
        FileNotFoundError: If cache file doesn't exist
    """
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    cache_file = data_dir / f'{cluster_type}_cluster_size_stats_{split}.json'

    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cluster size stats cache not found: {cache_file}\n"
            f"Please run: python eval/compute_cluster_size_stats.py --cluster-type {cluster_type}"
            + (f" --split {split}" if split != 'all' else "")
        )

    with open(cache_file) as f:
        return json.load(f)


def find_performance_file_by_parts(model_base: str, variant: str,
                                    split: str, context: str) -> Optional[Path]:
    """Find the latest performance file given model base + variant separately.

    Used by LaTeX table generation where model names are split into base/variant.

    Args:
        model_base: Model base name (e.g. 'qwen3-vl-4b')
        variant: Variant suffix (e.g. 'instruct', 'thinking')
        split: Dataset split
        context: Context type

    Returns:
        Path to latest performance file, or None if not found
    """
    model_name = f"{model_base}-{variant}"
    target_dir = get_eval_dir() / model_name / split / context / "baseline"

    if not target_dir.exists():
        return None

    perf_files = sorted(target_dir.glob("performance_*.json"))
    return perf_files[-1] if perf_files else None
