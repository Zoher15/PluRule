#!/usr/bin/env python3
"""
Compute and cache rule cluster size statistics from test dataset.

This script extracts metadata about rule clusters:
- Number of unique subreddits with rules in each cluster
- Number of unique rules in each cluster
- Number of thread pairs per cluster

The output cache file enables fast plotting without re-processing the full dataset.

Usage:
    python eval/compute_cluster_size_stats.py
    python eval/compute_cluster_size_stats.py --split train
"""

import sys
import json
import argparse
import zstandard as zstd
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS


def load_dataset(dataset_path: Path) -> dict:
    """Load and decompress zstd-compressed JSON dataset.

    Args:
        dataset_path: Path to .json.zst file

    Returns:
        Loaded dataset dictionary
    """
    print(f"Loading dataset: {dataset_path.name}")
    with open(dataset_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            dataset = json.loads(reader.read())
    print(f"  ✓ Loaded {len(dataset['subreddits'])} subreddits")
    return dataset


def extract_cluster_sizes(datasets: list, cluster_type: str = 'rule') -> dict:
    """Extract cluster size statistics from one or more datasets.

    Args:
        datasets: List of (split_name, dataset_dict) tuples
        cluster_type: 'rule' or 'subreddit'

    Returns:
        Tuple of (cluster_stats dict, total_pairs)
    """
    cluster_subreddits = defaultdict(set)
    cluster_rules = defaultdict(set)
    cluster_pairs = Counter()
    total_pairs = 0

    for split_name, dataset in datasets:
        print(f"Extracting {cluster_type} cluster sizes from {split_name}...")
        for subreddit_data in dataset['subreddits']:
            subreddit_name = subreddit_data['subreddit']

            # Get cluster label from appropriate location
            if cluster_type == 'subreddit':
                # Subreddit cluster is at subreddit level
                subreddit_cluster_label = subreddit_data['subreddit_cluster_label']

            for pair in subreddit_data['thread_pairs']:
                total_pairs += 1

                # Determine cluster label based on type
                if cluster_type == 'rule':
                    cluster_label = pair['metadata']['rule_cluster_label']
                else:  # subreddit
                    cluster_label = subreddit_cluster_label

                cluster_subreddits[cluster_label].add(subreddit_name)
                rule_text = pair['metadata']['rule']
                cluster_rules[cluster_label].add((subreddit_name, rule_text))
                cluster_pairs[cluster_label] += 1

    print(f"  ✓ Processed {total_pairs} thread pairs")
    print(f"  ✓ Found {len(cluster_subreddits)} unique {cluster_type} clusters")

    cluster_stats = {}
    for cluster_label in sorted(cluster_subreddits.keys()):
        cluster_stats[cluster_label] = {
            'n_subreddits': len(cluster_subreddits[cluster_label]),
            'n_rules': len(cluster_rules[cluster_label]),
            'n_thread_pairs': cluster_pairs[cluster_label]
        }

    return cluster_stats, total_pairs


def save_cluster_size_cache(cluster_stats: dict, total_pairs: int,
                            cache_path: Path, source_dataset: str):
    """Save cluster size statistics to cache file.

    Args:
        cluster_stats: Dictionary of cluster statistics
        total_pairs: Total number of thread pairs processed
        cache_path: Path to output cache file
        source_dataset: Name of source dataset file
    """
    cache_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source_dataset': source_dataset,
            'total_thread_pairs': total_pairs,
            'total_clusters': len(cluster_stats)
        },
        'cluster_stats': cluster_stats
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print(f"  ✓ Saved cache to: {cache_path.name}")


def validate_cluster_coverage(cluster_stats: dict, stage10_stats_path: Path):
    """Validate that extracted clusters match stage10 statistics.

    Args:
        cluster_stats: Extracted cluster statistics
        stage10_stats_path: Path to stage10_cluster_assignment_stats.json
    """
    if not stage10_stats_path.exists():
        print(f"  ⚠ Stage10 stats not found: {stage10_stats_path.name}")
        print("    Skipping validation...")
        return

    print("Validating cluster coverage...")

    with open(stage10_stats_path) as f:
        stage10_data = json.load(f)

    # Get rule clusters from test split
    test_stats = stage10_data.get('cluster_assignment_statistics', {}).get('test', {})
    expected_clusters = test_stats.get('rule_clusters', {})

    if not expected_clusters:
        print("  ⚠ No test split rule clusters found in stage10 stats")
        return

    # Compare cluster names
    extracted_clusters = set(cluster_stats.keys())
    expected_cluster_names = set(expected_clusters.keys())

    missing = expected_cluster_names - extracted_clusters
    extra = extracted_clusters - expected_cluster_names

    if missing:
        print(f"  ⚠ Missing clusters: {missing}")
    if extra:
        print(f"  ⚠ Extra clusters: {extra}")

    # Compare thread pair counts
    mismatches = []
    for cluster, expected_count in expected_clusters.items():
        if cluster in cluster_stats:
            actual_count = cluster_stats[cluster]['n_thread_pairs']
            if actual_count != expected_count:
                mismatches.append(f"{cluster}: expected {expected_count}, got {actual_count}")

    if mismatches:
        print("  ⚠ Thread pair count mismatches:")
        for mismatch in mismatches:
            print(f"    - {mismatch}")

    if not missing and not extra and not mismatches:
        print("  ✓ Validation passed: All clusters match stage10 stats")


def main():
    parser = argparse.ArgumentParser(
        description='Compute and cache cluster size statistics'
    )
    parser.add_argument(
        '--split',
        default='all',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to process (default: all)'
    )
    parser.add_argument(
        '--cluster-type',
        default='rule',
        choices=['rule', 'subreddit'],
        help='Cluster type to extract (default: rule)'
    )
    args = parser.parse_args()

    # Paths
    data_dir = Path(PATHS['data'])
    stage10_stats_path = data_dir / 'stage10_cluster_assignment_stats.json'

    # Load datasets
    if args.split == 'all':
        splits = ['train', 'val', 'test']
        cache_path = data_dir / f'{args.cluster_type}_cluster_size_stats_all.json'
        source_dataset = 'train+val+test_hydrated_clustered.json.zst'
    else:
        splits = [args.split]
        cache_path = data_dir / f'{args.cluster_type}_cluster_size_stats_{args.split}.json'
        source_dataset = f'{args.split}_hydrated_clustered.json.zst'

    datasets = []
    for split in splits:
        dataset_path = data_dir / f'{split}_hydrated_clustered.json.zst'
        if not dataset_path.exists():
            print(f"⚠ Dataset not found: {dataset_path}, skipping")
            continue
        datasets.append((split, load_dataset(dataset_path)))

    if not datasets:
        print(f"❌ No datasets found")
        return 1

    # Extract cluster sizes
    cluster_stats, total_pairs = extract_cluster_sizes(datasets, cluster_type=args.cluster_type)

    # Save cache
    save_cluster_size_cache(
        cluster_stats,
        total_pairs,
        cache_path,
        source_dataset
    )

    # Validate (only for single splits, not 'all')
    if args.split != 'all':
        validate_cluster_coverage(cluster_stats, stage10_stats_path)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total clusters: {len(cluster_stats)}")
    print(f"Total thread pairs: {total_pairs}")
    print(f"\nCluster statistics:")
    print(f"  Min subreddits: {min(s['n_subreddits'] for s in cluster_stats.values())}")
    print(f"  Max subreddits: {max(s['n_subreddits'] for s in cluster_stats.values())}")
    print(f"  Min rules: {min(s['n_rules'] for s in cluster_stats.values())}")
    print(f"  Max rules: {max(s['n_rules'] for s in cluster_stats.values())}")
    print(f"  Min pairs: {min(s['n_thread_pairs'] for s in cluster_stats.values())}")
    print(f"  Max pairs: {max(s['n_thread_pairs'] for s in cluster_stats.values())}")
    print("\n✅ Done!")

    return 0


if __name__ == '__main__':
    exit(main())
