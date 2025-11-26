#!/usr/bin/env python3
"""
Stage 10: Assign Cluster Labels to Datasets

Reads cluster labels from analysis/label_clusters.py output and assigns them
to thread pairs and subreddits in the train/val/test datasets.

Input:
- output/embeddings/all_rule_metadata.tsv (with cluster_id and cluster_label columns)
- output/embeddings/all_subreddit_metadata.tsv (with cluster_id and cluster_label columns)
- data/{split}_hydrated.json.zst (train/val/test datasets from Stage 9)

Output:
- data/{split}_hydrated_clustered.json.zst (updated datasets with cluster labels)
- data/{split}_dehydrated_clustered.json.zst (dehydrated versions)
- data/test_hydrated_clustered.json (uncompressed test set)
- data/stage10_cluster_assignment_stats.json
- data/stage10_rule_cluster_distribution.png (visualization of rule cluster distribution)
- data/stage10_rule_cluster_distribution.pdf (publication-quality version)
- data/stage10_subreddit_cluster_distribution.png (visualization of subreddit cluster distribution)
- data/stage10_subreddit_cluster_distribution.pdf (publication-quality version)

The script adds cluster labels to:
1. Thread pairs (rule clusters):
   - rule_cluster_id: The cluster ID for the matched rule (-1 for Other)
   - rule_cluster_label: The semantic label for the cluster (e.g., "spoiler tags", "civility rules")
   - rule_cluster_probability: Cluster assignment probability

2. Subreddits (subreddit clusters):
   - subreddit_cluster_id: The cluster ID for the subreddit (-1 for Other)
   - subreddit_cluster_label: The semantic label for the cluster
   - subreddit_cluster_probability: Cluster assignment probability
"""

import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import write_json_file, read_compressed_json, write_compressed_json

# ============================================================================
# Data Loading
# ============================================================================

def load_cluster_metadata(filename: str, required_cols: list, logger) -> pd.DataFrame:
    """Load and validate cluster metadata from a TSV file.

    Args:
        filename: Name of the metadata file (e.g., 'all_rule_metadata.tsv')
        required_cols: List of required column names
        logger: Logger instance

    Returns:
        DataFrame with cluster metadata, or empty DataFrame on error
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_file = os.path.join(base_dir, 'output', 'embeddings', filename)

    if not os.path.exists(metadata_file):
        logger.error(f"❌ Metadata not found: {metadata_file}")
        logger.error("Please run analysis/label_clusters.py first")
        return pd.DataFrame()

    logger.info(f"📋 Loading cluster mappings from {metadata_file}...")
    df = pd.read_csv(metadata_file, sep='\t')

    # Check for required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns in metadata: {missing}")
        return pd.DataFrame()

    return df


def load_rule_cluster_mapping(logger) -> Dict[Tuple[str, str], Dict]:
    """Load cluster labels from all_rule_metadata.tsv.

    Returns:
        Dict mapping (subreddit, short_name) -> {cluster_id, cluster_label, cluster_probability}
    """
    required_cols = ['subreddit', 'short_name', 'cluster_id', 'cluster_label']
    df = load_cluster_metadata('all_rule_metadata.tsv', required_cols, logger)

    if df.empty:
        return {}

    # Build mapping using short_name (not short_name_clean) to match dataset rules
    # Note: Stage 9 uses short_name_clean as the 'rule' field in metadata
    # Note: The subreddit field may contain comma-separated subreddits for shared rules
    mapping = {}
    for _, row in df.iterrows():
        subreddits_str = str(row['subreddit'])
        short_name = str(row['short_name'])

        # Split comma-separated subreddits and create entry for each
        subreddits = [s.strip().lower() for s in subreddits_str.split(',')]

        cluster_id = int(row['cluster_id'])
        cluster_label = str(row['cluster_label'])

        # Replace "Noise" with "Other" for cluster_id -1
        if cluster_id == -1:
            cluster_label = 'Other'

        cluster_info = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_probability': float(row.get('cluster_probability', 0.0))
        }

        for subreddit in subreddits:
            key = (subreddit, short_name)
            mapping[key] = cluster_info

    logger.info(f"  ✅ Loaded cluster mappings for {len(mapping)} rules")
    return mapping


def load_subreddit_cluster_mapping(logger) -> Dict[str, Dict]:
    """Load cluster labels from all_subreddit_metadata.tsv.

    Returns:
        Dict mapping subreddit -> {cluster_id, cluster_label, cluster_probability}
    """
    required_cols = ['subreddit', 'cluster_id', 'cluster_label']
    df = load_cluster_metadata('all_subreddit_metadata.tsv', required_cols, logger)

    if df.empty:
        return {}

    # Build mapping
    mapping = {}
    for _, row in df.iterrows():
        subreddit = str(row['subreddit']).strip().lower()

        cluster_id = int(row['cluster_id'])
        cluster_label = str(row['cluster_label'])

        # Replace "Noise" with "Other" for cluster_id -1
        if cluster_id == -1:
            cluster_label = 'Other'

        cluster_info = {
            'cluster_id': cluster_id,
            'cluster_label': cluster_label,
            'cluster_probability': float(row.get('cluster_probability', 0.0))
        }

        mapping[subreddit] = cluster_info

    logger.info(f"  ✅ Loaded cluster mappings for {len(mapping)} subreddits")
    return mapping




# ============================================================================
# Cluster Assignment
# ============================================================================

def assign_clusters_to_dataset(dataset: Dict, rule_mapping: Dict[Tuple[str, str], Dict],
                               subreddit_mapping: Dict[str, Dict], logger) -> Tuple[Dict, Dict]:
    """Assign cluster labels to all thread pairs and subreddits in a dataset.

    Args:
        dataset: The hydrated dataset from Stage 9
        rule_mapping: Dict mapping (subreddit, short_name) -> cluster info
        subreddit_mapping: Dict mapping subreddit -> cluster info
        logger: Logger instance

    Returns:
        (updated_dataset, statistics)
    """
    stats = {
        'total_subreddits': 0,
        'total_thread_pairs': 0,
        'pairs_with_rule_clusters': 0,
        'subreddits_with_clusters': 0,
        'rule_cluster_distribution': Counter(),
        'subreddit_cluster_distribution': Counter(),
        'subreddit_cluster_pair_distribution': Counter()  # Track pairs by subreddit cluster
    }

    for sub_data in dataset['subreddits']:
        subreddit = sub_data['subreddit'].lower()
        stats['total_subreddits'] += 1

        # Assign subreddit cluster
        subreddit_cluster_info = subreddit_mapping[subreddit]
        sub_data['subreddit_cluster_id'] = subreddit_cluster_info['cluster_id']
        sub_data['subreddit_cluster_label'] = subreddit_cluster_info['cluster_label']
        sub_data['subreddit_cluster_probability'] = subreddit_cluster_info['cluster_probability']
        stats['subreddits_with_clusters'] += 1
        stats['subreddit_cluster_distribution'][subreddit_cluster_info['cluster_label']] += 1

        # Assign rule clusters to each thread pair
        for pair in sub_data['thread_pairs']:
            stats['total_thread_pairs'] += 1

            # Get the matched rule from metadata
            matched_rule = pair['metadata']['rule']

            # Look up cluster info
            key = (subreddit, matched_rule)
            cluster_info = rule_mapping[key]

            # Assign cluster info
            pair['metadata']['rule_cluster_id'] = cluster_info['cluster_id']
            pair['metadata']['rule_cluster_label'] = cluster_info['cluster_label']
            pair['metadata']['rule_cluster_probability'] = cluster_info['cluster_probability']
            stats['pairs_with_rule_clusters'] += 1
            stats['rule_cluster_distribution'][cluster_info['cluster_label']] += 1

            # Track thread pairs by subreddit cluster label
            stats['subreddit_cluster_pair_distribution'][subreddit_cluster_info['cluster_label']] += 1

    return dataset, stats


def dehydrate_dataset(hydrated: Dict) -> Dict:
    """Create dehydrated version (IDs only)."""
    dehydrated = {'metadata': hydrated['metadata'].copy(), 'subreddits': []}

    for sub_data in hydrated['subreddits']:
        dehydrated_subs = {}
        for sub_id, sub in sub_data['submissions'].items():
            dehydrated_subs[sub_id] = {
                'id': sub_id,
                'submission_object': '[NEEDS_HYDRATION]',
                'num_media': sub.get('num_media', 0),
                'media_files': ['[NEEDS_HYDRATION]'] * sub.get('num_media', 0)
            }

        dehydrated_pairs = []
        for pair in sub_data['thread_pairs']:
            dehydrated_pairs.append({
                'mod_comment_id': pair['mod_comment_id'],
                'mod_comment': '[NEEDS_HYDRATION]',
                'moderated_thread': ['[NEEDS_HYDRATION]'] * len(pair['moderated_thread']),
                'unmoderated_thread': ['[NEEDS_HYDRATION]'] * len(pair['unmoderated_thread']),
                'moderated_answer_options': pair['moderated_answer_options'],
                'moderated_correct_answer': pair['moderated_correct_answer'],
                'unmoderated_answer_options': pair['unmoderated_answer_options'],
                'unmoderated_correct_answer': pair['unmoderated_correct_answer'],
                'metadata': pair['metadata']  # Keep full metadata (includes cluster labels)
            })

        dehydrated['subreddits'].append({
            'subreddit': sub_data['subreddit'],
            'language': sub_data['language'],
            'data_version': sub_data['data_version'],
            'last_updated': sub_data['last_updated'],
            'total_thread_pairs': sub_data['total_thread_pairs'],
            'jsd_from_uniform': sub_data['jsd_from_uniform'],
            'rules': sub_data['rules'],
            'submissions': dehydrated_subs,
            'thread_pairs': dehydrated_pairs,
            'rank': sub_data.get('rank'),
            'subreddit_cluster_id': sub_data.get('subreddit_cluster_id', -1),
            'subreddit_cluster_label': sub_data.get('subreddit_cluster_label', 'Other'),
            'subreddit_cluster_probability': sub_data.get('subreddit_cluster_probability', 0.0)
        })

    dehydrated['metadata']['instructions'] = 'Use hydration script. All text fields contain [NEEDS_HYDRATION].'
    return dehydrated


# ============================================================================
# Visualization
# ============================================================================

def create_cluster_distribution_plot(all_stats: Dict, cluster_type: str, logger):
    """Create a bar plot showing the percentage distribution of thread pairs across clusters.

    Args:
        all_stats: Dictionary containing statistics for each split
        cluster_type: Either 'rule' or 'subreddit'
        logger: Logger instance
    """
    # Aggregate cluster counts across all splits
    total_cluster_counts = Counter()
    total_pairs = 0

    cluster_key = f'{cluster_type}_clusters'

    for split, stats in all_stats.items():
        # Get cluster distribution from all_clusters
        if cluster_key in stats:
            for cluster_label, count in stats[cluster_key].items():
                total_cluster_counts[cluster_label] += count
        if cluster_type == 'rule':
            total_pairs += stats.get('pairs_with_rule_clusters', 0)
        else:
            total_pairs += stats.get('total_thread_pairs', 0)

    # Exclude 'Other' cluster (cluster_id -1)
    filtered_counts = {label: count for label, count in total_cluster_counts.items()
                      if label.lower().strip() != 'other'}

    if not filtered_counts:
        logger.warning(f"  ⚠️  No {cluster_type} cluster data available for plotting")
        return

    # Calculate percentages
    cluster_labels = list(filtered_counts.keys())
    cluster_counts = list(filtered_counts.values())
    cluster_percentages = [100 * count / total_pairs for count in cluster_counts]

    # Sort by percentage (descending)
    sorted_data = sorted(zip(cluster_labels, cluster_percentages), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_percentages = zip(*sorted_data)

    # Create the plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(sorted_labels)), sorted_percentages, color='steelblue', edgecolor='black', linewidth=0.5)

    # Customize the plot
    plt.xlabel('Cluster Label', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Thread Pairs (%)', fontsize=12, fontweight='bold')
    title = f'Distribution of Thread Pairs Across {cluster_type.title()} Clusters'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on top of bars
    for i, (bar, pct) in enumerate(zip(bars, sorted_percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(PATHS['data'], f'stage10_{cluster_type}_cluster_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"  ✅ Saved {cluster_type} cluster distribution plot to: {plot_file}")

    # Also save as PDF for publication quality
    pdf_file = os.path.join(PATHS['data'], f'stage10_{cluster_type}_cluster_distribution.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    logger.info(f"  ✅ Saved PDF version to: {pdf_file}")

    plt.close()

    # Log summary
    logger.info(f"  📊 Plotted {len(sorted_labels)} {cluster_type} clusters")
    logger.info(f"  📊 Total pairs: {sum(cluster_counts):,}")
    logger.info(f"  📊 Top 5 {cluster_type} clusters:")
    for i, (label, pct) in enumerate(zip(sorted_labels[:5], sorted_percentages[:5])):
        logger.info(f"      {i+1}. {label}: {pct:.2f}%")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    logger = get_stage_logger(10, "assign_cluster_labels")
    log_stage_start(logger, 10, "Assign Cluster Labels to Datasets")
    start_time = time.time()

    try:
        create_directories()

        # Load cluster mappings
        rule_mapping = load_rule_cluster_mapping(logger)
        if not rule_mapping:
            logger.error("❌ Failed to load rule cluster mappings")
            log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
            return 1

        subreddit_mapping = load_subreddit_cluster_mapping(logger)
        if not subreddit_mapping:
            logger.error("❌ Failed to load subreddit cluster mappings")
            log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process each split
        logger.info("\n" + "="*80)
        logger.info("PROCESSING DATASETS")
        logger.info("="*80)

        splits = ['test', 'val', 'train']
        all_stats = {}
        output_files = {}

        for split in splits:
            logger.info(f"\n{'='*80}")
            logger.info(f"{split.upper()} SPLIT")
            logger.info(f"{'='*80}")

            # Check if input file exists
            input_file = os.path.join(PATHS['data'], f'{split}_hydrated.json.zst')
            if not os.path.exists(input_file):
                logger.warning(f"  ⚠️  {split} dataset not found, skipping...")
                continue

            # Load dataset
            dataset = read_compressed_json(input_file, logger)

            # Assign clusters
            logger.info(f"  Assigning cluster labels to {split} dataset...")
            updated_dataset, stats = assign_clusters_to_dataset(dataset, rule_mapping, subreddit_mapping, logger)

            # Update metadata version
            updated_dataset['metadata']['version'] = '1.1'
            updated_dataset['metadata']['cluster_labels_added'] = time.strftime('%Y-%m-%d')

            # Log statistics
            logger.info(f"  📊 Statistics:")
            logger.info(f"    Total subreddits: {stats['total_subreddits']}")
            logger.info(f"    Total thread pairs: {stats['total_thread_pairs']}")
            logger.info(f"    ")
            logger.info(f"    Rule Clusters:")
            logger.info(f"      Total pairs with rule clusters: {stats['pairs_with_rule_clusters']}")
            logger.info(f"    ")
            logger.info(f"    Subreddit Clusters:")
            logger.info(f"      Total subreddits with clusters: {stats['subreddits_with_clusters']}")

            # Save hydrated version
            hydrated_output = os.path.join(PATHS['data'], f'{split}_hydrated_clustered.json.zst')
            hydrated_size = write_compressed_json(updated_dataset, hydrated_output, logger=logger)

            # Create and save dehydrated version
            dehydrated = dehydrate_dataset(updated_dataset)
            dehydrated_output = os.path.join(PATHS['data'], f'{split}_dehydrated_clustered.json.zst')
            dehydrated_size = write_compressed_json(dehydrated, dehydrated_output, logger=logger)

            # Save uncompressed test set
            if split == 'test':
                uncompressed_file = os.path.join(PATHS['data'], 'test_hydrated_clustered.json')
                with open(uncompressed_file, 'w') as f:
                    json.dump(updated_dataset, f, indent=2)
                uncompressed_size = os.path.getsize(uncompressed_file) / (1024 * 1024)
                logger.info(f"  ✅ {uncompressed_file} ({uncompressed_size:.1f} MB)")
                output_files[split] = {
                    'hydrated': {'path': hydrated_output, 'size_mb': hydrated_size},
                    'dehydrated': {'path': dehydrated_output, 'size_mb': dehydrated_size},
                    'uncompressed': {'path': uncompressed_file, 'size_mb': uncompressed_size}
                }
            else:
                output_files[split] = {
                    'hydrated': {'path': hydrated_output, 'size_mb': hydrated_size},
                    'dehydrated': {'path': dehydrated_output, 'size_mb': dehydrated_size}
                }

            # Save split statistics
            all_stats[split] = {
                'total_subreddits': stats['total_subreddits'],
                'total_thread_pairs': stats['total_thread_pairs'],
                'pairs_with_rule_clusters': stats['pairs_with_rule_clusters'],
                'subreddits_with_clusters': stats['subreddits_with_clusters'],
                'top_10_rule_clusters': dict(stats['rule_cluster_distribution'].most_common(10)),
                'rule_clusters': dict(stats['rule_cluster_distribution']),  # Save all clusters for plotting
                'top_10_subreddit_clusters': dict(stats['subreddit_cluster_distribution'].most_common(10)),
                'subreddit_clusters': dict(stats['subreddit_cluster_pair_distribution'])  # Save for plotting (by thread pairs)
            }

        # Save overall statistics
        logger.info("\n" + "="*80)
        logger.info("SAVING STATISTICS")
        logger.info("="*80)

        summary_stats = {
            'metadata': {
                'stage': 10,
                'stage_name': 'Assign Cluster Labels',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': time.time() - start_time
            },
            'cluster_assignment_statistics': all_stats,
            'output_files': output_files,
            'total_unique_rules_mapped': len(rule_mapping),
            'total_unique_subreddits_mapped': len(subreddit_mapping)
        }

        stats_file = os.path.join(PATHS['data'], 'stage10_cluster_assignment_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)
        logger.info(f"  ✅ Saved statistics to: {stats_file}")

        # Create cluster distribution plots
        logger.info("\n" + "="*80)
        logger.info("CREATING CLUSTER DISTRIBUTION PLOTS")
        logger.info("="*80)

        logger.info("\n📊 Creating rule cluster distribution plot...")
        create_cluster_distribution_plot(all_stats, 'rule', logger)

        logger.info("\n📊 Creating subreddit cluster distribution plot...")
        create_cluster_distribution_plot(all_stats, 'subreddit', logger)

        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info(f"🎉 Stage 10 Complete! ({elapsed:.1f}s)")
        logger.info("="*80)
        log_stage_end(logger, 10, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 10 execution")
        log_stage_end(logger, 10, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
