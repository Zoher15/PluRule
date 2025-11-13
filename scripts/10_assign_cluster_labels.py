#!/usr/bin/env python3
"""
Stage 10: Assign Cluster Labels to Datasets

Reads cluster labels from analysis/label_clusters.py output and assigns them
to thread pairs in the train/val/test datasets based on matched rules.

Input:
- output/embeddings/all_rule_metadata.tsv (with cluster_id and cluster_label columns)
- data/{split}_hydrated.json.zst (train/val/test datasets from Stage 9)

Output:
- data/{split}_hydrated_clustered.json.zst (updated datasets with cluster labels)
- data/{split}_dehydrated_clustered.json.zst (dehydrated versions)
- data/test_hydrated_clustered.json (uncompressed test set)
- data/stage10_cluster_assignment_stats.json

The script adds two new fields to each thread pair's metadata:
- rule_cluster_id: The cluster ID for the matched rule (-1 for noise/unknown)
- rule_cluster_label: The semantic label for the cluster (e.g., "spoiler tags", "civility rules")
"""

import sys
import os
import time
import json
import pandas as pd
from typing import Dict, Tuple
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import write_json_file, read_compressed_json, write_compressed_json

# ============================================================================
# Data Loading
# ============================================================================

def load_rule_cluster_mapping(logger) -> Dict[Tuple[str, str], Dict]:
    """Load cluster labels from all_rule_metadata.tsv.

    Returns:
        Dict mapping (subreddit, short_name_clean) -> {cluster_id, cluster_label, cluster_probability}
    """
    # output/embeddings is at BASE_DATA/output/embeddings
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metadata_file = os.path.join(base_dir, 'output', 'embeddings', 'all_rule_metadata.tsv')

    if not os.path.exists(metadata_file):
        logger.error(f"❌ Rule metadata not found: {metadata_file}")
        logger.error("Please run analysis/label_clusters.py first")
        return {}

    logger.info(f"📋 Loading rule cluster mappings from {metadata_file}...")
    df = pd.read_csv(metadata_file, sep='\t')

    # Check for required columns
    required_cols = ['subreddit', 'short_name', 'cluster_id', 'cluster_label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns in metadata: {missing}")
        logger.error("Please run analysis/cluster_test_1k.py --apply-best and analysis/label_clusters.py")
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

        cluster_info = {
            'cluster_id': int(row['cluster_id']),
            'cluster_label': str(row['cluster_label']),
            'cluster_probability': float(row.get('cluster_probability', 0.0))
        }

        for subreddit in subreddits:
            key = (subreddit, short_name)
            mapping[key] = cluster_info

    logger.info(f"  ✅ Loaded cluster mappings for {len(mapping)} rules")
    return mapping




# ============================================================================
# Cluster Assignment
# ============================================================================

def assign_clusters_to_dataset(dataset: Dict, rule_mapping: Dict[Tuple[str, str], Dict], logger) -> Tuple[Dict, Dict]:
    """Assign cluster labels to all thread pairs in a dataset.

    Args:
        dataset: The hydrated dataset from Stage 9
        rule_mapping: Dict mapping (subreddit, short_name) -> cluster info
        logger: Logger instance

    Returns:
        (updated_dataset, statistics)
    """
    stats = {
        'total_subreddits': 0,
        'total_thread_pairs': 0,
        'pairs_with_clusters': 0,
        'pairs_without_clusters': 0,
        'cluster_distribution': Counter(),
        'missing_rules': Counter(),
        'noise_clusters': 0
    }

    for sub_data in dataset['subreddits']:
        subreddit = sub_data['subreddit'].lower()
        stats['total_subreddits'] += 1

        for pair in sub_data['thread_pairs']:
            stats['total_thread_pairs'] += 1

            # Get the matched rule from metadata
            matched_rule = pair['metadata'].get('rule', '')

            if not matched_rule:
                # No rule matched - shouldn't happen in normal data
                pair['metadata']['rule_cluster_id'] = -1
                pair['metadata']['rule_cluster_label'] = 'Unknown'
                pair['metadata']['rule_cluster_probability'] = 0.0
                stats['pairs_without_clusters'] += 1
                stats['missing_rules']['<empty>'] += 1
                continue

            # Look up cluster info
            key = (subreddit, matched_rule)
            cluster_info = rule_mapping.get(key)

            if cluster_info is None:
                # Rule not found in mapping (shouldn't happen if data is consistent)
                pair['metadata']['rule_cluster_id'] = -1
                pair['metadata']['rule_cluster_label'] = 'Unknown'
                pair['metadata']['rule_cluster_probability'] = 0.0
                stats['pairs_without_clusters'] += 1
                stats['missing_rules'][f"{subreddit}:{matched_rule}"] += 1
            else:
                # Assign cluster info
                pair['metadata']['rule_cluster_id'] = cluster_info['cluster_id']
                pair['metadata']['rule_cluster_label'] = cluster_info['cluster_label']
                pair['metadata']['rule_cluster_probability'] = cluster_info['cluster_probability']
                stats['pairs_with_clusters'] += 1
                stats['cluster_distribution'][cluster_info['cluster_label']] += 1

                if cluster_info['cluster_id'] == -1:
                    stats['noise_clusters'] += 1

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
            'rank': sub_data.get('rank')
        })

    dehydrated['metadata']['instructions'] = 'Use hydration script. All text fields contain [NEEDS_HYDRATION].'
    return dehydrated


# ============================================================================
# Main Execution
# ============================================================================

def main():
    logger = get_stage_logger(10, "assign_cluster_labels")
    log_stage_start(logger, 10, "Assign Cluster Labels to Datasets")
    start_time = time.time()

    try:
        create_directories()

        # Load rule cluster mappings
        rule_mapping = load_rule_cluster_mapping(logger)
        if not rule_mapping:
            logger.error("❌ Failed to load rule cluster mappings")
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
            updated_dataset, stats = assign_clusters_to_dataset(dataset, rule_mapping, logger)

            # Update metadata version
            updated_dataset['metadata']['version'] = '1.1'
            updated_dataset['metadata']['cluster_labels_added'] = time.strftime('%Y-%m-%d')

            # Log statistics
            logger.info(f"  📊 Statistics:")
            logger.info(f"    Total thread pairs: {stats['total_thread_pairs']}")
            logger.info(f"    Pairs with clusters: {stats['pairs_with_clusters']} ({100*stats['pairs_with_clusters']/max(stats['total_thread_pairs'],1):.1f}%)")
            logger.info(f"    Pairs without clusters: {stats['pairs_without_clusters']}")
            logger.info(f"    Noise clusters: {stats['noise_clusters']}")

            if stats['missing_rules']:
                logger.warning(f"    Missing rules: {len(stats['missing_rules'])} unique rules")
                # Show top 5 missing
                top_missing = stats['missing_rules'].most_common(5)
                for rule, count in top_missing:
                    logger.warning(f"      {rule}: {count} pairs")

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
                'pairs_with_clusters': stats['pairs_with_clusters'],
                'pairs_without_clusters': stats['pairs_without_clusters'],
                'noise_clusters': stats['noise_clusters'],
                'coverage_percentage': 100 * stats['pairs_with_clusters'] / max(stats['total_thread_pairs'], 1),
                'top_10_clusters': dict(stats['cluster_distribution'].most_common(10)),
                'missing_rules_count': len(stats['missing_rules']),
                'top_missing_rules': dict(stats['missing_rules'].most_common(10))
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
            'total_unique_rules_mapped': len(rule_mapping)
        }

        stats_file = os.path.join(PATHS['data'], 'stage10_cluster_assignment_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)
        logger.info(f"  ✅ Saved statistics to: {stats_file}")

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
