#!/usr/bin/env python3
"""
Reapply Cluster Labels from Analysis Text Files

After manually adding "NEW LABEL:" overrides in cluster analysis text files,
this script propagates those changes to JSON and metadata TSV files.

CLUSTER MERGING: If multiple clusters are given the same NEW LABEL, they will
be merged into a single logical cluster (keeping lowest cluster_id).

Workflow:
1. Review output/clustering/{entity}_cluster_analysis.txt
2. After any "LABEL: <original>" line, add "NEW LABEL: <your override>"
3. Run this script to propagate changes to JSON and metadata TSV

Usage:
    python reapply_cluster_labels.py                    # Reapply both subreddits and rules
    python reapply_cluster_labels.py --entity subreddit # Reapply only subreddits
    python reapply_cluster_labels.py --entity rule      # Reapply only rules

Input (source of truth for manual overrides):
- output/clustering/subreddit_cluster_analysis.txt (look for "NEW LABEL:" lines)
- output/clustering/rule_cluster_analysis.txt (look for "NEW LABEL:" lines)

Output (updated):
- output/clustering/subreddit_cluster_labels.json (updated with new labels)
- output/clustering/rule_cluster_labels.json (updated with new labels)
- output/embeddings/test_1k_subreddit_metadata.tsv (cluster_label column + cluster_id merged)
- output/embeddings/test_1k_rule_metadata.tsv (cluster_label column + cluster_id merged)
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Tuple
from collections import defaultdict


def parse_label_overrides(analysis_file: Path, logger) -> Dict[int, str]:
    """Parse cluster analysis text file for manual label overrides.

    Looks for patterns like:
        CLUSTER 5
        ...
        LABEL: original label
        NEW LABEL: my override

    Args:
        analysis_file: Path to cluster analysis text file
        logger: Logger instance

    Returns:
        Dict mapping cluster_id -> new_label (only for overridden clusters)
    """
    if not analysis_file.exists():
        logger.error(f"❌ Error: {analysis_file} not found")
        return {}

    logger.info(f"Parsing label overrides from {analysis_file}...")

    overrides = {}
    current_cluster_id = None
    last_line_was_label = False

    with open(analysis_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # Track current cluster
            if line.startswith('CLUSTER '):
                try:
                    current_cluster_id = int(line.replace('CLUSTER ', '').strip())
                except ValueError:
                    current_cluster_id = None

            # Track when we see a LABEL: line
            elif line.startswith('LABEL: '):
                last_line_was_label = True

            # Check for NEW LABEL: override
            elif line.startswith('NEW LABEL: '):
                if current_cluster_id is not None and last_line_was_label:
                    new_label = line.replace('NEW LABEL: ', '').strip()
                    overrides[current_cluster_id] = new_label
                    logger.info(f"  Found override for cluster {current_cluster_id}: {new_label}")
                last_line_was_label = False

            else:
                last_line_was_label = False

    if not overrides:
        logger.info(f"  No label overrides found (no 'NEW LABEL:' lines)")
    else:
        logger.info(f"  Found {len(overrides)} label overrides")

    return overrides


def detect_cluster_merges(overrides: Dict[int, str], all_labels: Dict[int, str], logger) -> Tuple[Dict[int, int], Dict[str, list]]:
    """Detect clusters that should be merged based on identical labels.

    This includes:
    1. Multiple overridden clusters with the same NEW LABEL
    2. Overridden clusters matching existing (non-overridden) cluster labels

    Args:
        overrides: Dict mapping cluster_id -> new_label (only overridden clusters)
        all_labels: Dict mapping cluster_id -> label (all clusters, including non-overridden)
        logger: Logger instance

    Returns:
        (merge_mapping, label_groups) where:
            - merge_mapping: Dict mapping old_cluster_id -> new_cluster_id (lowest in group)
            - label_groups: Dict mapping label -> list of cluster_ids with that label
    """
    # Build complete label mapping (overrides take precedence)
    complete_labels = dict(all_labels)  # Start with all existing labels
    complete_labels.update(overrides)   # Apply overrides

    # Group clusters by their label
    label_to_clusters = defaultdict(list)
    for cluster_id, label in complete_labels.items():
        label_to_clusters[label].append(cluster_id)

    # Build merge mapping (map to lowest cluster_id in each group)
    merge_mapping = {}
    label_groups = {}

    for label, cluster_ids in label_to_clusters.items():
        if len(cluster_ids) > 1:
            # Multiple clusters with same label - merge to lowest ID
            sorted_ids = sorted(cluster_ids)
            target_id = sorted_ids[0]
            label_groups[label] = sorted_ids

            # Only map clusters that were overridden or that need to merge
            for cid in sorted_ids:
                if cid != target_id:
                    merge_mapping[cid] = target_id

            # Check if this involves any overridden clusters
            overridden_ids = [cid for cid in sorted_ids if cid in overrides]
            if overridden_ids:
                logger.info(f"  Merging clusters {sorted_ids} → {target_id} (label: '{label}')")

    return merge_mapping, label_groups


def reapply_entity_labels(entity_type: str, embeddings_dir: Path, clustering_dir: Path, logger) -> None:
    """Reapply cluster labels from analysis text file to JSON and metadata.

    Args:
        entity_type: 'subreddit' or 'rule'
        embeddings_dir: Path to embeddings directory
        clustering_dir: Path to clustering directory
        logger: Logger instance
    """
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} - Reapplying Labels")
    logger.info("="*80)

    # Parse manual overrides from analysis text file (source of truth)
    analysis_file = clustering_dir / f'{entity_type}_cluster_analysis.txt'
    overrides = parse_label_overrides(analysis_file, logger)

    if not overrides:
        logger.info(f"No changes to apply for {entity_type}")
        return

    # Load existing JSON labels first (to detect merges with existing labels)
    labels_file = clustering_dir / f'{entity_type}_cluster_labels.json'
    if not labels_file.exists():
        logger.error(f"❌ Error: {labels_file} not found")
        logger.error("Run label_clusters.py first to generate initial labels")
        return

    logger.info(f"Loading existing labels from {labels_file}...")
    with open(labels_file) as f:
        cluster_data = json.load(f)

    # Build all_labels dict (existing labels for all clusters)
    all_labels = {int(cid): data['label'] for cid, data in cluster_data.items()}

    # Detect cluster merges (clusters with identical labels)
    logger.info("\nDetecting cluster merges...")
    merge_mapping, label_groups = detect_cluster_merges(overrides, all_labels, logger)

    if not merge_mapping:
        logger.info("  No merges detected (all labels are unique)")

    # Apply overrides to JSON
    logger.info("Applying overrides to JSON...")
    for cluster_id, new_label in overrides.items():
        cluster_key = str(cluster_id)
        if cluster_key in cluster_data:
            old_label = cluster_data[cluster_key]['label']
            cluster_data[cluster_key]['label'] = new_label
            logger.info(f"  Cluster {cluster_id}: '{old_label}' → '{new_label}'")
        else:
            logger.warning(f"  ⚠️  Cluster {cluster_id} not found in JSON, skipping")

    # Save updated JSON
    with open(labels_file, 'w') as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Updated {labels_file}")

    # Update metadata TSV
    metadata_file = embeddings_dir / f'test_1k_{entity_type}_metadata.tsv'
    if not metadata_file.exists():
        logger.error(f"❌ Error: {metadata_file} not found")
        return

    logger.info(f"\nUpdating metadata file: {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep='\t')

    if 'cluster_id' not in metadata.columns:
        logger.error(f"❌ Error: cluster_id column not found in {metadata_file}")
        logger.error("Run cluster_test_1k.py --apply-best first")
        return

    # Apply cluster merges (remap cluster_ids)
    if merge_mapping:
        logger.info("Applying cluster merges to metadata...")
        original_counts = metadata['cluster_id'].value_counts().to_dict()

        metadata['cluster_id'] = metadata['cluster_id'].map(
            lambda x: merge_mapping.get(x, x)
        )

        # Log merge statistics
        for label, cluster_ids in label_groups.items():
            target_id = min(cluster_ids)
            total_items = sum(original_counts.get(cid, 0) for cid in cluster_ids)
            logger.info(f"  Merged '{label}': {len(cluster_ids)} clusters → cluster {target_id} ({total_items} items)")

    # Build complete label mapping from updated JSON
    cluster_labels = {int(cid): data['label'] for cid, data in cluster_data.items()}

    # Apply labels (Noise for -1, use cluster_labels dict for others)
    metadata['cluster_label'] = metadata['cluster_id'].map(
        lambda x: cluster_labels.get(x, 'Noise' if x == -1 else f'Cluster {x}')
    )

    # Save updated metadata
    metadata.to_csv(metadata_file, sep='\t', index=False)
    logger.info(f"✅ Updated cluster_label column in {metadata_file}")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Reapply cluster labels from analysis text files to JSON and metadata'
    )
    parser.add_argument(
        '--entity',
        choices=['subreddit', 'rule'],
        help='Reapply labels for only one entity type'
    )
    args = parser.parse_args()

    # Create directories
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / 'logs' / 'clustering'
    embeddings_dir = base_dir / 'output' / 'embeddings'
    clustering_dir = base_dir / 'output' / 'clustering'

    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'reapply_labels_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(str(log_file)), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    logger.info("\nManual Override Workflow:")
    logger.info("  1. Edit output/clustering/{entity}_cluster_analysis.txt")
    logger.info("  2. Add 'NEW LABEL: <your label>' after any 'LABEL:' line")
    logger.info("  3. Run this script to apply changes")
    logger.info("\nCluster Merging:")
    logger.info("  - If multiple clusters get the same NEW LABEL, they will be merged")
    logger.info("  - Merged clusters keep the lowest cluster_id\n")

    try:
        # Determine entity types to process
        entity_types = [args.entity] if args.entity else ['subreddit', 'rule']

        # Reapply labels for each entity type
        for entity_type in entity_types:
            reapply_entity_labels(entity_type, embeddings_dir, clustering_dir, logger)

        logger.info("\n" + "="*80)
        logger.info("✅ COMPLETE - Labels reapplied successfully")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("  - Review updated JSON files in output/clustering/")
        logger.info("  - Review updated metadata files in output/embeddings/")
        logger.info("  - Regenerate plots with: python analysis/plot_clusters.py")

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
