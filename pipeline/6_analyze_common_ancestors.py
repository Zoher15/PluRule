#!/usr/bin/env python3
"""
Analyze Common Ancestors Distribution from Stage 6 Discussion Threads

Reads all discussion_threads pickle files and computes distribution of
common ancestors between moderated and unmoderated thread pairs.

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 6)

Output:
- data/stage6_common_ancestors_analysis.json
"""

import sys
import os
import pickle
from collections import defaultdict
from typing import Dict, List
import statistics

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.logging import get_stage_logger
from utils.files import write_json_file


def load_discussion_threads(threads_file: str, logger) -> List[Dict]:
    """Load discussion threads from pickle file."""
    try:
        with open(threads_file, 'rb') as f:
            data = pickle.load(f)
            return data.get('thread_pairs', [])
    except Exception as e:
        logger.warning(f"Failed to load {threads_file}: {e}")
        return []


def analyze_common_ancestors(logger):
    """Analyze common ancestors distribution from all discussion threads."""
    logger.info("📊 Analyzing common ancestors distribution...")

    threads_dir = PATHS['discussion_threads']

    if not os.path.exists(threads_dir):
        logger.error(f"Discussion threads directory not found: {threads_dir}")
        return None

    # Find all discussion thread files
    thread_files = [f for f in os.listdir(threads_dir)
                   if f.endswith('_discussion_threads.pkl')]

    if not thread_files:
        logger.error(f"No discussion thread files found in {threads_dir}")
        return None

    logger.info(f"Found {len(thread_files)} discussion thread files")

    # Collect common ancestors data
    common_ancestors_list = []
    distribution = defaultdict(int)

    for thread_file in thread_files:
        filepath = os.path.join(threads_dir, thread_file)
        thread_pairs = load_discussion_threads(filepath, logger)

        for pair in thread_pairs:
            metadata = pair.get('metadata', {})
            common_ancestors = metadata.get('common_ancestors')

            if common_ancestors is not None:
                common_ancestors_list.append(common_ancestors)
                distribution[common_ancestors] += 1

    if not common_ancestors_list:
        logger.error("No common ancestors data found!")
        return None

    # Calculate statistics
    total_pairs = len(common_ancestors_list)
    average = statistics.mean(common_ancestors_list)
    median = statistics.median(common_ancestors_list)

    # Sort distribution by key for readability
    sorted_distribution = {str(k): v for k, v in sorted(distribution.items())}

    logger.info(f"✅ Analyzed {total_pairs:,} thread pairs")
    logger.info(f"   Average common ancestors: {average:.2f}")
    logger.info(f"   Median common ancestors: {median:.1f}")

    return {
        'total_pairs': total_pairs,
        'average': average,
        'median': median,
        'common_ancestors_distribution': sorted_distribution,
        'min': min(common_ancestors_list),
        'max': max(common_ancestors_list)
    }


def main():
    """Main execution function."""
    logger = get_stage_logger(6, "analyze_common_ancestors")
    logger.info("=" * 80)
    logger.info("Stage 6: Analyze Common Ancestors Distribution")
    logger.info("=" * 80)

    try:
        # Analyze common ancestors
        results = analyze_common_ancestors(logger)

        if results is None:
            logger.error("❌ Analysis failed!")
            return 1

        # Save results
        output_file = os.path.join(PATHS['data'], 'stage6_common_ancestors_analysis.json')
        write_json_file(results, output_file, pretty=True)

        logger.info(f"💾 Saved analysis to: {output_file}")
        logger.info("")
        logger.info("📊 Distribution summary:")
        logger.info(f"   Total pairs: {results['total_pairs']:,}")
        logger.info(f"   Average: {results['average']:.2f}")
        logger.info(f"   Median: {results['median']:.1f}")
        logger.info(f"   Range: {results['min']} - {results['max']}")
        logger.info("")
        logger.info("🎉 Analysis complete!")

        return 0

    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
