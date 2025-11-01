#!/usr/bin/env python3
"""
Extract Mod Comments with Matched Rules Sorted by Similarity Score

Extracts moderator comments from train, val, and test_1k datasets,
including body, body_clean, and matched_rule information.
Outputs a single JSON file with all mod comments sorted by similarity score (ascending).

Input:
- data/train_hydrated.json.zst
- data/val_hydrated.json.zst
- data/test_1k_hydrated.json.zst

Output:
- data/mod_comments_sorted_by_similarity.json
"""

import sys
import os
import json
import zstandard
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS


def load_compressed_json(filepath: str) -> Dict:
    """Load a compressed JSON file."""
    print(f"Loading {filepath}...")

    with open(filepath, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            json_bytes = reader.read()
            return json.loads(json_bytes.decode('utf-8'))


def extract_mod_comments(dataset: Dict, split_name: str) -> List[Dict[str, Any]]:
    """
    Extract mod comments with body, body_clean, and matched_rule from a dataset.

    Returns:
        List of dictionaries with mod comment information
    """
    mod_comments = []

    for subreddit_data in dataset.get('subreddits', []):
        subreddit_name = subreddit_data.get('subreddit')

        for thread_pair in subreddit_data.get('thread_pairs', []):
            mod_comment = thread_pair.get('mod_comment', {})
            matched_rule = thread_pair.get('matched_rule', {})

            # Extract the required fields
            entry = {
                'split': split_name,
                'subreddit': subreddit_name,
                'mod_comment_id': thread_pair.get('mod_comment_id'),
                'submission_id': thread_pair.get('submission_id'),
                'body': mod_comment.get('body', ''),
                'body_clean': mod_comment.get('body_clean', ''),
                'matched_rule': {
                    'rule_index': matched_rule.get('rule_index'),
                    'short_name_clean': matched_rule.get('short_name_clean', ''),
                    'description_clean': matched_rule.get('description_clean', ''),
                    'similarity_score': matched_rule.get('similarity_score', 0.0)
                }
            }

            mod_comments.append(entry)

    return mod_comments


def main():
    """Main execution function."""
    print("=" * 80)
    print("Extracting Mod Comments Sorted by Similarity Score")
    print("=" * 80)

    all_mod_comments = []

    # Define dataset files to process
    datasets = [
        ('train', os.path.join(PATHS['data'], 'train_hydrated.json.zst')),
        ('val', os.path.join(PATHS['data'], 'val_hydrated.json.zst')),
        ('test_1k', os.path.join(PATHS['data'], 'test_1k_hydrated.json.zst'))
    ]

    # Extract mod comments from each dataset
    for split_name, filepath in datasets:
        if not os.path.exists(filepath):
            print(f"⚠️  Warning: {filepath} not found, skipping...")
            continue

        dataset = load_compressed_json(filepath)
        mod_comments = extract_mod_comments(dataset, split_name)
        all_mod_comments.extend(mod_comments)

        print(f"✅ Extracted {len(mod_comments):,} mod comments from {split_name}")

    if not all_mod_comments:
        print("❌ No mod comments found!")
        return 1

    print(f"\n📊 Total mod comments extracted: {len(all_mod_comments):,}")

    # Sort by similarity score (ascending order)
    print("🔄 Sorting by similarity score (ascending)...")
    all_mod_comments.sort(key=lambda x: x['matched_rule']['similarity_score'])

    # Show score range
    min_score = all_mod_comments[0]['matched_rule']['similarity_score']
    max_score = all_mod_comments[-1]['matched_rule']['similarity_score']
    print(f"   Similarity score range: {min_score:.4f} to {max_score:.4f}")

    # Create output structure
    output = {
        'metadata': {
            'total_mod_comments': len(all_mod_comments),
            'splits_included': [split_name for split_name, _ in datasets if os.path.exists(_)],
            'sorted_by': 'matched_rule.similarity_score (ascending)',
            'min_similarity_score': min_score,
            'max_similarity_score': max_score
        },
        'mod_comments': all_mod_comments
    }

    # Write output file
    output_file = os.path.join(PATHS['data'], 'mod_comments_sorted_by_similarity.json')
    print(f"\n💾 Writing output to: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Output saved: {output_size_mb:.1f} MB")

    # Show some statistics
    print(f"\n📈 Statistics by split:")
    from collections import Counter
    split_counts = Counter(comment['split'] for comment in all_mod_comments)
    for split_name, count in sorted(split_counts.items()):
        print(f"   {split_name}: {count:,} comments")

    print(f"\n🎉 Complete!")
    return 0


if __name__ == "__main__":
    exit(main())
