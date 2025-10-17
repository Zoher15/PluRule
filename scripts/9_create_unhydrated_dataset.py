#!/usr/bin/env python3
"""
Stage 9: Create Final Datasets

Creates two versions of the final dataset:
1. Hydrated (full objects with text) - for internal use
2. Unhydrated (IDs only) - for public release

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 6)
- comment_trees/{subreddit}_comment_trees.pkl (from Stage 6)
- submissions/{subreddit}_submissions.zst (from Stage 7)
- media/{subreddit}/* (from Stage 8)
- stage8_successful_submission_ids.json
- stage2_top_N_sfw_subreddits.json

Output:
- reddit_moderation_dataset_hydrated_v1.0.json.zst (full objects, internal use)
- reddit_moderation_dataset_unhydrated_v1.0.json.zst (IDs only, public release)
"""

import sys
import os
import time
import pickle
import zstandard
import json
import random
from typing import Dict, List, Any, Set
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS, PROCESSES, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads
from utils.stats import calculate_jsd_from_uniform, rank_by_score


def load_successful_submissions(logger) -> tuple:
    """Load submission IDs from Stage 8 and submission objects from Stage 7."""
    success_file = os.path.join(PATHS['data'], 'stage8_successful_submission_ids.json')
    if not os.path.exists(success_file):
        logger.error(f"Stage 8 success file not found: {success_file}")
        return {}, {}

    logger.info(f"Loading successful submission IDs from Stage 8...")
    data = read_json_file(success_file)
    subreddit_successful_ids = {sub: set(ids) for sub, ids in data.get('subreddit_submission_ids', {}).items()}

    total_ids = sum(len(ids) for ids in subreddit_successful_ids.values())
    logger.info(f"Loaded {total_ids} successful submission IDs across {len(subreddit_successful_ids)} subreddits")

    # Load submission objects from Stage 7
    logger.info(f"Loading submission objects from Stage 7...")
    subreddit_submission_objects = {}

    for subreddit, submission_ids in subreddit_successful_ids.items():
        submissions_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")
        if not os.path.exists(submissions_file):
            logger.warning(f"  Submissions file not found for r/{subreddit}")
            subreddit_submission_objects[subreddit] = {}
            continue

        submission_objects = {}
        for line in read_zst_lines(submissions_file):
            submission = json_loads(line)
            if submission.get('id') in submission_ids:
                submission_objects[submission['id']] = submission

        logger.info(f"  r/{subreddit}: Loaded {len(submission_objects)}/{len(submission_ids)} submissions")
        subreddit_submission_objects[subreddit] = submission_objects

    logger.info(f"✅ Loaded {sum(len(objs) for objs in subreddit_submission_objects.values())} total submissions")
    return subreddit_successful_ids, subreddit_submission_objects


def load_subreddit_rules(logger) -> Dict[str, List[Dict]]:
    """Load rules for all subreddits from Stage 2."""
    rules_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

    if not os.path.exists(rules_file):
        logger.error(f"Stage 2 rules file not found: {rules_file}")
        return {}

    logger.info(f"Loading subreddit rules from Stage 2...")
    data = read_json_file(rules_file)

    subreddit_rules = {}
    subreddit_languages = {}

    for entry in data.get('subreddits', []):
        subreddit_data = entry.get('subreddit', {})
        subreddit_name = subreddit_data.get('display_name', '').lower()

        if subreddit_name:
            subreddit_rules[subreddit_name] = entry.get('rules', [])
            subreddit_languages[subreddit_name] = subreddit_data.get('lang', 'unknown')

    logger.info(f"Loaded rules for {len(subreddit_rules)} subreddits")
    return subreddit_rules, subreddit_languages


def filter_thread_pairs(thread_pairs: List[Dict], successful_submission_ids: Set[str]) -> List[Dict]:
    """Filter thread pairs to only include those with successful Stage 8 submissions."""
    filtered_pairs = []

    for pair in thread_pairs:
        submission_id = pair.get('metadata', {}).get('submission_id')
        if submission_id and submission_id in successful_submission_ids:
            filtered_pairs.append(pair)

    return filtered_pairs


def collect_media_files(subreddit: str, submission_id: str, logger) -> List[str]:
    """Collect media file paths for a submission."""
    media_dir = os.path.join(PATHS['media'], subreddit)

    if not os.path.exists(media_dir):
        return []

    media_files = []
    prefix = f"{submission_id}_"

    try:
        for filename in os.listdir(media_dir):
            # Check for both submission_id_ prefix and submission_id.ext pattern
            if filename.startswith(prefix) or filename.startswith(f"{submission_id}."):
                media_files.append(f"{subreddit}/{filename}")

        return sorted(media_files)

    except Exception as e:
        logger.warning(f"Error reading media directory for r/{subreddit}/{submission_id}: {e}")
        return []


def strip_to_ids(hydrated_dataset: Dict) -> Dict:
    """Strip hydrated dataset to IDs only for public release."""

    unhydrated_dataset = {
        'metadata': hydrated_dataset['metadata'].copy(),
        'subreddits': []
    }

    for subreddit_data in hydrated_dataset['subreddits']:
        # Strip submissions to keep ID and media info with placeholder
        unhydrated_submissions = {}
        for sub_id, sub_data in subreddit_data['submissions'].items():
            unhydrated_submissions[sub_id] = {
                'id': sub_id,
                'submission_object': '[NEEDS_HYDRATION]',
                'num_media': sub_data.get('num_media', 0),
                'media_files': ['[NEEDS_HYDRATION]' for _ in sub_data.get('media_files', [])]  # Placeholder for each media file
            }

        unhydrated_subreddit = {
            'subreddit': subreddit_data['subreddit'],
            'data_version': subreddit_data['data_version'],
            'last_updated': subreddit_data['last_updated'],
            'rank': subreddit_data.get('rank'),
            'jsd_from_uniform': subreddit_data['jsd_from_uniform'],
            'language': subreddit_data['language'],
            'rules': subreddit_data['rules'],
            'total_thread_pairs': subreddit_data['total_thread_pairs'],
            'rule_distribution': subreddit_data['rule_distribution'],
            'submissions': unhydrated_submissions,
            'submission_trees': {},
            'thread_pairs': []
        }

        # Strip trees to IDs only
        for submission_id, tree in subreddit_data['submission_trees'].items():
            unhydrated_subreddit['submission_trees'][submission_id] = {
                'root_comments': tree.get('root_comments', []),
                'total_comments': tree.get('total_comments', 0),
                'max_depth': max(tree.get('depth_levels', {}).keys()) if tree.get('depth_levels') else 0,
                'depth_levels': {str(k): v for k, v in tree.get('depth_levels', {}).items()}
            }

        # Strip thread pairs to IDs only
        for pair in subreddit_data['thread_pairs']:
            unhydrated_pair = {
                'mod_comment_id': pair['mod_comment_id'],
                'mod_comment': '[NEEDS_HYDRATION]',  # Placeholder for hydration
                'mod_comment_date': pair.get('mod_comment_date'),  # Preserve timestamp
                'submission_id': pair['submission_id'],
                'submission_date': pair.get('submission_date'),  # Preserve timestamp
                'matched_rule': pair['matched_rule'],
                'rule_options': pair['rule_options'],  # Keep shuffled rule options
                'moderated_thread': [
                    {
                        'comment_id': comment.get('id'),
                        'comment_date': comment.get('created_utc'),  # Preserve timestamp
                        'level': comment.get('level'),
                        'comment_object': '[NEEDS_HYDRATION]'
                    }
                    for comment in pair['moderated_thread']
                ],
                'unmoderated_thread': [
                    {
                        'comment_id': comment.get('id'),
                        'comment_date': comment.get('created_utc'),  # Preserve timestamp
                        'level': comment.get('level'),
                        'comment_object': '[NEEDS_HYDRATION]'
                    }
                    for comment in pair['unmoderated_thread']
                ],
                'unmod_thread_metadata': pair['unmod_thread_metadata']
            }
            unhydrated_subreddit['thread_pairs'].append(unhydrated_pair)

        unhydrated_dataset['subreddits'].append(unhydrated_subreddit)

    return unhydrated_dataset


def process_subreddit(subreddit: str, successful_submission_ids: Set[str],
                     subreddit_rules: List[Dict], submission_objects: Dict[str, Dict], logger) -> Dict[str, Any]:
    """Process a single subreddit and extract hydrated data (full objects)."""

    logger.info(f"Processing r/{subreddit}...")

    # Load discussion threads
    threads_file = os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")
    if not os.path.exists(threads_file):
        logger.warning(f"  Thread file not found: {threads_file}")
        return None

    with open(threads_file, 'rb') as f:
        threads_data = pickle.load(f)

    # Filter thread pairs
    original_pairs = threads_data.get('thread_pairs', [])
    filtered_pairs = filter_thread_pairs(original_pairs, successful_submission_ids)

    if not filtered_pairs:
        logger.warning(f"  No thread pairs remain after Stage 8 filtering")
        return None

    logger.info(f"  Filtered {len(original_pairs)} -> {len(filtered_pairs)} thread pairs")

    # Skip subreddits with fewer than 500 pairs
    if len(filtered_pairs) < 500:
        logger.warning(f"  Skipping r/{subreddit}: only {len(filtered_pairs)} pairs (<500 required)")
        return None

    # Sample exactly 500 pairs if we have more than 500 (using seed 0 for reproducibility)
    if len(filtered_pairs) > 500:
        random.seed(0)
        filtered_pairs = random.sample(filtered_pairs, 500)
        logger.info(f"  Sampled {len(original_pairs)} -> 500 thread pairs (seed=0)")

    # Load comment trees
    trees_file = os.path.join(PATHS['comment_trees'], f"{subreddit}_comment_trees.pkl")
    if not os.path.exists(trees_file):
        logger.warning(f"  Trees file not found: {trees_file}")
        return None

    with open(trees_file, 'rb') as f:
        trees_data = pickle.load(f)

    # Extract submission IDs from filtered thread pairs
    submission_ids_in_pairs = set()
    for pair in filtered_pairs:
        submission_id = pair.get('metadata', {}).get('submission_id')
        if submission_id:
            submission_ids_in_pairs.add(submission_id)

    # Build HYDRATED structure (keep full objects)
    hydrated_pairs = []

    # Initialize rule_distribution with all rules at 0 (like Stage 6/8)
    rule_distribution = {}
    for rule in subreddit_rules:
        rule_name = rule.get('short_name_clean', '')
        if rule_name:
            rule_distribution[rule_name] = 0

    submissions = {}
    submission_trees = {}

    for pair in filtered_pairs:
        metadata = pair.get('metadata', {})
        submission_id = metadata.get('submission_id')

        # Keep FULL thread structure with all comment objects
        moderated_thread = pair.get('moderated_thread', [])
        unmoderated_thread = pair.get('unmoderated_thread', [])

        # Get matched rule
        matched_rule = metadata.get('rule')
        rule_similarity = metadata.get('rule_similarity_score', 0)

        # Find full rule data (use cleaned versions to match Stage 4)
        rule_data = None
        for rule in subreddit_rules:
            if rule.get('short_name') == matched_rule or rule.get('short_name_clean') == matched_rule:
                rule_data = {
                    'rule_index': rule.get('rule_index'),
                    'short_name': rule.get('short_name_clean'),
                    'description': rule.get('description_clean', ''),
                    'similarity_score': rule_similarity
                }
                break

        if not rule_data:
            logger.warning(f"  Could not find rule data for: {matched_rule}")
            continue

        # Count rule distribution
        rule_distribution[matched_rule] += 1

        # Create shuffled rule options using mod_comment_id as seed
        mod_comment_id = pair.get('mod_comment_id')

        # Create list of rule short names
        rule_short_names = [rule.get('short_name_clean') for rule in subreddit_rules if rule.get('short_name_clean')]

        # Shuffle using mod_comment_id as seed (convert ID to integer hash)
        if mod_comment_id:
            seed = int(mod_comment_id, 36) if isinstance(mod_comment_id, str) else hash(mod_comment_id)
            random.seed(seed)
            shuffled_rules = rule_short_names.copy()
            random.shuffle(shuffled_rules)
        else:
            shuffled_rules = rule_short_names

        # Create labeled rule options: (a) rule1, (b) rule2, etc.
        rule_options = []
        for i, rule_name in enumerate(shuffled_rules):
            label = chr(ord('a') + i)  # a, b, c, ...
            rule_options.append({
                'label': f'({label})',
                'rule': rule_name
            })

        # Create hydrated pair (with full objects)
        hydrated_pair = {
            'mod_comment_id': mod_comment_id,
            'mod_comment': pair.get('mod_comment'),  # Full moderator comment object
            'mod_comment_date': pair.get('mod_comment', {}).get('created_utc'),
            'submission_id': submission_id,
            'submission_date': submission_objects.get(submission_id, {}).get('created_utc'),
            'matched_rule': rule_data,
            'rule_options': rule_options,  # Shuffled list with labels
            'moderated_thread': moderated_thread,  # Full objects
            'unmoderated_thread': unmoderated_thread,  # Full objects
            'unmod_thread_metadata': {
                'common_ancestors': metadata.get('common_ancestors', 0),
                'moderated_comment_depth': len(moderated_thread) - 1 if moderated_thread else 0,
                'target_length': metadata.get('target_length', 0),
                'moderated_score': metadata.get('moderated_score', 0),
                'unmoderated_score': metadata.get('unmoderated_score', 0)
            }
        }

        hydrated_pairs.append(hydrated_pair)

    # Process submissions with media info and trees (keep full trees)
    for submission_id in submission_ids_in_pairs:
        # Collect media files
        media_files = collect_media_files(subreddit, submission_id, logger)

        # Get full submission object (or empty dict if not found)
        submission_obj = submission_objects.get(submission_id, {})

        submissions[submission_id] = {
            **submission_obj,  # Full submission object (title, selftext, url, author, score, etc.)
            'num_media': len(media_files),
            'media_files': media_files
        }

        # Keep FULL tree structure (convert depth_levels keys to strings for JSON compatibility)
        if submission_id in trees_data.get('trees', {}):
            tree = trees_data['trees'][submission_id].copy()
            # Convert integer keys to strings in depth_levels
            if 'depth_levels' in tree:
                tree['depth_levels'] = {str(k): v for k, v in tree['depth_levels'].items()}
            submission_trees[submission_id] = tree

    # Calculate JSD from uniform distribution
    jsd_score = calculate_jsd_from_uniform(dict(rule_distribution))

    logger.info(f"  ✅ r/{subreddit}: {len(hydrated_pairs)} pairs, {len(submissions)} submissions, JSD={jsd_score:.4f}")

    return {
        'subreddit': subreddit,
        'rules': subreddit_rules,
        'total_thread_pairs': len(hydrated_pairs),
        'rule_distribution': dict(rule_distribution),
        'jsd_from_uniform': jsd_score,
        'submissions': submissions,
        'submission_trees': submission_trees,
        'thread_pairs': hydrated_pairs  # Full objects
    }


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(9, "create_final_datasets")
    log_stage_start(logger, 9, "Create Final Datasets (Hydrated + Unhydrated)")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Load successful submissions from Stage 8 and submission objects from Stage 7
        logger.info("📋 Loading successful submissions...")
        subreddit_successful_ids, subreddit_submission_objects = load_successful_submissions(logger)

        if not subreddit_successful_ids:
            logger.error("❌ No successful submissions found!")
            log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load subreddit rules from Stage 2
        logger.info("📚 Loading subreddit rules from Stage 2...")
        subreddit_rules_map, subreddit_languages = load_subreddit_rules(logger)

        if not subreddit_rules_map:
            logger.error("❌ No subreddit rules found!")
            log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process each subreddit
        logger.info(f"🚀 Processing {len(subreddit_successful_ids)} subreddits...")

        subreddit_data = []
        total_thread_pairs = 0
        total_submissions = 0
        total_media_files = 0

        for subreddit, successful_ids in subreddit_successful_ids.items():
            rules = subreddit_rules_map.get(subreddit, [])
            language = subreddit_languages.get(subreddit, 'unknown')
            submission_objs = subreddit_submission_objects.get(subreddit, {})

            result = process_subreddit(subreddit, successful_ids, rules, submission_objs, logger)

            if result:
                # Add metadata
                result['data_version'] = '1.0'
                result['last_updated'] = time.strftime('%Y-%m-%d')
                result['language'] = language

                subreddit_data.append(result)

                total_thread_pairs += result['total_thread_pairs']
                total_submissions += len(result['submissions'])
                total_media_files += sum(s['num_media'] for s in result['submissions'].values())

        if not subreddit_data:
            logger.error("❌ No subreddit data collected!")
            log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"✅ Processed {len(subreddit_data)} subreddits successfully (all have exactly 500 pairs)")

        # Rank ALL subreddits by JSD (lower = better), regardless of language
        # Note: All subreddits already have exactly 500 pairs (sampled in process_subreddit)
        logger.info(f"📊 Ranking {len(subreddit_data)} subreddits by JSD...")

        final_subreddit_data = rank_by_score(subreddit_data, 'jsd_from_uniform', ascending=True)

        # Count English vs non-English for statistics
        english_count = sum(1 for s in final_subreddit_data if s.get('language') == 'en')
        non_english_count = len(final_subreddit_data) - english_count

        logger.info(f"  ✅ Ranked {len(final_subreddit_data)} subreddits ({english_count} English, {non_english_count} non-English)")

        # Load pipeline statistics
        stage1_stats = read_json_file(os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json'))
        stage4_stats = read_json_file(os.path.join(PATHS['data'], 'stage4_matching_summary.json'))
        stage6_stats = read_json_file(os.path.join(PATHS['data'], 'stage6_trees_and_threads_summary.json'))
        stage7_stats = read_json_file(os.path.join(PATHS['data'], 'stage7_submission_collection_stats.json'))
        stage8_stats = read_json_file(os.path.join(PATHS['data'], 'stage8_media_collection_stats.json'))

        # Create final HYDRATED dataset (full objects)
        hydrated_dataset = {
            'metadata': {
                'version': '1.0',
                'data_version': '1.0',
                'creation_date': time.strftime('%Y-%m-%d'),
                'total_subreddits': len(final_subreddit_data),
                'total_thread_pairs': total_thread_pairs,
                'total_submissions': total_submissions,
                'total_submissions_with_media': sum(1 for s in subreddit_data for sub in s['submissions'].values() if sub['num_media'] > 0),
                'total_media_files': total_media_files,

                # Pipeline info
                'embedding_model': stage4_stats.get('embedding_model', 'unknown'),
                'gold_threshold': stage4_stats.get('gold_threshold', 0),
                'ambiguous_threshold': stage4_stats.get('ambiguous_threshold', 0),
                'date_range': stage4_stats.get('date_range', ('unknown', 'unknown')),

                # Pipeline statistics
                'pipeline_statistics': {
                    'stage1_total_mod_comments': stage1_stats.get('summary', {}).get('total_mod_comments', 0),
                    'stage4_matched_comments': stage4_stats.get('total_matched', 0),
                    'stage6_successful_thread_pairs': stage6_stats.get('summary', {}).get('total_successful_pairs', 0),
                    'stage7_submissions_collected': stage7_stats.get('summary', {}).get('total_submissions_found', 0),
                    'stage8_final_thread_pairs': total_thread_pairs
                },

                'instructions': 'Internal use only - full dataset with complete comment/submission objects',
                'citation': 'TBD'
            },

            'subreddits': final_subreddit_data
        }

        # Save HYDRATED version (JSON with zstandard compression)
        hydrated_file = os.path.join(PATHS['data'], 'reddit_moderation_dataset_hydrated_v1.0.json.zst')
        logger.info(f"💾 Writing hydrated dataset to: {hydrated_file} (using {PROCESSES} threads)")

        with open(hydrated_file, 'wb') as f:
            compressor = zstandard.ZstdCompressor(level=3, threads=PROCESSES)
            with compressor.stream_writer(f) as writer:
                json_str = json.dumps(hydrated_dataset, indent=2)
                writer.write(json_str.encode('utf-8'))

        hydrated_size_mb = os.path.getsize(hydrated_file) / (1024 * 1024)
        logger.info(f"  ✅ Hydrated dataset saved: {hydrated_size_mb:.1f} MB")

        # Strip to IDs for UNHYDRATED version
        logger.info(f"🔄 Stripping to IDs for public release...")
        unhydrated_dataset = strip_to_ids(hydrated_dataset)

        # Update instructions for unhydrated version
        unhydrated_dataset['metadata']['instructions'] = 'Use hydration script at github.com/... to add comment/submission text and media files. Media files follow naming convention: {subreddit}/{submission_id}_{index}_{media_id}_{source}.{ext} for galleries or {submission_id}_{source}.{ext} for single files. Timestamps are provided in *_date fields (Unix epoch). All text fields and media_files contain [NEEDS_HYDRATION] placeholders.'

        # Save UNHYDRATED version (JSON with zstandard compression)
        unhydrated_file = os.path.join(PATHS['data'], 'reddit_moderation_dataset_unhydrated_v1.0.json.zst')
        logger.info(f"💾 Writing unhydrated dataset to: {unhydrated_file} (using {PROCESSES} threads)")

        with open(unhydrated_file, 'wb') as f:
            compressor = zstandard.ZstdCompressor(level=3, threads=PROCESSES)
            with compressor.stream_writer(f) as writer:
                json_str = json.dumps(unhydrated_dataset, indent=2)
                writer.write(json_str.encode('utf-8'))

        unhydrated_size_mb = os.path.getsize(unhydrated_file) / (1024 * 1024)
        logger.info(f"  ✅ Unhydrated dataset saved: {unhydrated_size_mb:.1f} MB")

        # Save sample datasets (top 5 subreddits) as readable JSON
        logger.info(f"📝 Creating sample datasets (top 5 subreddits)...")

        sample_hydrated = {
            'metadata': hydrated_dataset['metadata'].copy(),
            'subreddits': final_subreddit_data[:5]
        }
        sample_hydrated['metadata']['total_subreddits'] = 5
        sample_hydrated['metadata']['instructions'] = 'SAMPLE dataset - top 5 subreddits only. For full dataset see reddit_moderation_dataset_hydrated_v1.0.json.zst'

        sample_unhydrated = {
            'metadata': unhydrated_dataset['metadata'].copy(),
            'subreddits': unhydrated_dataset['subreddits'][:5]
        }
        sample_unhydrated['metadata']['total_subreddits'] = 5
        sample_unhydrated['metadata']['instructions'] = 'SAMPLE dataset - top 5 subreddits only. For full dataset see reddit_moderation_dataset_unhydrated_v1.0.json.zst'

        # Save as pretty-printed JSON (uncompressed for readability)
        sample_hydrated_file = os.path.join(PATHS['data'], 'reddit_moderation_dataset_hydrated_SAMPLE.json')
        write_json_file(sample_hydrated, sample_hydrated_file, pretty=True)
        sample_hydrated_mb = os.path.getsize(sample_hydrated_file) / (1024 * 1024)
        logger.info(f"  ✅ Sample hydrated: {sample_hydrated_file} ({sample_hydrated_mb:.1f} MB)")

        sample_unhydrated_file = os.path.join(PATHS['data'], 'reddit_moderation_dataset_unhydrated_SAMPLE.json')
        write_json_file(sample_unhydrated, sample_unhydrated_file, pretty=True)
        sample_unhydrated_mb = os.path.getsize(sample_unhydrated_file) / (1024 * 1024)
        logger.info(f"  ✅ Sample unhydrated: {sample_unhydrated_file} ({sample_unhydrated_mb:.1f} MB)")

        # Save statistics summary
        summary_stats = {
            'metadata': {
                'stage': 9,
                'stage_name': 'Create Final Datasets',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': time.time() - start_time
            },
            'dataset_statistics': {
                'total_subreddits_in_dataset': len(final_subreddit_data),
                'english_subreddits': english_count,
                'non_english_subreddits': non_english_count,
                'thread_pairs_per_subreddit': 500,
                'total_thread_pairs': total_thread_pairs,
                'total_submissions': total_submissions,
                'total_submissions_with_media': hydrated_dataset['metadata']['total_submissions_with_media'],
                'total_media_files': total_media_files
            },
            'output_files': {
                'hydrated': {
                    'filename': os.path.basename(hydrated_file),
                    'path': hydrated_file,
                    'size_mb': hydrated_size_mb,
                    'format': 'json.zst',
                    'compression': f'zstandard (level 3, {PROCESSES} threads)',
                    'description': 'Full dataset with complete objects (internal use)'
                },
                'unhydrated': {
                    'filename': os.path.basename(unhydrated_file),
                    'path': unhydrated_file,
                    'size_mb': unhydrated_size_mb,
                    'format': 'json.zst',
                    'compression': f'zstandard (level 3, {PROCESSES} threads)',
                    'description': 'IDs-only dataset for public release'
                }
            },
            'pipeline_statistics': hydrated_dataset['metadata']['pipeline_statistics'],
            'top_10_subreddits': [
                {
                    'rank': s.get('rank', 0),
                    'subreddit': s['subreddit'],
                    'jsd_from_uniform': s['jsd_from_uniform'],
                    'total_thread_pairs': s['total_thread_pairs']
                }
                for s in final_subreddit_data[:10]
            ]
        }

        stats_file = os.path.join(PATHS['data'], 'stage9_final_datasets_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)
        logger.info(f"📊 Statistics saved to: {stats_file}")

        elapsed = time.time() - start_time

        logger.info(f"🎉 Stage 9 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Final dataset statistics:")
        logger.info(f"  Subreddits in dataset: {len(final_subreddit_data)} (all with exactly 500 pairs)")
        logger.info(f"  Thread pairs: {total_thread_pairs:,}")
        logger.info(f"  Submissions: {total_submissions:,}")
        logger.info(f"  Submissions with media: {hydrated_dataset['metadata']['total_submissions_with_media']:,}")
        logger.info(f"  Media files: {total_media_files:,}")
        logger.info(f"")
        logger.info(f"📁 Output files:")
        logger.info(f"  Hydrated (full objects):   {hydrated_file} ({hydrated_size_mb:.1f} MB)")
        logger.info(f"  Unhydrated (IDs only):     {unhydrated_file} ({unhydrated_size_mb:.1f} MB)")
        logger.info(f"  Sample hydrated (top 5):   {sample_hydrated_file} ({sample_hydrated_mb:.1f} MB)")
        logger.info(f"  Sample unhydrated (top 5): {sample_unhydrated_file} ({sample_unhydrated_mb:.1f} MB)")

        # Show top 10 ranked subreddits
        if final_subreddit_data:
            logger.info(f"🏆 Top 10 subreddits by JSD ranking:")
            for s in final_subreddit_data[:10]:
                rank = s.get('rank', 0)
                subreddit = s['subreddit']
                jsd = s['jsd_from_uniform']
                pairs = s['total_thread_pairs']
                logger.info(f"  {rank:2d}. r/{subreddit}: JSD={jsd:.4f}, {pairs:,} pairs")

        log_stage_end(logger, 9, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9 execution")
        log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
