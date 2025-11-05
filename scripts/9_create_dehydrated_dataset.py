#!/usr/bin/env python3
"""
Stage 9: Create Final Datasets with Train/Val/Test Splits

Creates train/val/test splits from subreddits with ≥25 thread pairs:
- Test: First 25 pairs from each subreddit
- Val: 10% of remaining pairs (after test) from each subreddit
- Train: 90% of remaining pairs (after test) from each subreddit

All three datasets are combined across subreddits (not per-subreddit splits).

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 6)
- comment_trees/{subreddit}_comment_trees.pkl (from Stage 6)
- submissions/{subreddit}_submissions.zst (from Stage 7)
- media/{subreddit}/* (from Stage 8)
- stage8_successful_submission_ids.json
- stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json

Output:
- train_hydrated.json.zst / train_dehydrated.json.zst
- val_hydrated.json.zst / val_dehydrated.json.zst
- test_hydrated.json.zst / test_dehydrated.json.zst
- test_hydrated.json (uncompressed)
- stage9_final_datasets_stats.json
"""

import sys
import os
import time
import pickle
import zstandard
import json
import random
from typing import Dict, List, Any, Set, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, MIN_MATCHED_COMMENTS, PROCESSES, MIN_TEST_THREAD_PAIRS,
                    TEST_PAIRS_PER_SUBREDDIT, VAL_SPLIT_RATIO, TRAIN_SPLIT_RATIO, create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads
from utils.stats import calculate_jsd_from_uniform, rank_by_score

# ============================================================================
# Data Loading
# ============================================================================

def load_successful_submissions(logger) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Dict]]]:
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

        # Convert to set for O(1) lookup
        needed_ids = set(submission_ids)
        submission_objects = {}
        found_count = 0
        removed_count = 0

        for line in read_zst_lines(submissions_file):
            if found_count >= len(needed_ids):
                break  # Early exit

            submission = json_loads(line)
            sub_id = submission.get('id')
            if sub_id in needed_ids:
                # Skip submissions with [removed] in selftext_html
                selftext_html = submission.get('selftext_html', '')
                if selftext_html and '[removed]' in selftext_html:
                    removed_count += 1
                    found_count += 1
                    continue

                submission_objects[sub_id] = submission
                found_count += 1

        if removed_count > 0:
            logger.info(f"  r/{subreddit}: Loaded {len(submission_objects)}/{len(submission_ids)} ({removed_count} filtered with [removed])")
        else:
            logger.info(f"  r/{subreddit}: Loaded {len(submission_objects)}/{len(submission_ids)}")
        subreddit_submission_objects[subreddit] = submission_objects

    logger.info(f"✅ Loaded {sum(len(objs) for objs in subreddit_submission_objects.values())} total submissions")
    return subreddit_successful_ids, subreddit_submission_objects


def load_subreddit_rules(logger) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """Load rules for all subreddits from Stage 2."""
    rules_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')

    if not os.path.exists(rules_file):
        logger.error(f"Stage 2 rules file not found: {rules_file}")
        return {}, {}

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

# ============================================================================
# Filtering and Sampling
# ============================================================================

def filter_thread_pairs(thread_pairs: List[Dict], successful_submission_ids: Set[str],
                       submission_objects: Dict[str, Dict]) -> List[Dict]:
    """
    Filter thread pairs to only include those with:
    - Successful Stage 8 submissions
    - Submissions not filtered out (e.g., [removed] in selftext_html)
    """
    filtered_pairs = []

    for pair in thread_pairs:
        metadata = pair.get('metadata', {})
        submission_id = metadata.get('submission_id')

        # Check Stage 8 success and submission exists
        if submission_id and submission_id in successful_submission_ids and submission_id in submission_objects:
            filtered_pairs.append(pair)

    return filtered_pairs


def split_pairs(thread_pairs: List[Dict], subreddit: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split thread pairs into test/val/train using deterministic sampling.

    Logic:
    - First TEST_PAIRS_PER_SUBREDDIT pairs → test
    - Remaining pairs: VAL_SPLIT_RATIO → val, TRAIN_SPLIT_RATIO → train

    Returns: (test_pairs, val_pairs, train_pairs)
    """
    # Create subreddit-specific RNG for deterministic sampling
    subreddit_rng = random.Random(hash(subreddit))

    # Shuffle all pairs
    shuffled_pairs = thread_pairs.copy()
    subreddit_rng.shuffle(shuffled_pairs)

    # First TEST_PAIRS_PER_SUBREDDIT → test
    test_pairs = shuffled_pairs[:TEST_PAIRS_PER_SUBREDDIT]
    remaining_pairs = shuffled_pairs[TEST_PAIRS_PER_SUBREDDIT:]

    if not remaining_pairs:
        return test_pairs, [], []

    # Split remaining: VAL_SPLIT_RATIO val, TRAIN_SPLIT_RATIO train
    val_size = int(len(remaining_pairs) * VAL_SPLIT_RATIO)

    val_pairs = remaining_pairs[:val_size]
    train_pairs = remaining_pairs[val_size:]

    return test_pairs, val_pairs, train_pairs

# ============================================================================
# Media Collection
# ============================================================================

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

# ============================================================================
# Dataset Processing
# ============================================================================

def create_shuffled_rule_options(rule_short_names: List[str], mod_comment_id: str, suffix: str) -> List[Dict]:
    """Create shuffled rule options with labels (a), (b), (c), etc. using deterministic seeding."""
    seed_str = mod_comment_id + suffix
    seed = int(mod_comment_id, 36) + hash(seed_str)
    shuffle_rng = random.Random(seed)
    shuffled_rules = rule_short_names.copy()
    shuffle_rng.shuffle(shuffled_rules)

    rule_options = []
    for i, rule_name in enumerate(shuffled_rules):
        label = chr(ord('a') + i)  # a, b, c, ...
        rule_options.append({
            'label': f'({label})',
            'rule': rule_name
        })

    return rule_options


def process_pairs_to_dataset(pairs: List[Dict], subreddit: str, subreddit_rules: List[Dict],
                             submission_objects: Dict[str, Dict], trees_data: Dict,
                             logger) -> Dict[str, Any]:
    """Process thread pairs into dataset format with full objects."""

    # Initialize rule_distribution with all rules at 0
    rule_distribution = {}
    for rule in subreddit_rules:
        rule_name = rule.get('short_name_clean', '')
        if rule_name:
            rule_distribution[rule_name] = 0

    hydrated_pairs = []
    submissions = {}
    submission_trees = {}

    # Extract submission IDs from pairs
    submission_ids_in_pairs = set()
    for pair in pairs:
        submission_id = pair.get('metadata', {}).get('submission_id')
        if submission_id:
            submission_ids_in_pairs.add(submission_id)

    for pair in pairs:
        metadata = pair.get('metadata', {})
        submission_id = metadata.get('submission_id')
        matched_rule = metadata.get('rule')
        rule_similarity = metadata.get('rule_similarity_score', 0)

        # Keep FULL thread structure
        moderated_thread = pair.get('moderated_thread', [])
        unmoderated_thread = pair.get('unmoderated_thread', [])

        # Find full rule data
        rule_data = None
        for rule in subreddit_rules:
            if rule.get('short_name') == matched_rule or rule.get('short_name_clean') == matched_rule:
                rule_data = {
                    'rule_index': rule.get('rule_index'),
                    'short_name_clean': rule.get('short_name_clean'),
                    'description_clean': rule.get('description_clean', ''),
                    'similarity_score': rule_similarity
                }
                break

        if not rule_data:
            logger.warning(f"  Could not find rule data for: {matched_rule}")
            continue

        # Count rule distribution
        rule_distribution[matched_rule] += 1

        mod_comment_id = pair.get('mod_comment_id')

        # Create list of rule short names plus "No rules broken"
        rule_short_names = [rule.get('short_name_clean') for rule in subreddit_rules if rule.get('short_name_clean')]
        rule_short_names.append('No rules broken')

        # Create different shuffled rule options for moderated and unmoderated threads
        moderated_rule_options = create_shuffled_rule_options(rule_short_names, mod_comment_id, "_mod")
        unmoderated_rule_options = create_shuffled_rule_options(rule_short_names, mod_comment_id, "_unmod")

        # Create hydrated pair
        hydrated_pair = {
            'mod_comment_id': mod_comment_id,
            'mod_comment': pair.get('mod_comment'),
            'mod_comment_date': pair.get('mod_comment', {}).get('created_utc'),
            'submission_id': submission_id,
            'submission_date': submission_objects.get(submission_id, {}).get('created_utc'),
            'matched_rule': rule_data,
            'moderated_rule_options': moderated_rule_options,
            'unmoderated_rule_options': unmoderated_rule_options,
            'moderated_thread': moderated_thread,
            'unmoderated_thread': unmoderated_thread,
            'unmod_thread_metadata': {
                'common_ancestors': metadata.get('common_ancestors', 0),
                'moderated_comment_depth': len(moderated_thread) - 1 if moderated_thread else 0,
                'target_length': metadata.get('target_length', 0),
                'moderated_score': metadata.get('moderated_score', 0),
                'unmoderated_score': metadata.get('unmoderated_score', 0)
            }
        }

        hydrated_pairs.append(hydrated_pair)

    # Process submissions with media info and trees
    for submission_id in submission_ids_in_pairs:
        # Collect media files
        media_files = collect_media_files(subreddit, submission_id, logger)

        # Get full submission object
        submission_obj = submission_objects.get(submission_id, {})

        submissions[submission_id] = {
            **submission_obj,
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

    return {
        'subreddit': subreddit,
        'rules': subreddit_rules,
        'total_thread_pairs': len(hydrated_pairs),
        'rule_distribution': dict(rule_distribution),
        'jsd_from_uniform': jsd_score,
        'submissions': submissions,
        'submission_trees': submission_trees,
        'thread_pairs': hydrated_pairs
    }


def process_subreddit(subreddit: str, successful_submission_ids: Set[str],
                     subreddit_rules: List[Dict], submission_objects: Dict[str, Dict],
                     language: str, logger) -> Tuple[Dict, Dict, Dict]:
    """
    Process a single subreddit and return test/val/train data.

    Returns: (test_data, val_data, train_data)
    Each is None if subreddit doesn't have enough pairs for that split.
    """

    logger.info(f"Processing r/{subreddit}...")

    # Load discussion threads
    threads_file = os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")
    if not os.path.exists(threads_file):
        logger.warning(f"  Thread file not found: {threads_file}")
        return None, None, None

    with open(threads_file, 'rb') as f:
        threads_data = pickle.load(f)

    # Filter thread pairs (Stage 8 success + [removed])
    original_pairs = threads_data.get('thread_pairs', [])
    filtered_pairs = filter_thread_pairs(original_pairs, successful_submission_ids, submission_objects)

    if len(filtered_pairs) < MIN_TEST_THREAD_PAIRS:
        logger.warning(f"  Skipping: only {len(filtered_pairs)} pairs (<{MIN_TEST_THREAD_PAIRS} required)")
        return None, None, None

    logger.info(f"  Filtered {len(original_pairs)} -> {len(filtered_pairs)} pairs (Stage 8 + [removed])")

    # Load comment trees
    trees_file = os.path.join(PATHS['comment_trees'], f"{subreddit}_comment_trees.pkl")
    if not os.path.exists(trees_file):
        logger.warning(f"  Trees file not found: {trees_file}")
        return None, None, None

    with open(trees_file, 'rb') as f:
        trees_data = pickle.load(f)

    # Split into test/val/train
    test_pairs, val_pairs, train_pairs = split_pairs(filtered_pairs, subreddit)

    logger.info(f"  Split: {len(test_pairs)} test, {len(val_pairs)} val, {len(train_pairs)} train")

    # Process each split
    test_data = None
    val_data = None
    train_data = None

    if test_pairs:
        test_data = process_pairs_to_dataset(test_pairs, subreddit, subreddit_rules,
                                            submission_objects, trees_data, logger)
        if test_data:
            test_data['data_version'] = '1.0'
            test_data['last_updated'] = time.strftime('%Y-%m-%d')
            test_data['language'] = language

    if val_pairs:
        val_data = process_pairs_to_dataset(val_pairs, subreddit, subreddit_rules,
                                           submission_objects, trees_data, logger)
        if val_data:
            val_data['data_version'] = '1.0'
            val_data['last_updated'] = time.strftime('%Y-%m-%d')
            val_data['language'] = language

    if train_pairs:
        train_data = process_pairs_to_dataset(train_pairs, subreddit, subreddit_rules,
                                             submission_objects, trees_data, logger)
        if train_data:
            train_data['data_version'] = '1.0'
            train_data['last_updated'] = time.strftime('%Y-%m-%d')
            train_data['language'] = language

    return test_data, val_data, train_data

# ============================================================================
# Dehydration
# ============================================================================

def dehydrate_dataset(hydrated_dataset: Dict) -> Dict:
    """Strip hydrated dataset to IDs only for public release."""

    dehydrated_dataset = {
        'metadata': hydrated_dataset['metadata'].copy(),
        'subreddits': []
    }

    for subreddit_data in hydrated_dataset['subreddits']:
        # Strip submissions to keep ID and media info with placeholder
        dehydrated_submissions = {}
        for sub_id, sub_data in subreddit_data['submissions'].items():
            dehydrated_submissions[sub_id] = {
                'id': sub_id,
                'submission_object': '[NEEDS_HYDRATION]',
                'num_media': sub_data.get('num_media', 0),
                'media_files': ['[NEEDS_HYDRATION]' for _ in sub_data.get('media_files', [])]
            }

        dehydrated_subreddit = {
            'subreddit': subreddit_data['subreddit'],
            'data_version': subreddit_data['data_version'],
            'last_updated': subreddit_data['last_updated'],
            'rank': subreddit_data.get('rank'),
            'jsd_from_uniform': subreddit_data['jsd_from_uniform'],
            'language': subreddit_data['language'],
            'rules': subreddit_data['rules'],
            'total_thread_pairs': subreddit_data['total_thread_pairs'],
            'rule_distribution': subreddit_data['rule_distribution'],
            'submissions': dehydrated_submissions,
            'submission_trees': {},
            'thread_pairs': []
        }

        # Strip trees to IDs only
        for submission_id, tree in subreddit_data['submission_trees'].items():
            depth_levels = tree.get('depth_levels', {})
            dehydrated_subreddit['submission_trees'][submission_id] = {
                'children_map': tree.get('children_map', {}),
                'parent_map': tree.get('parent_map', {}),
                'root_comments': tree.get('root_comments', []),
                'total_comments': tree.get('total_comments', 0),
                'max_depth': max(int(k) for k in depth_levels.keys()) if depth_levels else 0,
                'depth_levels': {str(k): v for k, v in depth_levels.items()}
            }

        # Strip thread pairs to IDs only
        for pair in subreddit_data['thread_pairs']:
            dehydrated_pair = {
                'mod_comment_id': pair['mod_comment_id'],
                'mod_comment': '[NEEDS_HYDRATION]',
                'mod_comment_date': pair.get('mod_comment_date'),
                'submission_id': pair['submission_id'],
                'submission_date': pair.get('submission_date'),
                'matched_rule': pair['matched_rule'],
                'moderated_rule_options': pair['moderated_rule_options'],
                'unmoderated_rule_options': pair['unmoderated_rule_options'],
                'moderated_thread': [
                    {
                        'comment_id': comment.get('id'),
                        'comment_date': comment.get('created_utc'),
                        'level': comment.get('level'),
                        'comment_object': '[NEEDS_HYDRATION]'
                    }
                    for comment in pair['moderated_thread']
                ],
                'unmoderated_thread': [
                    {
                        'comment_id': comment.get('id'),
                        'comment_date': comment.get('created_utc'),
                        'level': comment.get('level'),
                        'comment_object': '[NEEDS_HYDRATION]'
                    }
                    for comment in pair['unmoderated_thread']
                ],
                'unmod_thread_metadata': pair['unmod_thread_metadata']
            }
            dehydrated_subreddit['thread_pairs'].append(dehydrated_pair)

        dehydrated_dataset['subreddits'].append(dehydrated_subreddit)

    return dehydrated_dataset

# ============================================================================
# Writing Output
# ============================================================================

def write_compressed_dataset(dataset: Dict, filepath: str, logger):
    """Write dataset as compressed JSON."""
    logger.info(f"💾 Writing dataset to: {filepath} (using {PROCESSES} threads)")

    with open(filepath, 'wb') as f:
        compressor = zstandard.ZstdCompressor(level=3, threads=PROCESSES)
        with compressor.stream_writer(f) as writer:
            json_str = json.dumps(dataset)
            writer.write(json_str.encode('utf-8'))

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"  ✅ Dataset saved: {size_mb:.1f} MB")
    return size_mb

# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(9, "create_final_datasets")
    log_stage_start(logger, 9, "Create Final Datasets (Train/Val/Test)")

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

        test_subreddits = []
        val_subreddits = []
        train_subreddits = []

        for subreddit, successful_ids in subreddit_successful_ids.items():
            rules = subreddit_rules_map.get(subreddit, [])
            language = subreddit_languages.get(subreddit, 'unknown')
            submission_objs = subreddit_submission_objects.get(subreddit, {})

            test_data, val_data, train_data = process_subreddit(
                subreddit, successful_ids, rules, submission_objs, language, logger
            )

            if test_data:
                test_subreddits.append(test_data)
            if val_data:
                val_subreddits.append(val_data)
            if train_data:
                train_subreddits.append(train_data)

        logger.info(f"✅ Processed subreddits:")
        logger.info(f"  Test: {len(test_subreddits)} subreddits")
        logger.info(f"  Val: {len(val_subreddits)} subreddits")
        logger.info(f"  Train: {len(train_subreddits)} subreddits")

        if not test_subreddits:
            logger.error("❌ No subreddit data collected!")
            log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Rank subreddits by JSD for each split
        logger.info(f"📊 Ranking subreddits by JSD...")

        if test_subreddits:
            test_subreddits = rank_by_score(test_subreddits, 'jsd_from_uniform', ascending=True)
        if val_subreddits:
            val_subreddits = rank_by_score(val_subreddits, 'jsd_from_uniform', ascending=True)
        if train_subreddits:
            train_subreddits = rank_by_score(train_subreddits, 'jsd_from_uniform', ascending=True)

        # Load pipeline statistics
        stage1_stats = read_json_file(os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json'))
        stage4_stats = read_json_file(os.path.join(PATHS['data'], 'stage4_matching_summary.json'))
        stage6_stats = read_json_file(os.path.join(PATHS['data'], 'stage6_trees_and_threads_summary.json'))
        stage7_stats = read_json_file(os.path.join(PATHS['data'], 'stage7_submission_collection_stats.json'))

        # Create metadata template
        def create_metadata(split_name: str, subreddits: List[Dict]) -> Dict:
            total_pairs = sum(s['total_thread_pairs'] for s in subreddits)
            total_submissions = sum(len(s['submissions']) for s in subreddits)
            total_media = sum(s['num_media'] for sub in subreddits for s in sub['submissions'].values())

            # Count total comments across all thread pairs
            total_comments = 0
            for sub in subreddits:
                for pair in sub['thread_pairs']:
                    total_comments += len(pair.get('moderated_thread', []))
                    total_comments += len(pair.get('unmoderated_thread', []))

            return {
                'version': '1.0',
                'split': split_name,
                'creation_date': time.strftime('%Y-%m-%d'),
                'total_subreddits': len(subreddits),
                'total_thread_pairs': total_pairs,
                'total_submissions': total_submissions,
                'total_submissions_with_media': sum(1 for sub in subreddits for s in sub['submissions'].values() if s['num_media'] > 0),
                'total_media_files': total_media,
                'total_comments': total_comments,
                'embedding_model': stage4_stats.get('embedding_model', 'unknown'),
                'gold_threshold': stage4_stats.get('gold_threshold', 0),
                'ambiguous_threshold': stage4_stats.get('ambiguous_threshold', 0),
                'date_range': stage4_stats.get('date_range', ('unknown', 'unknown')),
                'pipeline_statistics': {
                    'stage1_total_mod_comments': stage1_stats.get('summary', {}).get('total_mod_comments', 0),
                    'stage4_matched_comments': stage4_stats.get('total_matched', 0),
                    'stage6_successful_thread_pairs': stage6_stats.get('summary', {}).get('total_successful_pairs', 0),
                    'stage7_submissions_collected': stage7_stats.get('summary', {}).get('total_submissions_found', 0),
                },
            }

        # Create datasets for each split
        datasets = {}

        if test_subreddits:
            datasets['test'] = {
                'metadata': create_metadata('test', test_subreddits),
                'subreddits': test_subreddits
            }

        if val_subreddits:
            datasets['val'] = {
                'metadata': create_metadata('val', val_subreddits),
                'subreddits': val_subreddits
            }

        if train_subreddits:
            datasets['train'] = {
                'metadata': create_metadata('train', train_subreddits),
                'subreddits': train_subreddits
            }

        # Write hydrated and dehydrated versions for each split
        logger.info(f"💾 Writing datasets...")

        output_files = {}

        for split_name, dataset in datasets.items():
            # Write hydrated version
            hydrated_file = os.path.join(PATHS['data'], f'{split_name}_hydrated.json.zst')
            hydrated_size = write_compressed_dataset(dataset, hydrated_file, logger)

            # Create and write dehydrated version
            logger.info(f"🔄 Dehydrating {split_name} dataset...")
            dehydrated_dataset = dehydrate_dataset(dataset)
            dehydrated_dataset['metadata']['instructions'] = 'Use hydration script to add comment/submission text and media files. All text fields contain [NEEDS_HYDRATION] placeholders.'

            dehydrated_file = os.path.join(PATHS['data'], f'{split_name}_dehydrated.json.zst')
            dehydrated_size = write_compressed_dataset(dehydrated_dataset, dehydrated_file, logger)

            output_files[split_name] = {
                'hydrated': {'path': hydrated_file, 'size_mb': hydrated_size},
                'dehydrated': {'path': dehydrated_file, 'size_mb': dehydrated_size}
            }

            # Write uncompressed JSON for test split
            if split_name == 'test':
                uncompressed_file = os.path.join(PATHS['data'], 'test_hydrated.json')
                logger.info(f"💾 Writing uncompressed test dataset to: {uncompressed_file}")
                with open(uncompressed_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
                uncompressed_size = os.path.getsize(uncompressed_file) / (1024 * 1024)
                logger.info(f"  ✅ Uncompressed test dataset saved: {uncompressed_size:.1f} MB")
                output_files[split_name]['uncompressed'] = {'path': uncompressed_file, 'size_mb': uncompressed_size}

        # Calculate overall statistics
        overall_comments = 0
        overall_threads = 0

        for subreddit_list in [test_subreddits, val_subreddits, train_subreddits]:
            for sub in subreddit_list:
                for pair in sub['thread_pairs']:
                    overall_comments += len(pair.get('moderated_thread', []))
                    overall_comments += len(pair.get('unmoderated_thread', []))
                    overall_threads += 2  # mod + unmod = 2 threads per pair

        # Save statistics summary
        summary_stats = {
            'metadata': {
                'stage': 9,
                'stage_name': 'Create Final Datasets',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processing_time_seconds': time.time() - start_time
            },
            'dataset_statistics': {
                'test_subreddits': len(test_subreddits),
                'val_subreddits': len(val_subreddits),
                'train_subreddits': len(train_subreddits),
                'test_pairs_per_subreddit': TEST_PAIRS_PER_SUBREDDIT,
                'val_ratio': VAL_SPLIT_RATIO,
                'train_ratio': TRAIN_SPLIT_RATIO,
                'total_comments_overall': overall_comments,
                'total_threads_overall': overall_threads,
            },
            'output_files': output_files
        }

        stats_file = os.path.join(PATHS['data'], 'stage9_final_datasets_stats.json')
        write_json_file(summary_stats, stats_file, pretty=True)
        logger.info(f"📊 Statistics saved to: {stats_file}")

        elapsed = time.time() - start_time

        logger.info(f"🎉 Stage 9 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Final dataset splits:")
        for split_name, files in output_files.items():
            logger.info(f"  {split_name}:")
            logger.info(f"    Hydrated:   {files['hydrated']['path']} ({files['hydrated']['size_mb']:.1f} MB)")
            logger.info(f"    Dehydrated: {files['dehydrated']['path']} ({files['dehydrated']['size_mb']:.1f} MB)")

        log_stage_end(logger, 9, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 9 execution")
        log_stage_end(logger, 9, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
