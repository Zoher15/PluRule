#!/usr/bin/env python3
"""
Stage 6: Build Comment Trees and Discussion Threads

Integrated stage that builds comment trees and creates moderated/unmoderated
discussion thread pairs. Eliminates intermediate file I/O by keeping trees
in memory for immediate thread creation.

Input:
- organized_comments/{subreddit}_submission_comments.pkl (from Stage 5)
- matched_comments/{subreddit}_match.jsonl.zst (from Stage 4)
- data/top_{N}_sfw_subreddits.json (for rule sets)

Output:
- comment_trees/{subreddit}_comment_trees.pkl
- discussion_threads/{subreddit}_discussion_threads.pkl
"""

import sys
import os
import time
import pickle
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, MIN_MATCHED_COMMENTS, MIN_EVAL_THREAD_PAIRS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, ensure_directory, get_file_size_gb)
from utils.reddit import (normalize_subreddit_name, extract_submission_id, extract_comment_id, is_moderator_comment)
from utils.stats import calculate_jsd_from_uniform, rank_by_score


def load_target_subreddits_and_rules(logger) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load subreddits with their languages and complete rule sets."""
    subreddits_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')

    try:
        data = read_json_file(subreddits_file)

        subreddit_languages = {}
        subreddit_rules = {}

        for item in data.get('subreddits', []):
            subreddit_data = item.get('subreddit', {})
            subreddit_name = normalize_subreddit_name(subreddit_data.get('display_name', ''))
            language = subreddit_data.get('lang', '')

            subreddit_languages[subreddit_name] = language

            # Initialize all rules for this subreddit with 0 counts
            # Use short_name_clean to match Stage 4's matched_rule format
            # Rules are at item level, not subreddit_data level
            rule_stats = {}
            for rule in item.get('rules', []):
                rule_name = rule.get('short_name_clean', '')
                if rule_name:
                    rule_stats[rule_name] = 0

            subreddit_rules[subreddit_name] = rule_stats

        logger.info(f"Loaded {len(subreddit_languages)} subreddits with complete rule sets")
        return subreddit_languages, subreddit_rules

    except Exception as e:
        logger.error(f"Error loading subreddits and rules: {e}")
        return {}, {}


def calculate_depth_levels(root_comments: List[str], children_map: Dict[str, List[str]]) -> Dict[int, List[str]]:
    """Calculate depth levels for comments using BFS."""
    depth_levels = {}

    if not root_comments:
        return depth_levels

    # Initialize depth 0 with root comments
    depth_levels[0] = root_comments[:]
    comment_depths = {comment_id: 0 for comment_id in root_comments}

    # BFS to assign depths using deque for O(1) popleft
    queue = deque((comment_id, 0) for comment_id in root_comments)

    while queue:
        current_comment, current_depth = queue.popleft()

        # Get children of current comment
        child_ids = children_map.get(current_comment, [])

        if child_ids:
            next_depth = current_depth + 1

            # Initialize depth level if not exists
            if next_depth not in depth_levels:
                depth_levels[next_depth] = []

            # Add children to next depth level
            for child_id in child_ids:
                if child_id not in comment_depths:  # Avoid cycles
                    depth_levels[next_depth].append(child_id)
                    comment_depths[child_id] = next_depth
                    queue.append((child_id, next_depth))

    return depth_levels


def build_submission_tree(submission_id: str, comments: Dict[str, Dict]) -> Dict:
    """Build tree structure for a single submission."""
    children_map = {}  # parent_id -> [child_ids]
    parent_map = {}  # child_id -> parent_id (clean IDs)
    root_comments = []  # comments that reply directly to submission

    # Build parent->children mapping
    for comment_id, comment_data in comments.items():
        parent_id = comment_data.get('parent_id', '')

        if not parent_id:
            continue

        # Clean parent_id using utility functions consistently
        if parent_id.startswith('t3_'):
            # Direct reply to submission
            clean_parent_id = extract_submission_id(parent_id)
            root_comments.append(comment_id)
        else:
            # Reply to another comment (t1_ prefix or already clean)
            clean_parent_id = extract_comment_id(parent_id)

        # Initialize parent in children_map if not exists
        if clean_parent_id not in children_map:
            children_map[clean_parent_id] = []

        # Add this comment as child of its parent
        children_map[clean_parent_id].append(comment_id)

        # Add to parent mapping (child -> parent)
        parent_map[comment_id] = clean_parent_id

    # Calculate depth levels using BFS
    depth_levels = calculate_depth_levels(root_comments, children_map)

    return {
        'children_map': children_map,
        'parent_map': parent_map,
        'root_comments': root_comments,
        'depth_levels': depth_levels,
        'total_comments': len(comments)
    }


def build_subreddit_trees(submission_comments: Dict[str, Dict]) -> Dict[str, Any]:
    """Build trees for all submissions in a subreddit's data."""
    trees = {}
    total_comments = 0

    for submission_id, comments in submission_comments.items():
        if not comments:  # Skip empty submissions
            continue

        tree = build_submission_tree(submission_id, comments)
        trees[submission_id] = tree
        total_comments += tree['total_comments']

    return {
        'trees': trees,
        'metadata': {
            'total_submissions': len(trees),
            'total_comments': total_comments
        }
    }


def load_mod_comments(match_file_path: str, logger) -> List[Dict]:
    """Load moderator comments from match file (already sampled in Stage 4)."""
    try:
        lines = read_zst_lines(match_file_path)
        mod_comments = []

        for line in lines:
            if line.strip():
                mod_comment = json_loads(line)
                mod_comments.append(mod_comment)

        logger.info(f"Loaded {len(mod_comments)} moderator comments from {os.path.basename(match_file_path)}")
        return mod_comments

    except Exception as e:
        logger.error(f"Failed to load match file {match_file_path}: {e}")
        return []


def build_thread_and_check_mod_response(comment_id: str, comments: Dict[str, Dict],
                                        tree: Dict[str, Any], mode: str = 'unmoderated') -> Tuple[List[Dict], bool, Optional[str]]:
    """
    Build thread from comment to root and check for moderator responses and media while traversing.

    Two modes for checking moderator responses:

    MODERATED MODE ('moderated'):
    - Used when building a thread that we EXPECT to have a mod response at the leaf
    - The leaf comment (comment_id) is the parent of a moderator comment
    - We check all comments in the thread for mod responses EXCEPT:
      * We skip checking if the LEAF comment has mod children (because it should - that's our mod action)
    - This ensures we reject threads that have OTHER mod responses besides the expected one

    UNMODERATED MODE ('unmoderated'):
    - Used when building an alternative thread that should be clean
    - We check EVERY comment in the thread, including the leaf
    - We check if ANY comment (including the starting one) has mod responses
    - We want to ensure this thread is completely free of moderation

    Both modes check:
    1. If each comment itself is a moderator comment
    2. If each comment has media content (embedded images/videos)
    3. If each comment has moderator responses as direct children (1 level)

    Args:
        comment_id: The leaf comment ID to build thread from
        comments: Dictionary of all comments for this submission
        tree: Tree structure with children mapping for this submission
        mode: Either 'moderated' or 'unmoderated' to control checking behavior

    Returns:
        - thread: List of comment dicts from root to leaf with 'level' field
        - has_issue: True if any moderator response or media found (based on mode rules)
        - issue_type: 'mod_response', 'has_media', or None
    """
    # Build path from leaf to root, checking for mod responses as we go
    path = []
    current_id = comment_id
    children_map = tree.get('children_map', {})

    # Track position while building path upward
    position = 0

    while current_id in comments:
        comment = comments[current_id]
        path.append(current_id)

        # Check if THIS comment is a moderator comment
        if is_moderator_comment(comment):
            # Early exit - no need to build full thread
            return [], True, 'mod_response'

        # Check if THIS comment contains media (embedded images/videos)
        if comment.get('media') or comment.get('media_metadata'):
            # Early exit - skip threads with media comments
            return [], True, 'has_media'

        # Check if THIS comment has moderator responses as direct children
        # MODERATED MODE: Skip this check for the leaf comment (position 0) because
        #                 that's where we expect the mod response to be
        # UNMODERATED MODE: Check all positions including the leaf
        should_check_children = (mode == 'unmoderated') or (position > 0)

        if should_check_children:
            child_ids = children_map.get(current_id, [])
            for child_id in child_ids:
                child_comment = comments.get(child_id, {})
                if is_moderator_comment(child_comment):
                    # Early exit - no need to build full thread
                    return [], True, 'mod_response'

        # Move to parent
        parent_id = comment.get('parent_id', '')

        # Stop if we reach submission
        if parent_id.startswith('t3_'):
            break

        # Clean parent_id using utility function (handles t1_ prefix or already clean)
        parent_id_clean = extract_comment_id(parent_id)
        current_id = parent_id_clean
        position += 1

    # No issues found, build the final thread in root-to-leaf order
    path.reverse()
    thread = []
    for level, thread_comment_id in enumerate(path):
        comment = comments[thread_comment_id].copy()
        comment['level'] = level
        thread.append(comment)

    return thread, False, None


def count_common_ancestors_from_threads(thread1: List[Dict], thread2: List[Dict]) -> int:
    """
    Count common ancestors between two threads (already built).

    Threads are lists of comment dicts ordered from root to leaf.
    We compare the comment IDs to find the common prefix.

    Args:
        thread1: First thread (root to leaf)
        thread2: Second thread (root to leaf)

    Returns:
        Number of common ancestors (comments that appear in the same position in both paths)
    """
    common_count = 0
    for i in range(min(len(thread1), len(thread2))):
        # Compare comment IDs
        if thread1[i].get('id') == thread2[i].get('id'):
            common_count += 1
        else:
            break

    return common_count


def find_best_alternative(moderated_comment_id: str, moderated_thread: List[Dict],
                         submission_id: str, comments: Dict[str, Dict],
                         trees: Dict[str, Any], logger) -> Tuple[Optional[str], Optional[List[Dict]], Optional[int]]:
    """Find best alternative comment for unmoderated thread.

    Args:
        moderated_comment_id: ID of the moderated comment
        moderated_thread: Already-built thread for moderated comment (to avoid rebuilding)
        submission_id: ID of the submission
        comments: All comments for this submission
        trees: Tree structures
        logger: Logger instance

    Returns:
        Tuple of (alt_comment_id, alt_thread, moderated_depth)
        - alt_comment_id: ID of the best alternative comment
        - alt_thread: The built thread for the alternative (to avoid rebuilding)
        - moderated_depth: Depth of the moderated comment (for failure tracking)
    """
    if submission_id not in trees['trees']:
        return None, None, None

    tree = trees['trees'][submission_id]
    depth_levels = tree.get('depth_levels', {})

    # Find moderated comment's depth
    moderated_depth = None
    for depth, comment_ids in depth_levels.items():
        if moderated_comment_id in comment_ids:
            moderated_depth = depth
            break

    if moderated_depth is None:
        return None, None, None

    # Get all comments at same depth
    same_depth_comments = depth_levels.get(moderated_depth, [])

    # Remove moderated comment from alternatives and sort for determinism
    alternatives = sorted([cid for cid in same_depth_comments if cid != moderated_comment_id])

    if not alternatives:
        return None, None, moderated_depth

    # Find best alternative based on common ancestors and score
    best_alternative = None
    best_alternative_thread = None
    max_common_ancestors = -1
    lowest_score = float('inf')

    for alt_id in alternatives:
        # Build thread and check for mod responses/media in one pass (UNMODERATED mode)
        # This checks the entire thread including the leaf for any mod responses or media
        alt_thread, has_issue, issue_type = build_thread_and_check_mod_response(
            alt_id, comments, tree, mode='unmoderated'
        )

        # Skip if thread has mod responses or media
        if has_issue:
            continue

        # Check if thread length matches exactly
        # Comments at same depth should have same thread length if data is complete
        if len(alt_thread) != len(moderated_thread):
            logger.warning(f"Length mismatch for alternative {alt_id} at depth {moderated_depth}: expected {len(moderated_thread)}, got {len(alt_thread)}. This suggests missing parent comments in the data.")
            continue

        # Count common ancestors using already-built threads (no need to rebuild paths)
        common_ancestors = count_common_ancestors_from_threads(moderated_thread, alt_thread)

        # Get comment score
        score = comments.get(alt_id, {}).get('score', 0)

        # Select best: most common ancestors first, then lowest score
        if (common_ancestors > max_common_ancestors or
            (common_ancestors == max_common_ancestors and score < lowest_score)):
            best_alternative = alt_id
            best_alternative_thread = alt_thread  # Save the thread to avoid rebuilding
            max_common_ancestors = common_ancestors
            lowest_score = score

    return best_alternative, best_alternative_thread, moderated_depth


def build_thread_pair(mod_comment: Dict, comments: Dict[str, Dict],
                     trees: Dict[str, Any], logger) -> Tuple[Optional[Dict], Optional[int], Optional[str]]:
    """Build moderated and unmoderated thread pair for a moderator comment.

    Returns:
        Tuple of (pair_dict, moderated_depth, failure_reason)
        - pair_dict: The thread pair if successful, None otherwise
        - moderated_depth: Depth of moderated comment (for failure tracking)
        - failure_reason: String indicating why pair failed ('moderated_thread_has_mod_response' or None)
    """
    mod_comment_id = mod_comment.get('id')
    moderated_comment_id = extract_comment_id(mod_comment.get('parent_id', ''))
    submission_id = extract_submission_id(mod_comment.get('link_id', ''))

    if not moderated_comment_id or not submission_id:
        return None, None, None

    # Check if we have the data
    if submission_id not in trees.get('trees', {}):
        return None, None, None

    if moderated_comment_id not in comments:
        return None, None, None

    # Get the tree for this submission
    tree = trees['trees'][submission_id]

    # Build moderated thread and check for OTHER mod responses/media in one pass (MODERATED mode)
    # This skips checking the leaf's children since we expect a mod response there
    moderated_thread, has_issue, issue_type = build_thread_and_check_mod_response(
        moderated_comment_id, comments, tree, mode='moderated'
    )

    if not moderated_thread:
        return None, None, None

    # Reject if moderated thread has OTHER moderator responses or media besides the expected mod action
    if has_issue:
        if issue_type == 'has_media':
            logger.warning(f"Moderated thread contains media for {moderated_comment_id} in submission {submission_id}.")
            return None, None, 'moderated_thread_has_media'
        elif issue_type == 'mod_response':
            logger.warning(f"Moderated thread has other mod responses besides the expected one for {moderated_comment_id} in submission {submission_id}.")
            return None, None, 'moderated_thread_has_mod_response'

    # Find best alternative at exact same depth (no fallback)
    alt_comment_id, unmoderated_thread, moderated_depth = find_best_alternative(
        moderated_comment_id, moderated_thread,
        submission_id, comments, trees, logger
    )

    if not alt_comment_id:
        return None, moderated_depth, None

    # unmoderated_thread already built in find_best_alternative (no need to rebuild)
    # common_ancestors already calculated in find_best_alternative using the threads
    # Calculate it again here for the final pair metadata
    common_ancestors = count_common_ancestors_from_threads(moderated_thread, unmoderated_thread)

    return {
        'mod_comment_id': mod_comment_id,
        'mod_comment': mod_comment,  # Store full moderator comment object
        'moderated_thread': moderated_thread,
        'unmoderated_thread': unmoderated_thread,
        'metadata': {
            'common_ancestors': common_ancestors,
            'rule': mod_comment.get('matched_rule', {}).get('short_name_clean', ''),
            'rule_similarity_score': mod_comment.get('matched_rule', {}).get('similarity_score', 0),
            'moderated_comment_id': moderated_comment_id,
            'unmoderated_comment_id': alt_comment_id,
            'submission_id': submission_id,
            'moderated_score': comments.get(moderated_comment_id, {}).get('score', 0),
            'unmoderated_score': comments.get(alt_comment_id, {}).get('score', 0),
            'target_length': len(moderated_thread)
        }
    }, None, None


def process_subreddit(args: tuple) -> Dict[str, Any]:
    """Process single subreddit: build trees and create discussion threads."""
    subreddit_name, complete_rule_set = args

    # Create worker logger with subreddit identifier in subdirectory
    worker_logger = get_stage_logger(6, "build_trees_and_threads", worker_identifier=f"subreddits/{subreddit_name}")

    worker_logger.info(f"🔄 Processing {subreddit_name}")
    start_time = time.time()

    try:
        # Define file paths
        submission_comments_file = os.path.join(PATHS['organized_comments'], f"{subreddit_name}_submission_comments.pkl")
        match_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")

        trees_output = os.path.join(PATHS['comment_trees'], f"{subreddit_name}_comment_trees.pkl")
        threads_output = os.path.join(PATHS['discussion_threads'], f"{subreddit_name}_discussion_threads.pkl")

        # Ensure both output directories exist
        ensure_directory(trees_output)
        ensure_directory(threads_output)

        # Check if input files exist
        for file_path in [submission_comments_file, match_file]:
            if not os.path.exists(file_path):
                return {
                    'subreddit': subreddit_name,
                    'status': 'failed',
                    'error': f"Missing file: {file_path}",
                    'trees_built': 0,
                    'successful_pairs': 0,
                    'total_mod_comments': 0
                }

        # Load submission comments
        with open(submission_comments_file, 'rb') as f:
            submission_comments = pickle.load(f)

        if not submission_comments:
            return {
                'subreddit': subreddit_name,
                'status': 'failed',
                'error': "No submission comments found",
                'trees_built': 0,
                'successful_pairs': 0,
                'total_mod_comments': 0
            }

        # Build comment trees (in memory)
        worker_logger.info(f"  🌳 Building trees for {len(submission_comments)} submissions")
        trees_data = build_subreddit_trees(submission_comments)
        trees_data['subreddit'] = subreddit_name
        trees_data['source_file'] = submission_comments_file

        # Save trees to disk
        with open(trees_output, 'wb') as f:
            pickle.dump(trees_data, f)

        trees_built = len(trees_data['trees'])
        total_comments = trees_data['metadata']['total_comments']
        trees_file_size = get_file_size_gb(trees_output)

        worker_logger.info(f"  ✅ Built {trees_built} trees with {total_comments:,} comments ({trees_file_size:.2f} GB)")

        # Load moderator comments
        mod_comments = load_mod_comments(match_file, worker_logger)

        if not mod_comments:
            return {
                'subreddit': subreddit_name,
                'status': 'completed',
                'trees_built': trees_built,
                'successful_pairs': 0,
                'total_mod_comments': 0,
                'rule_distribution': {},
                'jsd_from_uniform': 1.0
            }

        # Build discussion thread pairs
        worker_logger.info(f"  🧵 Building discussion threads from {len(mod_comments)} mod comments")

        thread_pairs = []
        success_count = 0
        rule_stats = complete_rule_set.copy()  # Initialize with complete rule set
        debug_counts = {
            'missing_submission': 0,
            'missing_parent_id': 0,
            'missing_moderated_comment': 0,
            'moderated_thread_has_mod_response': 0,
            'moderated_thread_has_media': 0,
            'no_alternative_found': 0
        }
        failed_depth_distribution = defaultdict(int)
        successful_depth_distribution = defaultdict(int)

        for mod_comment in mod_comments:
            # Extract submission ID using utility
            submission_id = extract_submission_id(mod_comment.get('link_id', ''))

            if submission_id not in submission_comments:
                debug_counts['missing_submission'] += 1
                continue

            comments = submission_comments[submission_id]

            # Check for missing parent_id
            moderated_comment_id = extract_comment_id(mod_comment.get('parent_id', ''))
            if not moderated_comment_id:
                debug_counts['missing_parent_id'] += 1
                continue

            # Check if moderated comment exists
            if moderated_comment_id not in comments:
                debug_counts['missing_moderated_comment'] += 1
                continue

            # Build thread pair
            pair, failed_depth, failure_reason = build_thread_pair(mod_comment, comments, trees_data, worker_logger)
            if pair:
                thread_pairs.append(pair)
                success_count += 1

                # Track successful depth distribution
                moderated_thread = pair.get('moderated_thread', [])
                if moderated_thread:
                    successful_depth = len(moderated_thread)
                    successful_depth_distribution[successful_depth] += 1

                # Track rule statistics
                matched_rule = mod_comment.get('matched_rule', {})
                # Use short_name_clean to match initialization in load_target_subreddits_and_rules
                rule = matched_rule.get('short_name_clean', 'unknown')
                rule_stats[rule] = rule_stats.get(rule, 0) + 1
            else:
                # Track specific failure reason if provided
                if failure_reason:
                    debug_counts[failure_reason] = debug_counts.get(failure_reason, 0) + 1
                else:
                    debug_counts['no_alternative_found'] += 1

                # Track failed depth distribution
                if failed_depth is not None:
                    failed_depth_distribution[failed_depth] += 1

        # Save discussion threads
        result_data = {
            'subreddit': subreddit_name,
            'thread_pairs': thread_pairs,
            'metadata': {
                'total_mod_comments': len(mod_comments),
                'successful_pairs': success_count,
                'success_rate': success_count / len(mod_comments) if mod_comments else 0
            }
        }

        with open(threads_output, 'wb') as f:
            pickle.dump(result_data, f)

        elapsed = time.time() - start_time
        success_rate = (success_count / len(mod_comments)) * 100 if mod_comments else 0
        threads_file_size = get_file_size_gb(threads_output)

        worker_logger.info(f"  🎉 {subreddit_name}: {trees_built} trees, {success_count}/{len(mod_comments)} thread pairs ({success_rate:.1f}%) in {elapsed:.1f}s")

        # Calculate JSD from uniform distribution
        jsd_score = calculate_jsd_from_uniform(rule_stats)

        return {
            'subreddit': subreddit_name,
            'status': 'completed',
            'trees_built': trees_built,
            'total_comments': total_comments,
            'trees_file_size_gb': trees_file_size,
            'successful_pairs': success_count,
            'total_mod_comments': len(mod_comments),
            'success_rate': success_count / len(mod_comments) if mod_comments else 0,
            'threads_file_size_gb': threads_file_size,
            'rule_distribution': rule_stats,
            'jsd_from_uniform': jsd_score,
            'debug_counts': debug_counts,
            'failed_depth_distribution': {str(k): v for k, v in failed_depth_distribution.items()},
            'successful_depth_distribution': {str(k): v for k, v in successful_depth_distribution.items()},
            'processing_time': elapsed
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing {subreddit_name}: {e}")
        return {
            'subreddit': subreddit_name,
            'status': 'failed',
            'error': str(e),
            'trees_built': 0,
            'successful_pairs': 0,
            'total_mod_comments': 0,
            'processing_time': 0
        }


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(6, "build_trees_and_threads")
    log_stage_start(logger, 6, "Build Comment Trees and Discussion Threads")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Load subreddits and rule sets
        logger.info("📚 Loading target subreddits and rule sets...")
        subreddit_languages, subreddit_rules = load_target_subreddits_and_rules(logger)

        if not subreddit_languages:
            logger.error("❌ No subreddits loaded!")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Find subreddits with required input files
        subreddits_to_process = []

        for subreddit in subreddit_languages.keys():
            submission_comments_file = os.path.join(PATHS['organized_comments'], f"{subreddit}_submission_comments.pkl")
            match_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_match.jsonl.zst")

            if os.path.exists(submission_comments_file) and os.path.exists(match_file):
                complete_rule_set = subreddit_rules.get(subreddit, {})
                subreddits_to_process.append((subreddit, complete_rule_set))

        if not subreddits_to_process:
            logger.error("❌ No subreddits found with required input files!")
            log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Found {len(subreddits_to_process)} subreddits to process")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Process subreddits in parallel
        logger.info("🌳 Processing subreddits to build trees and discussion threads...")
        results = process_files_parallel(subreddits_to_process, process_subreddit, PROCESSES, logger)

        # Collect statistics
        completed_results = [r for r in results if r.get('status') == 'completed']
        failed_results = [r for r in results if r.get('status') == 'failed']

        total_trees = sum(r.get('trees_built', 0) for r in completed_results)
        total_comments = sum(r.get('total_comments', 0) for r in completed_results)
        total_successful_pairs = sum(r.get('successful_pairs', 0) for r in completed_results)
        total_mod_comments = sum(r.get('total_mod_comments', 0) for r in completed_results)
        total_trees_size = sum(r.get('trees_file_size_gb', 0) for r in completed_results)
        total_threads_size = sum(r.get('threads_file_size_gb', 0) for r in completed_results)

        # Filter to subreddits with ≥MIN_EVAL_THREAD_PAIRS successful pairs (keep ALL, no top-N filtering)
        qualified_results = [r for r in completed_results if r.get('successful_pairs', 0) >= MIN_EVAL_THREAD_PAIRS]

        # Add language information to all qualified results
        for result in qualified_results:
            subreddit_name = result['subreddit']
            result['language'] = subreddit_languages.get(subreddit_name, 'unknown')

        # Separate English from other languages for ranking
        english_results = [r for r in qualified_results if r.get('language') == 'en']
        other_language_results = [r for r in qualified_results if r.get('language') != 'en']

        # Rank ALL English subreddits by JSD (ascending=True means lower JSD = better rank)
        english_results = rank_by_score(english_results, 'jsd_from_uniform', ascending=True)

        # Other languages get no rank
        for result in other_language_results:
            result['rank'] = None

        # Create summary statistics
        elapsed = time.time() - start_time

        summary = {
            'summary': {
                'total_subreddits_processed': len(subreddits_to_process),
                'completed_subreddits': len(completed_results),
                'failed_subreddits': len(failed_results),
                'subreddits_with_500_plus_pairs': len(qualified_results),
                'english_subreddits': len(english_results),
                'other_language_subreddits': len(other_language_results),
                'total_trees_built': total_trees,
                'total_comments_processed': total_comments,
                'total_successful_pairs': total_successful_pairs,
                'total_mod_comments': total_mod_comments,
                'overall_success_rate': total_successful_pairs / total_mod_comments if total_mod_comments > 0 else 0,
                'total_trees_size_gb': total_trees_size,
                'total_threads_size_gb': total_threads_size,
                'processing_time_seconds': elapsed,
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'subreddit_stats': english_results + other_language_results,
            'failed_subreddits': [{'subreddit': r['subreddit'], 'error': r.get('error', 'Unknown')} for r in failed_results]
        }

        # Save summary
        summary_file = os.path.join(PATHS['data'], 'stage6_trees_and_threads_summary.json')
        write_json_file(summary, summary_file, pretty=True)

        logger.info(f"🎉 Stage 6 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Processed {len(completed_results)}/{len(subreddits_to_process)} subreddits")
        logger.info(f"🌳 Built {total_trees:,} comment trees")
        logger.info(f"🧵 Created {total_successful_pairs:,} discussion thread pairs")
        logger.info(f"📈 Overall success rate: {total_successful_pairs/total_mod_comments*100:.1f}%" if total_mod_comments > 0 else "📈 No mod comments processed")
        logger.info(f"✅ Qualified subreddits (≥{MIN_EVAL_THREAD_PAIRS} pairs): {len(qualified_results)}")
        logger.info(f"   🏆 English subreddits: {len(english_results)}")
        logger.info(f"   🌍 Other language subreddits: {len(other_language_results)}")
        logger.info(f"Summary saved to: {summary_file}")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_results)}):")
            for result in failed_results[:10]:  # Show first 10
                logger.warning(f"  {result['subreddit']}: {result.get('error', 'Unknown error')}")
            if len(failed_results) > 10:
                logger.warning(f"  ... and {len(failed_results) - 10} more")

        log_stage_end(logger, 6, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 6 execution")
        log_stage_end(logger, 6, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())