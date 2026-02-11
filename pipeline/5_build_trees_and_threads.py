#!/usr/bin/env python3
"""
Stage 5: Build Comment Trees and Discussion Threads

Builds hierarchical comment tree structures and creates violating/compliant
discussion thread pairs for training data.

Input:
- organized_comments/{subreddit}/submission_{id}.pkl (from Stage 4)
- matched_comments/{subreddit}_match.jsonl.zst (from Stage 3)
- data/stage2_sfw_subreddits_min_{N}_comments.json (for rule sets)

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, MIN_MATCHED_COMMENTS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, ensure_directory, get_file_size_gb)
from utils.reddit import (normalize_subreddit_name, extract_submission_id, extract_comment_id, is_moderator_comment)
from utils.stats import calculate_jsd_from_uniform, rank_by_score


# ============================================================================
# CONFIGURATION & DATA LOADING
# ============================================================================

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
            subreddit_languages[subreddit_name] = subreddit_data.get('lang', '')

            # Initialize all rules with 0 counts (use short_name_clean to match Stage 4)
            subreddit_rules[subreddit_name] = {
                rule.get('short_name_clean', ''): 0
                for rule in item.get('rules', [])
                if rule.get('short_name_clean', '')
            }

        logger.info(f"Loaded {len(subreddit_languages)} subreddits with complete rule sets")
        return subreddit_languages, subreddit_rules

    except Exception as e:
        logger.error(f"Error loading subreddits and rules: {e}")
        return {}, {}


def load_submission_comments(submission_comments_dir: str, submission_id: str, logger) -> Optional[Dict]:
    """Load comments for a single submission from its pickle file."""
    filepath = os.path.join(submission_comments_dir, f"submission_{submission_id}.pkl")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load submission_{submission_id}.pkl: {e}")
        return None


def load_mod_comments(match_file_path: str, logger) -> List[Dict]:
    """Load moderator comments from match file (already sampled in Stage 4)."""
    try:
        lines = read_zst_lines(match_file_path)
        mod_comments = [json_loads(line) for line in lines if line.strip()]
        logger.info(f"Loaded {len(mod_comments)} moderator comments from {os.path.basename(match_file_path)}")
        return mod_comments
    except Exception as e:
        logger.error(f"Failed to load match file {match_file_path}: {e}")
        return []


# ============================================================================
# TREE BUILDING
# ============================================================================

def calculate_depth_levels(root_comments: List[str], children_map: Dict[str, List[str]]) -> Dict[int, List[str]]:
    """Calculate depth levels for comments using BFS."""
    if not root_comments:
        return {}

    depth_levels = {0: root_comments[:]}
    comment_depths = {comment_id: 0 for comment_id in root_comments}
    queue = deque((comment_id, 0) for comment_id in root_comments)

    while queue:
        current_comment, current_depth = queue.popleft()
        child_ids = children_map.get(current_comment, [])

        if child_ids:
            next_depth = current_depth + 1
            if next_depth not in depth_levels:
                depth_levels[next_depth] = []

            for child_id in child_ids:
                if child_id not in comment_depths:  # Avoid cycles
                    depth_levels[next_depth].append(child_id)
                    comment_depths[child_id] = next_depth
                    queue.append((child_id, next_depth))

    return depth_levels


def build_submission_tree(comments: Dict[str, Dict]) -> Dict:
    """Build tree structure for a single submission."""
    children_map = {}  # parent_id -> [child_ids]
    parent_map = {}    # child_id -> parent_id
    root_comments = []

    for comment_id, comment_data in comments.items():
        parent_id = comment_data.get('parent_id', '')
        if not parent_id or isinstance(parent_id, int):  # Skip ambiguous integer parent_ids
            continue

        # Determine if direct reply to submission or comment
        if parent_id.startswith('t3_'):
            clean_parent_id = extract_submission_id(parent_id)
            root_comments.append(comment_id)
        else:
            clean_parent_id = extract_comment_id(parent_id)

        children_map.setdefault(clean_parent_id, []).append(comment_id)
        parent_map[comment_id] = clean_parent_id

    return {
        'children_map': children_map,
        'parent_map': parent_map,
        'root_comments': root_comments,
        'depth_levels': calculate_depth_levels(root_comments, children_map),
        'total_comments': len(comments)
    }


def build_subreddit_trees(submission_comments_dir: str, subreddit_name: str, logger) -> Dict[str, Any]:
    """Build trees for all submissions in a subreddit by lazy loading from directory."""
    if not os.path.exists(submission_comments_dir) or not os.path.isdir(submission_comments_dir):
        logger.warning(f"Directory not found: {submission_comments_dir}")
        return {
            'trees': {},
            'metadata': {'total_submissions': 0, 'total_comments': 0},
            'subreddit': subreddit_name,
            'source_dir': submission_comments_dir
        }

    submission_files = [f for f in os.listdir(submission_comments_dir)
                       if f.startswith('submission_') and f.endswith('.pkl')]

    logger.info(f"  🌳 Building trees for {len(submission_files)} submissions (parallel)")

    def process_submission_tree(filename):
        """Process a single submission tree."""
        submission_id = filename[11:-4]  # Extract from "submission_{id}.pkl"
        comments = load_submission_comments(submission_comments_dir, submission_id, logger)
        if comments:
            return submission_id, build_submission_tree(comments)
        return None, None

    trees = {}
    total_comments = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_submission_tree, f): f for f in submission_files}
        for future in as_completed(futures):
            submission_id, tree = future.result()
            if tree:
                trees[submission_id] = tree
                total_comments += tree['total_comments']

    return {
        'trees': trees,
        'metadata': {'total_submissions': len(trees), 'total_comments': total_comments},
        'subreddit': subreddit_name,
        'source_dir': submission_comments_dir
    }


# ============================================================================
# THREAD BUILDING
# ============================================================================

def build_thread_and_check_mod_response(comment_id: str, comments: Dict[str, Dict],
                                        tree: Dict[str, Any], mode: str = 'compliant') -> Tuple[List[Dict], bool, Optional[str]]:
    """
    Build thread from comment to root and check for issues.

    Modes:
    - 'violating': Expects mod response at leaf, skips checking leaf's children
    - 'compliant': Checks that leaf has NO mod responses as direct children

    Both modes check:
    1. If ANY comment (including all ancestors) has been removed/deleted body OR author
    2. If any comment has media content
    3. If ANY comment in the thread is by a moderator (we don't want mod-user back-and-forth)

    Mode-specific checks:
    4. VIOLATING only: If leaf comment has been edited (edited != False)
    5. COMPLIANT only: If leaf comment has moderator responses as direct children

    Returns:
        (thread, has_issue, issue_type) where issue_type is 'mod_response', 'removed_or_deleted', 'has_media', 'moderator_in_thread', 'edited_comment', or None
    """
    path = []
    current_id = comment_id
    children_map = tree.get('children_map', {})
    position = 0
    visited = set()  # Prevent infinite loops from circular references

    while current_id in comments and current_id not in visited:
        visited.add(current_id)
        comment = comments[current_id]
        path.append(current_id)

        # Check if ANY comment has been removed/deleted (body OR author)
        body = comment.get('body', '')
        author = comment.get('author', '')
        if body in ['[removed]', '[deleted]'] or author in ['[deleted]', '[removed]']:
            return [], True, 'removed_or_deleted'

        # Check if ANY comment contains media
        if comment.get('media') or comment.get('media_metadata'):
            return [], True, 'has_media'

        # Check if ANY comment in the thread is by a moderator
        if is_moderator_comment(comment):
            return [], True, 'moderator_in_thread'

        # Check if LEAF comment has been edited (violating mode only)
        if position == 0 and mode == 'violating':
            edited = comment.get('edited', False)
            if edited is not False:
                return [], True, 'edited_comment'

        # Check if LEAF comment has moderator responses (mode-dependent)
        if position == 0 and mode == 'compliant':
            for child_id in children_map.get(current_id, []):
                if is_moderator_comment(comments.get(child_id, {})):
                    return [], True, 'mod_response'

        # Move to parent
        parent_id = comment.get('parent_id', '')
        if parent_id.startswith('t3_'):
            break

        current_id = extract_comment_id(parent_id)
        position += 1

    # Build final thread in root-to-leaf order
    path.reverse()
    thread = []
    for level, thread_comment_id in enumerate(path):
        comment = comments[thread_comment_id].copy()
        comment['level'] = level
        thread.append(comment)

    return thread, False, None


def count_common_ancestors(thread1: List[Dict], thread2: List[Dict]) -> int:
    """Count common ancestors between two threads (already built, root to leaf)."""
    common_count = 0
    for i in range(min(len(thread1), len(thread2))):
        if thread1[i].get('id') == thread2[i].get('id'):
            common_count += 1
        else:
            break
    return common_count


def find_best_alternative(violating_comment_id: str, violating_thread: List[Dict],
                         submission_id: str, comments: Dict[str, Dict],
                         trees: Dict[str, Any], used_alternatives: set) -> Tuple[Optional[str], Optional[List[Dict]], Optional[int], Optional[Dict]]:
    """
    Find best alternative comment for compliant thread at same depth or depth-1.

    Searches at:
    1. Same depth as violating comment
    2. Depth-1 (if violating_depth >= 2 and alternative is not direct parent of violating comment)

    Returns: (alt_comment_id, alt_thread, violating_depth, rejection_stats)
    """
    rejection_stats = {
        'total_alternatives_at_depth': 0,
        'total_alternatives_at_depth_minus_1': 0,
        'rejected_mod_response': 0,
        'rejected_removed_or_deleted': 0,
        'rejected_has_media': 0,
        'rejected_moderator_in_thread': 0,
        'rejected_length_mismatch': 0,
        'rejected_is_parent': 0
    }

    if submission_id not in trees['trees']:
        return None, None, None, rejection_stats

    tree = trees['trees'][submission_id]
    depth_levels = tree.get('depth_levels', {})
    parent_map = tree.get('parent_map', {})

    # Find violating comment's depth
    violating_depth = None
    for depth, comment_ids in depth_levels.items():
        if violating_comment_id in comment_ids:
            violating_depth = depth
            break

    if violating_depth is None:
        return None, None, None, rejection_stats

    # Get direct parent of violating comment
    violating_parent_id = parent_map.get(violating_comment_id)

    # Collect alternatives from same depth
    same_depth_comments = depth_levels.get(violating_depth, [])
    alternatives = [cid for cid in same_depth_comments
                   if cid != violating_comment_id and cid not in used_alternatives]
    rejection_stats['total_alternatives_at_depth'] = len(alternatives)

    # Collect alternatives from depth-1 (if depth >= 2)
    if violating_depth >= 2:
        depth_minus_1_comments = depth_levels.get(violating_depth - 1, [])
        for cid in depth_minus_1_comments:
            # Exclude if it's the direct parent of violating comment or already used
            if cid != violating_parent_id and cid not in used_alternatives:
                alternatives.append(cid)
        rejection_stats['total_alternatives_at_depth_minus_1'] = len([cid for cid in depth_minus_1_comments
                                                                      if cid != violating_parent_id and cid not in used_alternatives])

    # Sort for consistent processing
    alternatives = sorted(alternatives)

    if not alternatives:
        return None, None, violating_depth, rejection_stats

    # Find best alternative using priority: common ancestors (more better), length (longer better), score (lower better)
    best_alternative = None
    best_alternative_thread = None
    best_score_tuple = None  # (common_ancestors, length, -score) - all higher is better

    for alt_id in alternatives:
        alt_thread, has_issue, issue_type = build_thread_and_check_mod_response(
            alt_id, comments, tree, mode='compliant'
        )

        if has_issue:
            rejection_stats[f'rejected_{issue_type}'] = rejection_stats.get(f'rejected_{issue_type}', 0) + 1
            continue

        # Calculate ranking criteria
        # Priority: 1) common ancestors (more better), 2) length (longer better), 3) score (lower better)
        thread_length = len(alt_thread)
        common_ancestors = count_common_ancestors(violating_thread, alt_thread)
        score = comments.get(alt_id, {}).get('score', 0)

        # Create tuple for comparison (all values "higher is better")
        score_tuple = (common_ancestors, thread_length, -score)

        # Update best if this is better (tuple comparison does lexicographic ordering)
        if best_score_tuple is None or score_tuple > best_score_tuple:
            best_alternative = alt_id
            best_alternative_thread = alt_thread
            best_score_tuple = score_tuple

    return best_alternative, best_alternative_thread, violating_depth, rejection_stats


def build_thread_pair(mod_comment: Dict, comments: Dict[str, Dict],
                     trees: Dict[str, Any], logger, used_alternatives: set) -> Tuple[Optional[Dict], Optional[int], Optional[str], Optional[Dict]]:
    """
    Build violating and compliant thread pair for a moderator comment.

    Returns: (pair_dict, violating_depth, failure_reason, rejection_stats)
    """
    mod_comment_id = mod_comment.get('id')
    violating_comment_id = extract_comment_id(mod_comment.get('parent_id', ''))
    submission_id = extract_submission_id(mod_comment.get('link_id', ''))

    if not violating_comment_id or not submission_id:
        return None, None, None, None

    if submission_id not in trees.get('trees', {}) or violating_comment_id not in comments:
        return None, None, None, None

    tree = trees['trees'][submission_id]

    # Build violating thread (mode='violating' skips checking leaf's children)
    violating_thread, has_issue, issue_type = build_thread_and_check_mod_response(
        violating_comment_id, comments, tree, mode='violating'
    )

    if not violating_thread or has_issue:
        return None, None, f'violating_thread_has_{issue_type}' if issue_type else None, None

    # Find best alternative at same depth or depth-1
    alt_comment_id, compliant_thread, violating_depth, rejection_stats = find_best_alternative(
        violating_comment_id, violating_thread,
        submission_id, comments, trees, used_alternatives
    )

    if not alt_comment_id:
        return None, violating_depth, None, rejection_stats

    common_ancestors = count_common_ancestors(violating_thread, compliant_thread)

    return {
        'mod_comment_id': mod_comment_id,
        'mod_comment': mod_comment,
        'violating_thread': violating_thread,
        'compliant_thread': compliant_thread,
        'metadata': {
            'common_ancestors': common_ancestors,
            'rule': mod_comment.get('matched_rule', {}).get('short_name_clean', ''),
            'rule_similarity_score': mod_comment.get('matched_rule', {}).get('similarity_score', 0),
            'violating_comment_id': violating_comment_id,
            'compliant_comment_id': alt_comment_id,
            'submission_id': submission_id,
            'violating_score': comments.get(violating_comment_id, {}).get('score', 0),
            'compliant_score': comments.get(alt_comment_id, {}).get('score', 0),
            'violating_depth': violating_depth,
            'violating_length': len(violating_thread),
            'compliant_length': len(compliant_thread)
        }
    }, None, None, rejection_stats


def build_discussion_threads(mod_comments: List[Dict], submission_comments_dir: str,
                            trees_data: Dict[str, Any], complete_rule_set: Dict[str, int],
                            logger) -> Dict[str, Any]:
    """Build discussion thread pairs from moderator comments using parallel processing by submission."""
    # Group mod comments by submission
    mod_comments_by_submission = defaultdict(list)
    for mod_comment in mod_comments:
        submission_id = extract_submission_id(mod_comment.get('link_id', ''))
        if submission_id:
            mod_comments_by_submission[submission_id].append(mod_comment)

    # Initialize tracking structures
    thread_pairs = []
    rule_stats = complete_rule_set.copy()

    debug_counts = {
        'missing_submission': 0,
        'missing_parent_id': 0,
        'missing_violating_comment': 0,
        'violating_thread_has_mod_response': 0,
        'violating_thread_has_media': 0,
        'violating_thread_has_removed_or_deleted': 0,
        'violating_thread_has_moderator_in_thread': 0,
        'violating_thread_has_edited_comment': 0,
        'no_alternative_found': 0
    }

    alternative_rejection_stats = {
        'total_alternatives_checked': 0,
        'total_alternatives_at_depth': 0,
        'total_alternatives_at_depth_minus_1': 0,
        'rejected_mod_response': 0,
        'rejected_removed_or_deleted': 0,
        'rejected_has_media': 0,
        'rejected_moderator_in_thread': 0,
        'rejected_is_parent': 0
    }

    failed_depth_distribution = defaultdict(int)
    successful_depth_distribution = defaultdict(int)

    def process_submission_threads(submission_id, submission_mod_comments):
        """Process all mod comments for a single submission."""
        # Lazy load submission comments once for all mod comments in this submission
        comments = load_submission_comments(submission_comments_dir, submission_id, logger)
        if not comments:
            return {
                'pairs': [],
                'debug_counts': {'missing_submission': len(submission_mod_comments)},
                'rejection_stats': {},
                'failed_depths': {},
                'successful_depths': {},
                'rules': {}
            }

        # Track used alternatives within this submission (no lock needed - single thread per submission)
        used_alternatives = set()
        pairs = []
        local_debug_counts = defaultdict(int)
        local_rejection_stats = defaultdict(int)
        local_failed_depths = defaultdict(int)
        local_successful_depths = defaultdict(int)
        local_rules = defaultdict(int)

        for mod_comment in submission_mod_comments:
            violating_comment_id = extract_comment_id(mod_comment.get('parent_id', ''))
            if not violating_comment_id:
                local_debug_counts['missing_parent_id'] += 1
                continue

            if violating_comment_id not in comments:
                local_debug_counts['missing_violating_comment'] += 1
                continue

            # Build thread pair
            pair, failed_depth, failure_reason, rejection_stats = build_thread_pair(
                mod_comment, comments, trees_data, logger, used_alternatives
            )

            # Aggregate rejection statistics
            if rejection_stats:
                for key, value in rejection_stats.items():
                    local_rejection_stats[key] += value

            if pair:
                pairs.append(pair)

                # Mark alternative as used
                compliant_comment_id = pair.get('metadata', {}).get('compliant_comment_id')
                if compliant_comment_id:
                    used_alternatives.add(compliant_comment_id)

                # Track successful depth
                violating_thread = pair.get('violating_thread', [])
                if violating_thread:
                    local_successful_depths[len(violating_thread)] += 1

                # Track rule statistics
                rule = mod_comment.get('matched_rule', {}).get('short_name_clean', 'unknown')
                local_rules[rule] += 1

            else:
                # Track failure
                if failure_reason:
                    local_debug_counts[failure_reason] += 1
                else:
                    local_debug_counts['no_alternative_found'] += 1

                if failed_depth is not None:
                    local_failed_depths[failed_depth] += 1

        return {
            'pairs': pairs,
            'debug_counts': dict(local_debug_counts),
            'rejection_stats': dict(local_rejection_stats),
            'failed_depths': dict(local_failed_depths),
            'successful_depths': dict(local_successful_depths),
            'rules': dict(local_rules)
        }

    # Process submissions in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_submission_threads, sub_id, mods): sub_id
                  for sub_id, mods in mod_comments_by_submission.items()}

        for future in as_completed(futures):
            result = future.result()

            # Aggregate results (no lock needed - main thread only)
            thread_pairs.extend(result['pairs'])

            for key, value in result['debug_counts'].items():
                debug_counts[key] = debug_counts.get(key, 0) + value

            for key, value in result['rejection_stats'].items():
                alternative_rejection_stats[key] = alternative_rejection_stats.get(key, 0) + value

            for depth, count in result['failed_depths'].items():
                failed_depth_distribution[depth] += count

            for depth, count in result['successful_depths'].items():
                successful_depth_distribution[depth] += count

            for rule, count in result['rules'].items():
                rule_stats[rule] = rule_stats.get(rule, 0) + count

    return {
        'thread_pairs': thread_pairs,
        'success_count': len(thread_pairs),
        'rule_stats': rule_stats,
        'debug_counts': debug_counts,
        'alternative_rejection_stats': alternative_rejection_stats,
        'failed_depth_distribution': failed_depth_distribution,
        'successful_depth_distribution': successful_depth_distribution
    }


# ============================================================================
# SUBREDDIT PROCESSING
# ============================================================================

def process_subreddit(args: tuple) -> Dict[str, Any]:
    """Process single subreddit: build trees and create discussion threads."""
    subreddit_name, complete_rule_set = args

    worker_logger = get_stage_logger(5, "build_trees_and_threads", worker_identifier=f"subreddits/{subreddit_name}")
    worker_logger.info(f"🔄 Processing {subreddit_name}")
    start_time = time.time()

    try:
        # Define paths
        submission_comments_dir = os.path.join(PATHS['organized_comments'], subreddit_name)
        match_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
        trees_output = os.path.join(PATHS['comment_trees'], f"{subreddit_name}_comment_trees.pkl")
        threads_output = os.path.join(PATHS['discussion_threads'], f"{subreddit_name}_discussion_threads.pkl")

        ensure_directory(trees_output)
        ensure_directory(threads_output)

        # Validate inputs
        if not os.path.exists(submission_comments_dir) or not os.path.isdir(submission_comments_dir):
            return {
                'subreddit': subreddit_name,
                'status': 'failed',
                'error': f"Missing directory: {submission_comments_dir}",
                'trees_built': 0,
                'successful_pairs': 0,
                'total_mod_comments': 0
            }

        if not os.path.exists(match_file):
            return {
                'subreddit': subreddit_name,
                'status': 'failed',
                'error': f"Missing file: {match_file}",
                'trees_built': 0,
                'successful_pairs': 0,
                'total_mod_comments': 0
            }

        # Count submissions
        submission_files = [f for f in os.listdir(submission_comments_dir)
                           if f.startswith('submission_') and f.endswith('.pkl')]

        if not submission_files:
            return {
                'subreddit': subreddit_name,
                'status': 'failed',
                'error': "No submission pickle files found",
                'trees_built': 0,
                'successful_pairs': 0,
                'total_mod_comments': 0
            }

        # Build trees
        trees_data = build_subreddit_trees(submission_comments_dir, subreddit_name, worker_logger)
        total_comments = trees_data['metadata']['total_comments']

        # Save trees
        with open(trees_output, 'wb') as f:
            pickle.dump(trees_data, f)

        trees_built = len(trees_data['trees'])
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

        # Build discussion threads
        worker_logger.info(f"  🧵 Building discussion threads from {len(mod_comments)} mod comments")

        threads_result = build_discussion_threads(
            mod_comments, submission_comments_dir, trees_data, complete_rule_set, worker_logger
        )

        # Extract results
        thread_pairs = threads_result['thread_pairs']
        success_count = threads_result['success_count']
        rule_stats = threads_result['rule_stats']
        debug_counts = threads_result['debug_counts']
        alternative_rejection_stats = threads_result['alternative_rejection_stats']
        failed_depth_distribution = threads_result['failed_depth_distribution']
        successful_depth_distribution = threads_result['successful_depth_distribution']

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

        worker_logger.info(f"  🎉 {subreddit_name}: {trees_built} trees, {success_count}/{len(mod_comments)} "
                          f"thread pairs ({success_rate:.1f}%) in {elapsed:.1f}s")

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
            'alternative_rejection_stats': alternative_rejection_stats,
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(5, "build_trees_and_threads")
    log_stage_start(logger, 5, "Build Comment Trees and Discussion Threads")

    start_time = time.time()

    try:
        create_directories()

        # Load subreddits and rule sets
        logger.info("📚 Loading target subreddits and rule sets...")
        subreddit_languages, subreddit_rules = load_target_subreddits_and_rules(logger)

        if not subreddit_languages:
            logger.error("❌ No subreddits loaded!")
            log_stage_end(logger, 5, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load Stage 3 manifest to determine which subreddits have valid matches
        stage3_summary_file = os.path.join(PATHS['data'], 'stage3_matching_summary.json')
        stage3_manifest_subreddits = set()
        if os.path.exists(stage3_summary_file):
            stage3_summary = read_json_file(stage3_summary_file)
            for stats in stage3_summary.get('subreddit_stats', []):
                if stats.get('matched_comments', 0) > 0:
                    stage3_manifest_subreddits.add(stats['subreddit'])
            logger.info(f"📋 Stage 3 manifest: {len(stage3_manifest_subreddits)} subreddits with matches")
        else:
            logger.warning("⚠️  Stage 3 summary not found, falling back to file existence checks")

        # Find subreddits with required input files, using manifest when available
        subreddits_to_process = []
        for subreddit in subreddit_languages.keys():
            submission_comments_dir = os.path.join(PATHS['organized_comments'], subreddit)
            match_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_match.jsonl.zst")

            # Use manifest to validate: subreddit must be in Stage 3 manifest (if available)
            if stage3_manifest_subreddits and subreddit not in stage3_manifest_subreddits:
                continue

            if os.path.exists(submission_comments_dir) and os.path.isdir(submission_comments_dir) and os.path.exists(match_file):
                subreddits_to_process.append((subreddit, subreddit_rules.get(subreddit, {})))

        if not subreddits_to_process:
            logger.error("❌ No subreddits found with required input files!")
            log_stage_end(logger, 5, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Found {len(subreddits_to_process)} subreddits to process")

        # Pre-cleanup: Remove stale tree/thread files not in current processing set
        subreddits_to_process_names = {s[0] for s in subreddits_to_process}
        for output_dir, suffix in [(PATHS['comment_trees'], '_comment_trees.pkl'),
                                    (PATHS['discussion_threads'], '_discussion_threads.pkl')]:
            if os.path.exists(output_dir):
                existing_files = [f for f in os.listdir(output_dir) if f.endswith(suffix)]
                stale_files = [f for f in existing_files
                              if f.replace(suffix, '') not in subreddits_to_process_names]
                if stale_files:
                    logger.info(f"🧹 Pre-cleanup: Removing {len(stale_files)} stale files from {os.path.basename(output_dir)}/")
                    for stale_file in stale_files:
                        os.remove(os.path.join(output_dir, stale_file))

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

        # Add language information to all completed results
        for result in completed_results:
            result['language'] = subreddit_languages.get(result['subreddit'], 'unknown')

        # Separate and rank English subreddits
        english_results = [r for r in completed_results if r.get('language') == 'en']
        other_language_results = [r for r in completed_results if r.get('language') != 'en']

        english_results = rank_by_score(english_results, 'jsd_from_uniform', ascending=True)

        for result in other_language_results:
            result['rank'] = None

        # Create summary
        elapsed = time.time() - start_time

        summary = {
            'summary': {
                'total_subreddits_processed': len(subreddits_to_process),
                'completed_subreddits': len(completed_results),
                'failed_subreddits': len(failed_results),
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
        summary_file = os.path.join(PATHS['data'], 'stage5_trees_and_threads_summary.json')
        write_json_file(summary, summary_file, pretty=True)

        logger.info(f"🎉 Stage 5 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Processed {len(completed_results)}/{len(subreddits_to_process)} subreddits")
        logger.info(f"🌳 Built {total_trees:,} comment trees")
        logger.info(f"🧵 Created {total_successful_pairs:,} discussion thread pairs")
        logger.info(f"📈 Overall success rate: {total_successful_pairs/total_mod_comments*100:.1f}%"
                   if total_mod_comments > 0 else "📈 No mod comments processed")
        logger.info(f"   🏆 English subreddits: {len(english_results)}")
        logger.info(f"   🌍 Other language subreddits: {len(other_language_results)}")
        logger.info(f"Summary saved to: {summary_file}")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_results)}):")
            for result in failed_results[:10]:
                logger.warning(f"  {result['subreddit']}: {result.get('error', 'Unknown error')}")
            if len(failed_results) > 10:
                logger.warning(f"  ... and {len(failed_results) - 10} more")

        log_stage_end(logger, 5, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 5 execution")
        log_stage_end(logger, 5, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
