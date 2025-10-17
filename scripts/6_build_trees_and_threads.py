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
import random
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, ensure_directory, get_file_size_gb)
from utils.reddit import (normalize_subreddit_name, extract_submission_id, extract_comment_id)
from utils.stats import calculate_jsd_from_uniform, rank_by_score


def load_target_subreddits_and_rules(logger) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load subreddits with their languages and complete rule sets."""
    subreddits_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

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


def calculate_depth_levels(root_comments: List[str], children: Dict[str, List[str]]) -> Dict[int, List[str]]:
    """Calculate depth levels for comments using BFS."""
    depth_levels = {}

    if not root_comments:
        return depth_levels

    # Initialize depth 0 with root comments
    depth_levels[0] = root_comments[:]
    comment_depths = {comment_id: 0 for comment_id in root_comments}

    # BFS to assign depths
    queue = [(comment_id, 0) for comment_id in root_comments]

    while queue:
        current_comment, current_depth = queue.pop(0)

        # Get children of current comment
        child_ids = children.get(current_comment, [])

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
    children = {}  # parent_id -> [child_ids]
    parent_map = {}  # child_id -> parent_id (clean IDs)
    root_comments = []  # comments that reply directly to submission

    # Build parent->children mapping
    for comment_id, comment_data in comments.items():
        parent_id = comment_data.get('parent_id', '')

        if not parent_id:
            continue

        # Clean parent_id using utility
        if parent_id.startswith('t3_'):
            # Direct reply to submission
            clean_parent_id = extract_submission_id(parent_id)
            root_comments.append(comment_id)
        elif parent_id.startswith('t1_'):
            # Reply to another comment
            clean_parent_id = extract_comment_id(parent_id)
        else:
            # Already clean or unknown format
            clean_parent_id = parent_id

        # Initialize parent in children dict if not exists
        if clean_parent_id not in children:
            children[clean_parent_id] = []

        # Add this comment as child of its parent
        children[clean_parent_id].append(comment_id)

        # Add to parent mapping (child -> parent)
        parent_map[comment_id] = clean_parent_id

    # Calculate depth levels using BFS
    depth_levels = calculate_depth_levels(root_comments, children)

    return {
        'children': children,
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


def load_mod_comments(match_file_path: str, logger, sample_size: int = 2000) -> List[Dict]:
    """Load and sample moderator comments from match file."""
    try:
        lines = read_zst_lines(match_file_path)
        mod_comments = []

        for line in lines:
            if line.strip():
                mod_comment = json_loads(line)
                mod_comments.append(mod_comment)

        # Sample comments if needed
        random.seed(0)
        if len(mod_comments) > sample_size:
            mod_comments = random.sample(mod_comments, sample_size)

        logger.info(f"Loaded {len(mod_comments)} moderator comments from {os.path.basename(match_file_path)}")
        return mod_comments

    except Exception as e:
        logger.error(f"Failed to load match file {match_file_path}: {e}")
        return []


def build_thread_to_root(comment_id: str, comments: Dict[str, Dict]) -> List[Dict]:
    """Build thread from comment up to root (excluding submission)."""
    thread = []
    current_id = comment_id

    # Build path from comment to root
    path = []
    while current_id in comments:
        comment = comments[current_id].copy()
        parent_id = comment.get('parent_id', '')

        path.append(current_id)

        # Stop if we reach submission
        if parent_id.startswith('t3_'):
            break

        # Move to parent using utility
        parent_id_clean = extract_comment_id(parent_id) if parent_id.startswith('t1_') else parent_id
        current_id = parent_id_clean

    # Reverse to get root-to-leaf order and add levels
    path.reverse()
    for level, comment_id in enumerate(path):
        comment = comments[comment_id].copy()
        comment['level'] = level
        thread.append(comment)

    return thread


def find_common_ancestors(comment1_id: str, comment2_id: str, comments: Dict[str, Dict]) -> int:
    """Count common ancestors between two comments."""
    def get_path_to_root(comment_id: str) -> List[str]:
        path = []
        current_id = comment_id

        while current_id in comments:
            path.append(current_id)
            parent_id = comments[current_id].get('parent_id', '')

            if parent_id.startswith('t3_'):  # Reached submission
                break

            parent_id_clean = extract_comment_id(parent_id) if parent_id.startswith('t1_') else parent_id
            current_id = parent_id_clean

        return path[::-1]  # Root to leaf

    path1 = get_path_to_root(comment1_id)
    path2 = get_path_to_root(comment2_id)

    # Count common prefix
    common_count = 0
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            common_count += 1
        else:
            break

    return common_count


def find_best_alternative(moderated_comment_id: str, moderated_thread_length: int,
                         submission_id: str, comments: Dict[str, Dict],
                         trees: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Find best alternative comment for unmoderated thread."""
    if submission_id not in trees['trees']:
        return None, None

    tree = trees['trees'][submission_id]
    depth_levels = tree.get('depth_levels', {})

    # Find moderated comment's depth
    moderated_depth = None
    for depth, comment_ids in depth_levels.items():
        if moderated_comment_id in comment_ids:
            moderated_depth = depth
            break

    if moderated_depth is None:
        return None, None

    # Get all comments at same depth
    same_depth_comments = depth_levels.get(moderated_depth, [])

    # Remove moderated comment from alternatives and sort for determinism
    alternatives = sorted([cid for cid in same_depth_comments if cid != moderated_comment_id])

    if not alternatives:
        return None, moderated_depth

    # Find best alternative based on common ancestors and score
    best_alternative = None
    max_common_ancestors = -1
    lowest_score = float('inf')

    for alt_id in alternatives:
        # Check if we can build a thread of required length
        alt_thread = build_thread_to_root(alt_id, comments)
        if len(alt_thread) < moderated_thread_length:
            continue

        # Count common ancestors
        common_ancestors = find_common_ancestors(moderated_comment_id, alt_id, comments)

        # Get comment score
        score = comments.get(alt_id, {}).get('score', 0)

        # Select best: most common ancestors first, then lowest score
        if (common_ancestors > max_common_ancestors or
            (common_ancestors == max_common_ancestors and score < lowest_score)):
            best_alternative = alt_id
            max_common_ancestors = common_ancestors
            lowest_score = score

    return best_alternative, moderated_depth


def build_thread_pair(mod_comment: Dict, comments: Dict[str, Dict],
                     trees: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[int]]:
    """Build moderated and unmoderated thread pair for a moderator comment."""
    mod_comment_id = mod_comment.get('id')
    moderated_comment_id = extract_comment_id(mod_comment.get('parent_id', ''))
    submission_id = extract_submission_id(mod_comment.get('link_id', ''))

    if not moderated_comment_id or not submission_id:
        return None, None

    # Check if we have the data
    if submission_id not in trees.get('trees', {}):
        return None, None

    if moderated_comment_id not in comments:
        return None, None

    # Build moderated thread
    moderated_thread = build_thread_to_root(moderated_comment_id, comments)

    if not moderated_thread:
        return None, None

    # Find best alternative at exact same depth (no fallback)
    target_length = len(moderated_thread)
    alt_comment_id, moderated_depth = find_best_alternative(
        moderated_comment_id, target_length,
        submission_id, comments, trees
    )

    if not alt_comment_id:
        return None, moderated_depth

    # Build unmoderated thread
    unmoderated_thread = build_thread_to_root(alt_comment_id, comments)

    # Calculate metadata
    common_ancestors = find_common_ancestors(moderated_comment_id, alt_comment_id, comments)

    return {
        'mod_comment_id': mod_comment_id,
        'mod_comment': mod_comment,  # Store full moderator comment object
        'moderated_thread': moderated_thread,
        'unmoderated_thread': unmoderated_thread,
        'metadata': {
            'common_ancestors': common_ancestors,
            'rule': mod_comment.get('matched_rule', {}).get('short_name', ''),
            'rule_similarity_score': mod_comment.get('matched_rule', {}).get('similarity_score', 0),
            'moderated_comment_id': moderated_comment_id,
            'unmoderated_comment_id': alt_comment_id,
            'submission_id': submission_id,
            'moderated_score': comments.get(moderated_comment_id, {}).get('score', 0),
            'unmoderated_score': comments.get(alt_comment_id, {}).get('score', 0),
            'target_length': target_length
        }
    }, None


def process_subreddit(args: tuple) -> Dict[str, Any]:
    """Process single subreddit: build trees and create discussion threads."""
    subreddit_name, complete_rule_set = args

    # Create worker logger with subreddit identifier
    worker_logger = get_stage_logger(6, "build_trees_and_threads", worker_identifier=subreddit_name)

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
        mod_comments = load_mod_comments(match_file, worker_logger, 2000)

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
            'no_alternative_found': 0
        }
        failed_depth_distribution = {}
        successful_depth_distribution = {}

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
            pair, failed_depth = build_thread_pair(mod_comment, comments, trees_data)
            if pair:
                thread_pairs.append(pair)
                success_count += 1

                # Track successful depth distribution
                moderated_thread = pair.get('moderated_thread', [])
                if moderated_thread:
                    successful_depth = len(moderated_thread)
                    successful_depth_distribution[successful_depth] = successful_depth_distribution.get(successful_depth, 0) + 1

                # Track rule statistics
                matched_rule = mod_comment.get('matched_rule', {})
                rule = matched_rule.get('short_name', 'unknown')
                rule_stats[rule] = rule_stats.get(rule, 0) + 1
            else:
                debug_counts['no_alternative_found'] += 1

                # Track failed depth distribution
                if failed_depth is not None:
                    failed_depth_distribution[failed_depth] = failed_depth_distribution.get(failed_depth, 0) + 1

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

        # Filter to subreddits with ≥500 successful pairs (keep ALL, no top-N filtering)
        qualified_results = [r for r in completed_results if r.get('successful_pairs', 0) >= 500]

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
        logger.info(f"✅ Qualified subreddits (≥500 pairs): {len(qualified_results)}")
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