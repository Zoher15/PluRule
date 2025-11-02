#!/usr/bin/env python3
"""
Stage 5: Collect and Organize Submission Comments

Collects comments for target submission IDs using Arctic Shift subreddit-specific files.
Two-pass approach: filter with process_zst_file_multi, then deduplicate with [removed]/[deleted] logic.

Input:  Arctic Shift subreddit comment files + stage4_subreddit_submission_ids.json
Output: organized_comments/{subreddit}_submission_comments.pkl files
"""

import sys
import os
import time
import pickle
import tempfile
import shutil
from collections import defaultdict
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, ARCTIC_SHIFT_DATA, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, process_zst_file_multi)
from utils.reddit import extract_submission_id, normalize_subreddit_name


def load_submission_ids(logger):
    """Load submission IDs from Stage 4 output."""
    data = read_json_file(os.path.join(PATHS['data'], 'stage4_subreddit_submission_ids.json'))

    subreddit_to_ids = {}
    for subreddit, submission_ids in data['subreddit_submission_ids'].items():
        subreddit_to_ids[normalize_subreddit_name(subreddit)] = set(submission_ids)

    logger.info(f"📋 Loaded {sum(len(ids) for ids in subreddit_to_ids.values()):,} submission IDs from {len(subreddit_to_ids)} subreddits")
    return subreddit_to_ids


def get_arctic_shift_comment_file(subreddit: str) -> str:
    """Get path to Arctic Shift comment file with case-insensitive matching."""
    first_char = subreddit[0].lower() if subreddit else 'unknown'
    if not first_char.isalpha() and first_char.isdigit():
        first_char = str(first_char)

    dir_path = os.path.join(ARCTIC_SHIFT_DATA, first_char)
    if not os.path.exists(dir_path):
        return None

    target_filename_lower = f"{subreddit.lower()}_comments.zst"
    try:
        for filename in os.listdir(dir_path):
            if filename.lower() == target_filename_lower:
                return os.path.join(dir_path, filename)
    except Exception:
        pass
    return None


def process_subreddit_comments(args: tuple) -> Dict[str, Any]:
    """
    Process comments for a single subreddit from Arctic Shift file.
    Pass 1: Filter with process_zst_file_multi to temp file
    Pass 2: Deduplicate and apply [removed]/[deleted] preservation logic
    """
    subreddit, target_submission_ids, output_dir = args
    worker_logger = get_stage_logger(5, "collect_submission_comments", worker_identifier=f"subreddits/{subreddit}")

    worker_logger.info(f"🔄 Processing {subreddit} ({len(target_submission_ids)} target submissions)")
    start_time = time.time()

    arctic_file = get_arctic_shift_comment_file(subreddit)
    if not arctic_file:
        worker_logger.warning(f"⚠️  No Arctic Shift comment file found for {subreddit}")
        return {'subreddit': subreddit, 'comments_collected': 0, 'submissions_with_comments': 0,
                'lines_processed': 0, 'removed_deleted_count': 0, 'preserved_from_removal_count': 0,
                'overwritten_with_better_count': 0, 'processing_time': 0, 'success': False,
                'error': 'Arctic Shift file not found'}

    temp_dir = None
    try:
        # Pass 1: Filter comments to temp file
        temp_dir = tempfile.mkdtemp(prefix=f"stage5_{subreddit}_")
        temp_file = os.path.join(temp_dir, "filtered_comments.zst")

        worker_logger.info(f"   Pass 1: Filtering comments...")
        pass1_start = time.time()

        def comment_filter(line: str, state: Dict) -> Dict[str, Any]:
            try:
                comment = json_loads(line)
                submission_id = extract_submission_id(comment.get('link_id', ''))
                if submission_id in target_submission_ids:
                    return {'matched': True, 'output_files': [temp_file], 'data': comment}
            except Exception:
                pass
            return {'matched': False}

        filter_stats = process_zst_file_multi(arctic_file, comment_filter, {}, progress_interval=10_000_000, logger=worker_logger)
        worker_logger.info(f"   ✅ Pass 1: {filter_stats['lines_matched']:,} matched from {filter_stats['lines_processed']:,} lines in {time.time()-pass1_start:.1f}s")

        # Pass 2: Deduplicate with [removed]/[deleted] preservation
        worker_logger.info(f"   Pass 2: Deduplicating and organizing...")
        pass2_start = time.time()

        submission_comments = defaultdict(dict)
        total_comments = 0
        removed_deleted_count = 0
        preserved_from_removal_count = 0
        overwritten_with_better_count = 0

        for line_data in read_zst_lines(temp_file):
            try:
                comment = json_loads(line_data)
                submission_id = extract_submission_id(comment.get('link_id', ''))
                if not submission_id:
                    continue

                comment_id = comment.get('id', f'unknown_{total_comments}')
                body = comment.get('body', '')
                author = comment.get('author', '')
                is_removed_or_deleted = body in ['[removed]', '[deleted]'] or author in ['[deleted]', '[removed]']

                if comment_id not in submission_comments[submission_id]:
                    submission_comments[submission_id][comment_id] = comment
                    total_comments += 1
                    if is_removed_or_deleted:
                        removed_deleted_count += 1
                elif not is_removed_or_deleted:
                    old_comment = submission_comments[submission_id][comment_id]
                    old_was_removed = (old_comment.get('body', '') in ['[removed]', '[deleted]'] or
                                      old_comment.get('author', '') in ['[deleted]', '[removed]'])
                    submission_comments[submission_id][comment_id] = comment
                    if old_was_removed:
                        overwritten_with_better_count += 1
                else:
                    preserved_from_removal_count += 1
            except Exception:
                continue

        worker_logger.info(f"   ✅ Pass 2: {total_comments:,} unique comments in {time.time()-pass2_start:.1f}s")

        # Clean up and write output
        shutil.rmtree(temp_dir)
        temp_dir = None

        output_file = os.path.join(output_dir, f"{subreddit}_submission_comments.pkl")
        output_data = {submission_id: dict(comments) for submission_id, comments in submission_comments.items()}
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elapsed = time.time() - start_time
        total_submissions = len(submission_comments)
        avg = total_comments / total_submissions if total_submissions > 0 else 0

        worker_logger.info(f"✅ {subreddit}: {total_comments:,} comments across {total_submissions} submissions (avg {avg:.1f}) in {elapsed:.1f}s")
        if removed_deleted_count > 0:
            worker_logger.info(f"   📊 {removed_deleted_count:,} [removed]/[deleted]")
        if preserved_from_removal_count > 0:
            worker_logger.info(f"   🛡️  {preserved_from_removal_count:,} preserved from overwrite")
        if overwritten_with_better_count > 0:
            worker_logger.info(f"   ✨ {overwritten_with_better_count:,} recovered")

        return {
            'subreddit': subreddit,
            'comments_collected': total_comments,
            'submissions_with_comments': total_submissions,
            'lines_processed': filter_stats['lines_processed'],
            'removed_deleted_count': removed_deleted_count,
            'preserved_from_removal_count': preserved_from_removal_count,
            'overwritten_with_better_count': overwritten_with_better_count,
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        worker_logger.error(f"❌ Error processing {subreddit}: {e}")
        return {'subreddit': subreddit, 'comments_collected': 0, 'submissions_with_comments': 0,
                'lines_processed': 0, 'removed_deleted_count': 0, 'preserved_from_removal_count': 0,
                'overwritten_with_better_count': 0, 'processing_time': 0, 'success': False, 'error': str(e)}


def main():
    """Main function to orchestrate comment collection from Arctic Shift files."""
    logger = get_stage_logger(5, "collect_submission_comments")
    log_stage_start(logger, 5, "Collect and Organize Submission Comments")
    overall_start = time.time()

    try:
        create_directories()

        if not os.path.exists(ARCTIC_SHIFT_DATA):
            logger.error(f"❌ Arctic Shift directory not found: {ARCTIC_SHIFT_DATA}")
            log_stage_end(logger, 5, success=False, elapsed_time=time.time() - overall_start)
            return 1

        logger.info("📋 Loading submission IDs from Stage 4...")
        subreddit_to_ids = load_submission_ids(logger)

        if not subreddit_to_ids:
            logger.error("❌ No submission IDs found to process")
            log_stage_end(logger, 5, success=False, elapsed_time=time.time() - overall_start)
            return 1

        logger.info(f"🗂️  Processing {len(subreddit_to_ids)} subreddits with {PROCESSES} processes")
        processing_start = time.time()

        subreddit_args = [(subreddit, target_ids, PATHS['organized_comments'])
                         for subreddit, target_ids in subreddit_to_ids.items()]
        results = process_files_parallel(subreddit_args, process_subreddit_comments, PROCESSES, logger)

        processing_elapsed = time.time() - processing_start
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        total_comments = sum(r['comments_collected'] for r in successful_results)
        total_submissions = sum(r['submissions_with_comments'] for r in successful_results)
        total_lines = sum(r['lines_processed'] for r in successful_results)

        logger.info(f"✅ Processing complete in {processing_elapsed:.1f}s")
        logger.info(f"   📊 {len(successful_results)}/{len(subreddit_to_ids)} subreddits processed")
        logger.info(f"   📊 {total_lines:,} lines processed, {total_comments:,} comments collected")
        logger.info(f"   📊 {total_submissions:,} submissions with comments")

        # Write statistics
        stats_data = {
            'summary': {
                'total_subreddits': len(results),
                'successful_subreddits': len(successful_results),
                'failed_subreddits': len(failed_results),
                'total_comments_collected': total_comments,
                'total_submissions_with_comments': total_submissions,
                'total_lines_processed': total_lines,
                'total_removed_deleted': sum(r.get('removed_deleted_count', 0) for r in successful_results),
                'total_preserved_from_removal': sum(r.get('preserved_from_removal_count', 0) for r in successful_results),
                'total_overwritten_with_better': sum(r.get('overwritten_with_better_count', 0) for r in successful_results),
                'avg_comments_per_submission': round(total_comments / total_submissions, 2) if total_submissions > 0 else 0,
                'processing_time': round(processing_elapsed, 1),
                'total_time': round(time.time() - overall_start, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Arctic Shift subreddit files'
            },
            'subreddit_stats': [
                {
                    'subreddit': r['subreddit'],
                    'comments_collected': r['comments_collected'],
                    'submissions_with_comments': r['submissions_with_comments'],
                    'lines_processed': r['lines_processed'],
                    'removed_deleted_count': r.get('removed_deleted_count', 0),
                    'preserved_from_removal_count': r.get('preserved_from_removal_count', 0),
                    'overwritten_with_better_count': r.get('overwritten_with_better_count', 0),
                    'processing_time': round(r['processing_time'], 2),
                    'avg_comments_per_submission': round(r['comments_collected'] / r['submissions_with_comments'], 2) if r['submissions_with_comments'] > 0 else 0
                }
                for r in sorted(successful_results, key=lambda x: x['comments_collected'], reverse=True)
            ],
            'failed_subreddits': [{'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')} for r in failed_results]
        }

        stats_file = os.path.join(PATHS['data'], 'stage5_submission_comment_collection_stats.json')
        write_json_file(stats_data, stats_file, pretty=True)

        overall_elapsed = time.time() - overall_start
        logger.info(f"🎉 Stage 5 Complete!")
        logger.info(f"   ⏱️  Total time: {overall_elapsed:.1f}s")
        logger.info(f"   🗂️  Subreddits processed: {len(successful_results)}/{len(subreddit_to_ids)}")
        logger.info(f"   💬 Comments collected: {total_comments:,}")
        logger.info(f"   📝 Submissions with comments: {total_submissions:,}")
        logger.info(f"   📊 Avg comments/submission: {total_comments/total_submissions:.1f}" if total_submissions > 0 else "   📊 No submissions found")
        logger.info(f"   📈 Statistics: {stats_file}")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_results)}):")
            for r in failed_results[:10]:
                logger.warning(f"     {r['subreddit']}: {r.get('error', 'unknown')}")
            if len(failed_results) > 10:
                logger.warning(f"     ... and {len(failed_results) - 10} more")

        log_stage_end(logger, 5, success=True, elapsed_time=overall_elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 5 execution")
        log_stage_end(logger, 5, success=False, elapsed_time=time.time() - overall_start)
        return 1


if __name__ == "__main__":
    exit(main())
