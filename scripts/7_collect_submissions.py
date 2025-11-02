#!/usr/bin/env python3
"""
Stage 7: Collect Submissions from Discussion Threads using Arctic Shift

Collects submission data for submissions referenced in discussion threads.
Uses Arctic Shift subreddit-specific submission files for efficient lookup.

Input:  Arctic Shift submission files + discussion_threads/{subreddit}_discussion_threads.pkl
Output: submissions/{subreddit}_submissions.zst + stage7_submission_collection_stats.json
"""

import sys
import os
import time
import pickle
from typing import Dict, Set, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, ARCTIC_SHIFT_DATA, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_zst_lines, write_json_file, process_files_parallel,
                        json_loads, process_zst_file_multi, load_qualified_subreddits_from_stage6)
from utils.reddit import normalize_subreddit_name, validate_submission_structure


def extract_submission_ids_from_threads(subreddit: str, logger) -> Set[str]:
    """Extract unique submission IDs from a subreddit's discussion threads."""
    threads_file = os.path.join(PATHS['discussion_threads'], f"{subreddit}_discussion_threads.pkl")

    if not os.path.exists(threads_file):
        logger.warning(f"⚠️  No discussion threads file found for {subreddit}")
        return set()

    try:
        with open(threads_file, 'rb') as f:
            threads_data = pickle.load(f)

        submission_ids = set()
        thread_pairs = threads_data.get('thread_pairs', [])

        for pair in thread_pairs:
            submission_id = pair.get('submission_id')
            if submission_id:
                submission_ids.add(submission_id)

        return submission_ids

    except Exception as e:
        logger.error(f"❌ Error reading threads for {subreddit}: {e}")
        return set()


def get_arctic_shift_submission_file(subreddit: str) -> str:
    """Get path to Arctic Shift submission file for a subreddit (case-insensitive)."""
    first_char = subreddit[0].lower() if subreddit else 'unknown'
    if not first_char.isalpha() and first_char.isdigit():
        first_char = str(first_char)

    dir_path = os.path.join(ARCTIC_SHIFT_DATA, first_char)
    if not os.path.exists(dir_path):
        return None

    target_filename_lower = f"{subreddit.lower()}_submissions.zst"
    try:
        for filename in os.listdir(dir_path):
            if filename.lower() == target_filename_lower:
                return os.path.join(dir_path, filename)
    except Exception:
        pass
    return None


def process_subreddit_submissions(args: tuple) -> Dict[str, Any]:
    """
    Collect submissions for a single subreddit from Arctic Shift.
    Uses process_zst_file_multi for efficient streaming.
    """
    subreddit, target_submission_ids, output_dir = args
    worker_logger = get_stage_logger(7, "collect_submissions", worker_identifier=f"subreddits/{subreddit}")

    worker_logger.info(f"🔄 Processing r/{subreddit} ({len(target_submission_ids)} target submissions)")
    start_time = time.time()

    # Get Arctic Shift file path
    arctic_file = get_arctic_shift_submission_file(subreddit)

    if not arctic_file:
        worker_logger.warning(f"⚠️  No Arctic Shift submission file found for {subreddit}")
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'lines_processed': 0,
            'processing_time': 0,
            'success': False,
            'error': 'Arctic Shift file not found'
        }

    output_file = os.path.join(output_dir, f"{subreddit}_submissions.zst")

    try:
        def submission_filter(line: str, state: Dict) -> Dict[str, Any]:
            """Filter for target submission IDs."""
            try:
                submission = json_loads(line)

                # Extract submission ID
                submission_id = submission.get('id')
                if not submission_id or submission_id not in target_submission_ids:
                    return {'matched': False}

                # Validate structure
                if not validate_submission_structure(submission):
                    return {'matched': False}

                # Match found - write to output
                return {
                    'matched': True,
                    'output_files': [output_file],
                    'data': submission
                }

            except Exception:
                return {'matched': False}

        # Use process_zst_file_multi for efficient streaming
        stats = process_zst_file_multi(arctic_file, submission_filter, {},
                                      progress_interval=10_000_000, logger=worker_logger)

        elapsed = time.time() - start_time
        submissions_collected = stats['lines_matched']

        worker_logger.info(f"✅ r/{subreddit}: {submissions_collected:,} submissions from {stats['lines_processed']:,} lines in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'submissions_collected': submissions_collected,
            'lines_processed': stats['lines_processed'],
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing r/{subreddit}: {e}")
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'lines_processed': 0,
            'processing_time': 0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main execution function."""
    logger = get_stage_logger(7, "collect_submissions")
    log_stage_start(logger, 7, "Collect Submissions from Arctic Shift")
    start_time = time.time()

    try:
        create_directories()

        if not os.path.exists(ARCTIC_SHIFT_DATA):
            logger.error(f"❌ Arctic Shift directory not found: {ARCTIC_SHIFT_DATA}")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load qualified subreddits from Stage 6
        logger.info("📋 Loading qualified subreddits from Stage 6...")
        qualified_subreddits = load_qualified_subreddits_from_stage6(PATHS['data'], logger)

        if not qualified_subreddits:
            logger.error("❌ No qualified subreddits found from Stage 6!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Extract submission IDs from discussion threads for each subreddit
        logger.info("🔍 Extracting submission IDs from discussion threads...")
        subreddit_submission_ids = {}

        for subreddit in qualified_subreddits:
            submission_ids = extract_submission_ids_from_threads(subreddit, logger)
            if submission_ids:
                subreddit_submission_ids[subreddit] = submission_ids

        logger.info(f"📊 Found submission IDs for {len(subreddit_submission_ids)} subreddits")
        total_submission_ids = sum(len(ids) for ids in subreddit_submission_ids.values())
        logger.info(f"📊 Total unique submission IDs to collect: {total_submission_ids:,}")

        if not subreddit_submission_ids:
            logger.error("❌ No submission IDs found!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Process subreddits in parallel
        logger.info(f"🗂️  Processing {len(subreddit_submission_ids)} subreddits with {PROCESSES} processes")
        processing_start = time.time()

        subreddit_args = [
            (subreddit, submission_ids, PATHS['submissions'])
            for subreddit, submission_ids in subreddit_submission_ids.items()
        ]
        results = process_files_parallel(subreddit_args, process_subreddit_submissions, PROCESSES, logger)

        processing_elapsed = time.time() - processing_start
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]

        total_submissions = sum(r['submissions_collected'] for r in successful_results)
        total_lines = sum(r['lines_processed'] for r in successful_results)

        logger.info(f"✅ Processing complete in {processing_elapsed:.1f}s")
        logger.info(f"   📊 {len(successful_results)}/{len(subreddit_submission_ids)} subreddits processed")
        logger.info(f"   📊 {total_lines:,} lines processed, {total_submissions:,} submissions collected")

        # Write statistics
        stats_data = {
            'summary': {
                'total_subreddits': len(results),
                'successful_subreddits': len(successful_results),
                'failed_subreddits': len(failed_results),
                'total_submissions_collected': total_submissions,
                'total_lines_processed': total_lines,
                'processing_time': round(processing_elapsed, 1),
                'total_time': round(time.time() - start_time, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Arctic Shift subreddit submission files'
            },
            'subreddit_stats': [
                {
                    'subreddit': r['subreddit'],
                    'submissions_collected': r['submissions_collected'],
                    'lines_processed': r['lines_processed'],
                    'processing_time': round(r['processing_time'], 2)
                }
                for r in sorted(successful_results, key=lambda x: x['submissions_collected'], reverse=True)
            ],
            'failed_subreddits': [
                {'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')}
                for r in failed_results
            ]
        }

        stats_file = os.path.join(PATHS['data'], 'stage7_submission_collection_stats.json')
        write_json_file(stats_data, stats_file, pretty=True)

        overall_elapsed = time.time() - start_time
        logger.info(f"🎉 Stage 7 Complete!")
        logger.info(f"   ⏱️  Total time: {overall_elapsed:.1f}s")
        logger.info(f"   🗂️  Subreddits processed: {len(successful_results)}/{len(subreddit_submission_ids)}")
        logger.info(f"   📄 Submissions collected: {total_submissions:,}")
        logger.info(f"   📈 Statistics: {stats_file}")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_results)}):")
            for r in failed_results[:10]:
                logger.warning(f"     r/{r['subreddit']}: {r.get('error', 'unknown')}")
            if len(failed_results) > 10:
                logger.warning(f"     ... and {len(failed_results) - 10} more")

        log_stage_end(logger, 7, success=True, elapsed_time=overall_elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 7 execution")
        log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
