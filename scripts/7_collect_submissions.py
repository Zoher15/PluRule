#!/usr/bin/env python3
"""
Stage 7: Collect Submissions from Discussion Threads

Collects submission data for all submissions referenced in discussion threads.
Extracts submission IDs from thread pair metadata and processes RS files to
find the corresponding submission objects.

Input:
- discussion_threads/{subreddit}_discussion_threads.pkl (from Stage 6)
- stage6_trees_and_threads_summary.json (for qualified subreddits)
- reddit_submissions/RS_*.zst files

Output:
- submissions/{subreddit}_submissions.zst
- stage7_submission_collection_stats.json
"""

import sys
import os
import time
import json
import pickle
from typing import Dict, Set, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, DATE_RANGE, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import (read_zst_lines, write_json_file, write_zst_json_objects, process_files_parallel,
                        get_files_in_date_range, process_zst_file_multi,
                        json_loads, get_file_size_gb, load_qualified_subreddits_from_stage6)
from utils.reddit import (normalize_subreddit_name, validate_submission_structure)


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
            metadata = pair.get('metadata', {})
            submission_id = metadata.get('submission_id')
            if submission_id:
                submission_ids.add(submission_id)

        logger.info(f"  📋 {subreddit}: {len(submission_ids)} unique submission IDs from {len(thread_pairs)} thread pairs")
        return submission_ids

    except Exception as e:
        logger.error(f"❌ Error loading threads for {subreddit}: {e}")
        return set()


def collect_subreddit_submission_ids(logger) -> Dict[str, Set[str]]:
    """Collect submission IDs for all qualified subreddits."""
    qualified_subreddit_stats = load_qualified_subreddits_from_stage6(logger)

    if not qualified_subreddit_stats:
        logger.error("❌ No qualified subreddits found!")
        return {}

    logger.info(f"📊 Extracting submission IDs from {len(qualified_subreddit_stats)} subreddits...")

    subreddit_submission_ids = {}
    total_unique_ids = set()

    for subreddit_stat in qualified_subreddit_stats:
        subreddit = subreddit_stat['subreddit']
        submission_ids = extract_submission_ids_from_threads(subreddit, logger)
        if submission_ids:
            subreddit_submission_ids[subreddit] = submission_ids
            total_unique_ids.update(submission_ids)

    logger.info(f"✅ Collected {len(total_unique_ids)} unique submission IDs across {len(subreddit_submission_ids)} subreddits")
    return subreddit_submission_ids


def process_rs_file(args: tuple) -> Dict[str, Any]:
    """
    Process single RS file and collect submissions for target subreddits.
    Uses temp subdirectories similar to Stage 3/5 pattern.
    """
    rs_file_path, subreddit_submission_ids, temp_dir = args

    rs_filename = os.path.basename(rs_file_path)
    # Extract RS identifier for meaningful logging (e.g., "RS_2023-02")
    rs_identifier = rs_filename.replace('.zst', '')

    # Create worker logger with meaningful identifier (in rs_files subdirectory)
    worker_logger = get_stage_logger(7, "collect_submissions", worker_identifier=f"rs_files/{rs_identifier}")
    rs_date = rs_filename.split('_')[1].split('.')[0]  # Extract YYYY-MM

    def submission_processor(line: str, state: Dict) -> Dict[str, Any]:
        """Process each submission line and route to appropriate subreddit temp file."""
        try:
            submission = json_loads(line)

            # Validate submission structure
            if not validate_submission_structure(submission):
                return {'matched': False}

            # Get submission info
            submission_id = submission.get('id', '')
            subreddit = normalize_subreddit_name(submission.get('subreddit', ''))

            if not submission_id or not subreddit:
                return {'matched': False}

            # Check if this submission is needed by any subreddit
            if subreddit in subreddit_submission_ids:
                target_submission_ids = subreddit_submission_ids[subreddit]
                if submission_id in target_submission_ids:
                    # Output to subreddit's temp directory
                    output_file = os.path.join(temp_dir, subreddit, f"RS_{rs_date}.zst")

                    return {
                        'matched': True,
                        'output_files': [output_file],
                        'data': submission
                    }

        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Skip malformed lines

        return {'matched': False}

    worker_logger.info(f"🔄 Processing {rs_filename}")

    try:
        # Process with multi-output utility
        stats = process_zst_file_multi(rs_file_path, submission_processor, {},
                                       progress_interval=1_000_000, logger=worker_logger)

        # Build output info from stats
        subreddits_with_submissions = len(stats["output_stats"])

        worker_logger.info(f"✅ {rs_filename}: {stats['lines_processed']:,} lines, {stats['lines_matched']:,} submissions -> {subreddits_with_submissions} subreddits")

        return {
            'rs_file': rs_filename,
            'total_submissions': stats['lines_processed'],
            'matched_submissions': stats['lines_matched'],
            'subreddits_with_data': subreddits_with_submissions,
            'success': True
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing {rs_filename}: {e}")
        return {
            'rs_file': rs_filename,
            'total_submissions': 0,
            'matched_submissions': 0,
            'subreddits_with_data': 0,
            'success': False,
            'error': str(e)
        }


def consolidate_subreddit_submissions(args: tuple) -> Dict[str, Any]:
    """Consolidate temp files for a single subreddit into final output."""
    subreddit, temp_dir, needed_submission_ids = args

    # Create worker logger with subreddit identifier (in subreddits subdirectory)
    worker_logger = get_stage_logger(7, "collect_submissions", worker_identifier=f"subreddits/{subreddit}")

    subreddit_temp_dir = os.path.join(temp_dir, subreddit)

    if not os.path.exists(subreddit_temp_dir):
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'success': False,
            'error': 'No temp directory found'
        }

    worker_logger.info(f"🔄 Consolidating submissions for {subreddit}")

    # Get all RS temp files for this subreddit
    temp_files = []
    for filename in os.listdir(subreddit_temp_dir):
        if filename.startswith('RS_') and filename.endswith('.zst'):
            temp_files.append(os.path.join(subreddit_temp_dir, filename))

    if not temp_files:
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'success': False,
            'error': 'No temp files found'
        }

    # Sort by date for chronological order
    temp_files.sort()

    # Stream submissions from temp files directly to output (memory-efficient)
    output_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")

    try:
        # Collect submissions in memory to deduplicate and filter removed/deleted
        # Simple logic: Skip all removed/deleted content, deduplicate by ID
        submissions_dict = {}
        skipped_removed_deleted = 0

        for temp_file in temp_files:
            try:
                for line in read_zst_lines(temp_file):
                    if line.strip():
                        submission = json_loads(line)
                        submission_id = submission.get('id', '')

                        if not submission_id:
                            continue

                        # Check if submission content is removed/deleted
                        selftext = submission.get('selftext', '')
                        selftext_html = submission.get('selftext_html', '')
                        removed_by_category = submission.get('removed_by_category')
                        is_removed_or_deleted = (
                            selftext in ['[removed]', '[deleted]'] or '[removed]' in selftext_html or '[deleted]' in selftext_html or
                            removed_by_category is not None
                        )

                        # Skip removed/deleted submissions entirely
                        if is_removed_or_deleted:
                            skipped_removed_deleted += 1
                            continue

                        # Add/update submission (later entries overwrite earlier ones)
                        submissions_dict[submission_id] = submission

            except Exception as e:
                worker_logger.warning(f"⚠️  Error reading {os.path.basename(temp_file)}: {e}")
                continue

        submissions_found = len(submissions_dict)

        worker_logger.info(f"  Filtered out {skipped_removed_deleted} removed/deleted submissions")

        # Write filtered and deduplicated submissions
        write_zst_json_objects(output_file, submissions_dict.values())

        if submissions_found == 0:
            # Clean up empty file
            if os.path.exists(output_file):
                os.remove(output_file)
            return {
                'subreddit': subreddit,
                'submissions_collected': 0,
                'success': False,
                'error': 'No submissions found in temp files'
            }

        file_size = get_file_size_gb(output_file)

        # Calculate coverage
        submissions_needed = len(needed_submission_ids)
        coverage_rate = submissions_found / submissions_needed if submissions_needed > 0 else 0

        worker_logger.info(f"✅ {subreddit}: {submissions_found:,}/{submissions_needed:,} submissions ({coverage_rate*100:.1f}% coverage) from {len(temp_files)} temp files ({file_size:.3f} GB)")

        # Clean up temp directory
        # import shutil
        # shutil.rmtree(subreddit_temp_dir)

        return {
            'subreddit': subreddit,
            'submissions_needed': submissions_needed,
            'submissions_collected': submissions_found,
            'submissions_skipped_removed_deleted': skipped_removed_deleted,
            'coverage_rate': coverage_rate,
            'output_file': output_file,
            'file_size_gb': file_size,
            'temp_files_processed': len(temp_files),
            'success': True
        }

    except Exception as e:
        return {
            'subreddit': subreddit,
            'submissions_collected': 0,
            'success': False,
            'error': f'Error writing final file: {e}'
        }


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(7, "collect_submissions")
    log_stage_start(logger, 7, "Collect Submissions from Discussion Threads")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Phase 1: Extract submission IDs from discussion threads
        logger.info("📋 Phase 1: Extracting submission IDs from discussion threads...")
        subreddit_submission_ids = collect_subreddit_submission_ids(logger)

        if not subreddit_submission_ids:
            logger.error("❌ No submission IDs collected!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Phase 2: Process RS files to collect submissions
        logger.info("🗃️  Phase 2: Processing RS files to collect submissions...")

        # Get RS files in date range
        rs_files = get_files_in_date_range(
            PATHS['reddit_submissions'],
            'RS_',
            DATE_RANGE,
            logger
        )

        if not rs_files:
            logger.error("❌ No RS files found to process!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Found {len(rs_files)} RS files to process")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Setup temp directory
        temp_dir = os.path.join(PATHS['submissions'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Process RS files in parallel
        rs_args = [(rs_file, subreddit_submission_ids, temp_dir) for rs_file in rs_files]
        rs_results = process_files_parallel(rs_args, process_rs_file, PROCESSES, logger)

        # Check results
        successful_rs = [r for r in rs_results if r.get('success', False)]
        failed_rs = [r for r in rs_results if not r.get('success', False)]

        total_submissions_processed = sum(r.get('total_submissions', 0) for r in successful_rs)
        total_submissions_collected = sum(r.get('matched_submissions', 0) for r in successful_rs)

        logger.info(f"✅ Phase 2 complete: {len(successful_rs)}/{len(rs_files)} files processed")
        logger.info(f"   📊 {total_submissions_processed:,} submissions processed, {total_submissions_collected:,} collected")

        if failed_rs:
            logger.warning(f"   ⚠️  {len(failed_rs)} files failed processing")

        # Phase 3: Consolidate submissions by subreddit
        logger.info(f"🗂️  Phase 3: Consolidating submissions for {len(subreddit_submission_ids)} subreddits...")

        # Find subreddits that have temp data
        subreddits_with_data = set()
        for result in successful_rs:
            # Check which subreddits actually got data
            for subreddit in subreddit_submission_ids.keys():
                subreddit_temp_dir = os.path.join(temp_dir, subreddit)
                if os.path.exists(subreddit_temp_dir) and os.listdir(subreddit_temp_dir):
                    subreddits_with_data.add(subreddit)

        if not subreddits_with_data:
            logger.error("❌ No subreddits have submission data!")
            log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Found {len(subreddits_with_data)} subreddits with submission data")

        # Consolidate in parallel
        consolidate_args = [(subreddit, temp_dir, subreddit_submission_ids[subreddit]) for subreddit in subreddits_with_data]
        consolidate_results = process_files_parallel(consolidate_args, consolidate_subreddit_submissions, PROCESSES, logger)

        # # Phase 4: Cleanup and statistics
        # logger.info("🧹 Phase 4: Cleanup and statistics...")

        # # Clean up remaining temp directory
        # import shutil
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        #     logger.info(f"   🗑️  Removed temp directory: {temp_dir}")

        # Collect final statistics
        successful_consolidations = [r for r in consolidate_results if r.get('success', False)]
        failed_consolidations = [r for r in consolidate_results if not r.get('success', False)]

        total_submissions_needed = sum(r.get('submissions_needed', 0) for r in successful_consolidations)
        total_submissions_final = sum(r.get('submissions_collected', 0) for r in successful_consolidations)
        total_submissions_skipped = sum(r.get('submissions_skipped_removed_deleted', 0) for r in successful_consolidations)
        total_file_size = sum(r.get('file_size_gb', 0) for r in successful_consolidations)
        overall_coverage = total_submissions_final / total_submissions_needed if total_submissions_needed > 0 else 0

        elapsed = time.time() - start_time

        # Create summary statistics
        summary = {
            'summary': {
                'total_qualified_subreddits': len(subreddit_submission_ids),
                'subreddits_with_submissions': len(successful_consolidations),
                'total_unique_submission_ids': len(set().union(*subreddit_submission_ids.values())),
                'total_submissions_needed': total_submissions_needed,
                'total_submissions_found': total_submissions_final,
                'total_submissions_skipped_removed_deleted': total_submissions_skipped,
                'overall_coverage_rate': overall_coverage,
                'total_rs_files_processed': len(successful_rs),
                'total_submissions_processed': total_submissions_processed,
                'total_submissions_collected': total_submissions_final,
                'collection_rate': total_submissions_collected / total_submissions_processed if total_submissions_processed > 0 else 0,
                'total_output_size_gb': total_file_size,
                'processing_time_seconds': elapsed,
                'failed_rs_files': len(failed_rs),
                'failed_consolidations': len(failed_consolidations),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'subreddit_stats': [
                {
                    'subreddit': r['subreddit'],
                    'submissions_needed': r.get('submissions_needed', 0),
                    'submissions_collected': r['submissions_collected'],
                    'submissions_skipped_removed_deleted': r.get('submissions_skipped_removed_deleted', 0),
                    'coverage_rate': r.get('coverage_rate', 0),
                    'file_size_gb': r.get('file_size_gb', 0),
                    'temp_files_processed': r.get('temp_files_processed', 0)
                }
                for r in sorted(successful_consolidations, key=lambda x: x.get('submissions_collected', 0), reverse=True)
            ],
            'failed_subreddits': [
                {
                    'subreddit': r['subreddit'],
                    'error': r.get('error', 'Unknown error')
                }
                for r in failed_consolidations
            ]
        }

        # Save summary
        summary_file = os.path.join(PATHS['data'], 'stage7_submission_collection_stats.json')
        write_json_file(summary, summary_file, pretty=True)

        logger.info(f"🎉 Stage 7 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Processed {len(successful_consolidations)}/{len(subreddit_submission_ids)} subreddits")
        logger.info(f"📝 Collected {total_submissions_final:,}/{total_submissions_needed:,} submissions ({overall_coverage*100:.1f}% coverage)")
        logger.info(f"🚫 Filtered out {total_submissions_skipped:,} removed/deleted submissions")
        logger.info(f"💾 Total size: {total_file_size:.2f} GB")
        logger.info(f"📈 RS file collection rate: {(total_submissions_collected/total_submissions_processed)*100:.1f}%" if total_submissions_processed > 0 else "📈 No submissions processed")
        logger.info(f"Summary saved to: {summary_file}")

        if failed_consolidations:
            logger.warning(f"⚠️  Failed subreddits ({len(failed_consolidations)}):")
            for result in failed_consolidations[:10]:  # Show first 10
                logger.warning(f"  {result['subreddit']}: {result.get('error', 'Unknown error')}")
            if len(failed_consolidations) > 10:
                logger.warning(f"  ... and {len(failed_consolidations) - 10} more")

        # Show top 10 subreddits by submission count
        if successful_consolidations:
            top_subreddits = sorted(successful_consolidations, key=lambda x: x.get('submissions_collected', 0), reverse=True)[:10]
            logger.info(f"🏆 Top 10 subreddits by submission count:")
            for i, result in enumerate(top_subreddits):
                count = result['submissions_collected']
                size = result.get('file_size_gb', 0)
                logger.info(f"  {i+1:2d}. {result['subreddit']}: {count:,} submissions ({size:.3f} GB)")

        log_stage_end(logger, 7, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 7 execution")
        log_stage_end(logger, 7, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())