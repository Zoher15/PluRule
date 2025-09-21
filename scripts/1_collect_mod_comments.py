#!/usr/bin/env python3
"""
Stage 1: Collect Moderator Comments

Extracts all distinguished moderator comments from Reddit comment archives,
filtering out bots and AutoModerator comments.

Input:  RC_*.zst files (Reddit comment archives)
Output: *_mod_comments.zst + subreddit_mod_comment_rankings.json
"""

import sys
import os
import time
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, DATE_RANGE, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import get_files_in_date_range, process_zst_file, process_files_parallel, write_json_file
from utils.reddit import is_bot_or_automoderator, is_moderator_comment, normalize_subreddit_name, filter_reddit_line


def process_comment_line(line: str, subreddit_counts: dict = None) -> bool:
    """Check if line contains a valid moderator comment and collect statistics."""
    # Quick pre-filter before JSON parsing (much faster)
    if not ('"distinguished":"moderator"' in line and '"parent_id":"t1_' in line):
        return False

    def check_comment(comment):
        if not is_moderator_comment(comment):
            return False

        # Filter out bots and AutoModerator
        author = comment.get('author', '')
        if is_bot_or_automoderator(author):
            return False

        # Collect subreddit statistics during processing (if dict provided)
        if subreddit_counts is not None:
            subreddit = normalize_subreddit_name(comment.get('subreddit', 'unknown'))
            subreddit_counts[subreddit] += 1

        return True

    return filter_reddit_line(line, check_comment)


def process_single_file(file_path: str) -> dict:
    """Process a single RC file and extract moderator comments."""
    # Create worker logger
    worker_logger = get_stage_logger(1, "collect_mod_comments")

    input_file = file_path
    output_file = os.path.join(
        PATHS['mod_comments'],
        os.path.basename(file_path).replace('.zst', '_mod_comments.zst')
    )

    # Skip if output already exists
    if os.path.exists(output_file):
        worker_logger.info(f"Skipping {os.path.basename(input_file)} - output already exists")
        return {"skipped": True, "file": input_file}

    try:
        # Create subreddit counts dictionary for this file
        subreddit_counts = defaultdict(int)

        # Create a closure that includes the subreddit_counts
        def process_line_with_stats(line: str) -> bool:
            return process_comment_line(line, subreddit_counts)

        # Process file with statistics collection
        stats = process_zst_file(input_file, output_file, process_line_with_stats, progress_interval=50_000_000, logger=worker_logger)

        return {
            "file": input_file,
            "output": output_file,
            "subreddit_counts": dict(subreddit_counts),
            **stats
        }

    except Exception as e:
        worker_logger.error(f"Error processing {input_file}: {e}")
        return {"file": input_file, "error": str(e)}


def collect_subreddit_stats(results: list, logger) -> dict:
    """Collect subreddit statistics from all processed files."""
    logger.info("Aggregating subreddit statistics...")

    total_subreddit_counts = defaultdict(int)

    for result in results:
        if result.get("skipped") or result.get("error"):
            continue

        # Use in-memory statistics from processing (much faster than re-reading files)
        file_subreddit_counts = result.get("subreddit_counts", {})
        for subreddit, count in file_subreddit_counts.items():
            total_subreddit_counts[subreddit] += count

    logger.info(f"Aggregated statistics for {len(total_subreddit_counts)} subreddits")
    return dict(total_subreddit_counts)


def generate_rankings(subreddit_counts: dict) -> dict:
    """Generate subreddit rankings from counts."""
    # Sort by count (descending)
    sorted_subreddits = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)

    rankings_data = {
        "summary": {
            "total_subreddits": len(subreddit_counts),
            "total_mod_comments": sum(subreddit_counts.values()),
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "rankings": [
            {
                "rank": rank,
                "subreddit": subreddit,
                "mod_comment_count": count
            }
            for rank, (subreddit, count) in enumerate(sorted_subreddits, 1)
        ]
    }

    return rankings_data


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(1, "collect_mod_comments")
    log_stage_start(logger, 1, "Collecting Moderator Comments")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Get RC files to process
        files = get_files_in_date_range(PATHS['reddit_comments'], 'RC_', DATE_RANGE, logger)

        if not files:
            logger.error("No RC files found to process!")
            log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"Found {len(files)} RC files to process")
        logger.info(f"Processing with {PROCESSES} parallel processes")

        # Process files in parallel
        logger.info("🚀 Processing RC files...")
        results = process_files_parallel(files, process_single_file, PROCESSES, logger)

        # Count results
        successful = len([r for r in results if not r.get("error") and not r.get("skipped")])
        skipped = len([r for r in results if r.get("skipped")])
        failed = len([r for r in results if r.get("error")])

        logger.info(f"File Processing Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Skipped: {skipped}")
        logger.info(f"  Failed: {failed}")

        if failed > 0:
            logger.warning("Failed files:")
            for result in results:
                if result.get("error"):
                    logger.warning(f"  {os.path.basename(result['file'])}: {result['error']}")

        # Collect subreddit statistics
        subreddit_counts = collect_subreddit_stats(results, logger)

        if not subreddit_counts:
            logger.error("No subreddit statistics collected!")
            log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Generate and save rankings
        logger.info("📈 Generating subreddit rankings...")
        rankings_data = generate_rankings(subreddit_counts)
        rankings_file = os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json')
        write_json_file(rankings_data, rankings_file)

        # Print summary
        elapsed = time.time() - start_time
        total_comments = rankings_data['summary']['total_mod_comments']
        total_subreddits = rankings_data['summary']['total_subreddits']

        logger.info(f"Stage 1 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Total mod comments: {total_comments:,}")
        logger.info(f"Unique subreddits: {total_subreddits:,}")
        logger.info(f"Rankings saved to: {rankings_file}")

        # Show top 10 subreddits
        logger.info(f"🏆 Top 10 subreddits by mod comment count:")
        for entry in rankings_data['rankings'][:10]:
            logger.info(f"  {entry['rank']:2d}. r/{entry['subreddit']}: {entry['mod_comment_count']:,}")

        log_stage_end(logger, 1, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 1 execution")
        log_stage_end(logger, 1, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())