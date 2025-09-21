#!/usr/bin/env python3
"""
Stage 3: Filter and Consolidate Top N Subreddits

Filters mod comments from Stage 1 to only include comments from the top N
SFW subreddits identified in Stage 2, then consolidates them by subreddit.

Uses a two-phase approach:
- Phase 1: Filter RC files in parallel, writing to temp subreddit subdirs
- Phase 2: Consolidate temp files per subreddit in chronological order
- Phase 3: Clean up temp directories

Input:  mod_comments/*.zst files + top_N_sfw_subreddits.json
Output: top_subreddits/{subreddit}_mod_comments.jsonl.zst files
"""

import sys
import os
import time
import shutil
from collections import defaultdict
from typing import Dict, Set, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, write_zst_json_objects,
                        process_zst_file_multi)
from utils.reddit import clean_rule_text, normalize_subreddit_name, validate_comment_structure


def load_target_subreddits(logger) -> Set[str]:
    """Load the set of target subreddits from Stage 2 output."""
    subreddits_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

    if not os.path.exists(subreddits_file):
        logger.error(f"❌ Target subreddits file not found: {subreddits_file}")
        return set()

    logger.info(f"Loading target subreddits from: {subreddits_file}")
    subreddits_data = read_json_file(subreddits_file)

    target_subreddits = set()
    for entry in subreddits_data['subreddits']:
        # Use display_name or name field from the subreddit data
        subreddit_data = entry['subreddit']
        subreddit_name = subreddit_data.get('display_name') or subreddit_data.get('name', '')
        if subreddit_name:
            # Normalize to lowercase for case-insensitive matching
            target_subreddits.add(subreddit_name.lower())

    logger.info(f"Loaded {len(target_subreddits)} target subreddits")
    return target_subreddits


def process_single_file(args: tuple) -> Dict[str, Any]:
    """
    Process a single mod comments file and write to temp subreddit subdirs.
    Uses process_zst_file_multi for efficient streaming and multi-output.
    """
    file_path, target_subreddits = args

    # Create worker logger
    worker_logger = get_stage_logger(3, "filter_and_consolidate")

    file_name = os.path.basename(file_path)
    rc_date = file_name.replace('_mod_comments.zst', '').replace('RC_', '')
    temp_dir = os.path.join(PATHS['top_subreddits'], 'temp')

    def comment_processor(line: str, state: Dict) -> Dict[str, Any]:
        """Process each comment line and route to appropriate subreddit temp file."""
        try:
            comment = json_loads(line)

            # Validate comment structure
            if not validate_comment_structure(comment):
                return {'matched': False}

            # Normalize subreddit name for consistency
            subreddit = normalize_subreddit_name(comment.get('subreddit', ''))

            # Filter for target subreddits
            if subreddit in target_subreddits:
                # Add cleaned versions of text fields
                body = comment.get('body', '')
                removal_reason = comment.get('removal_reason', '')

                comment['body_clean'] = clean_rule_text(body)
                comment['removal_reason_clean'] = clean_rule_text(removal_reason)

                # Determine output file path
                output_file = os.path.join(temp_dir, subreddit, f"RC_{rc_date}_mod_comments.zst")

                return {
                    'matched': True,
                    'output_files': [output_file],
                    'data': comment
                }

        except Exception:
            pass  # Skip malformed lines

        return {'matched': False}

    worker_logger.info(f"🔄 Processing {file_name}")

    try:
        # Process with multi-output utility
        stats = process_zst_file_multi(file_path, comment_processor, {}, logger=worker_logger)

        # Build written_files info from output_stats
        written_files = {}
        for output_file, count in stats["output_stats"].items():
            subreddit = os.path.basename(os.path.dirname(output_file))
            written_files[subreddit] = {
                'temp_file': output_file,
                'comments': count,
                'file_size': os.path.getsize(output_file) if os.path.exists(output_file) else 0
            }

        worker_logger.info(f"✅ {file_name}: {stats['lines_processed']:,} lines, {stats['lines_matched']:,} target comments -> {len(written_files)} temp files")

        return {
            "file": file_path,
            "total_comments": stats["lines_processed"],
            "filtered_comments": stats["lines_matched"],
            "written_files": written_files
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing {file_path}: {e}")
        return {"file": file_path, "error": str(e)}


def consolidate_subreddit(args: tuple) -> Dict[str, Any]:
    """
    Consolidate temp files for a single subreddit into final output file.
    Processes run in parallel per subreddit.
    """
    subreddit, _ = args

    # Create worker logger
    worker_logger = get_stage_logger(3, "filter_and_consolidate")

    temp_dir = os.path.join(PATHS['top_subreddits'], 'temp', subreddit)

    if not os.path.exists(temp_dir):
        return {"subreddit": subreddit, "error": "No temp directory found"}

    # Get all RC temp files for this subreddit, sorted by date
    temp_files = []
    for filename in os.listdir(temp_dir):
        if filename.startswith('RC_') and filename.endswith('_mod_comments.zst'):
            temp_files.append(os.path.join(temp_dir, filename))

    if not temp_files:
        return {"subreddit": subreddit, "error": "No temp files found"}

    # Sort by RC date for chronological order
    temp_files.sort()

    worker_logger.info(f"🔄 Consolidating r/{subreddit}: {len(temp_files)} files")

    # Read all comments from temp files in date order
    all_comments = []
    total_temp_files = 0

    try:
        for temp_file in temp_files:
            total_temp_files += 1
            lines = read_zst_lines(temp_file)

            for line in lines:
                if line.strip():
                    comment = json_loads(line)
                    all_comments.append(comment)

    except Exception as e:
        return {"subreddit": subreddit, "error": f"Error reading temp files: {e}"}

    if not all_comments:
        return {"subreddit": subreddit, "error": "No comments found in temp files"}

    # Write final consolidated file
    output_file = os.path.join(PATHS['top_subreddits'], f"{subreddit}_mod_comments.jsonl.zst")

    try:
        # Write final consolidated file using centralized utility
        write_zst_json_objects(output_file, all_comments)
        file_size = os.path.getsize(output_file)

        worker_logger.info(f"✅ r/{subreddit}: {len(all_comments):,} comments from {total_temp_files} temp files -> {file_size:,} bytes")

        # Clean up this subreddit's temp directory after successful consolidation
        shutil.rmtree(temp_dir)

        return {
            "subreddit": subreddit,
            "output_file": output_file,
            "comments": len(all_comments),
            "file_size": file_size,
            "temp_files_processed": total_temp_files
        }

    except Exception as e:
        return {"subreddit": subreddit, "error": f"Error writing final file: {e}"}


def cleanup_temp_files(logger):
    """Remove temp directory after consolidation."""
    temp_dir = os.path.join(PATHS['top_subreddits'], 'temp')

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f"🧹 Cleaned up temp directory: {temp_dir}")


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(3, "filter_and_consolidate")
    log_stage_start(logger, 3, "Filter and Consolidate Top N Subreddits")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Load target subreddits
        logger.info("📚 Loading target subreddits...")
        target_subreddits = load_target_subreddits(logger)

        if not target_subreddits:
            logger.error("❌ No target subreddits loaded!")
            log_stage_end(logger, 3, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Get mod comment files to process
        mod_comment_files = []
        mod_comments_dir = PATHS['mod_comments']

        if not os.path.exists(mod_comments_dir):
            logger.error(f"❌ Mod comments directory not found: {mod_comments_dir}")
            log_stage_end(logger, 3, success=False, elapsed_time=time.time() - start_time)
            return 1

        for filename in os.listdir(mod_comments_dir):
            if filename.endswith('_mod_comments.zst'):
                mod_comment_files.append(os.path.join(mod_comments_dir, filename))

        if not mod_comment_files:
            logger.error("❌ No mod comment files found to process!")
            log_stage_end(logger, 3, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Sort files by date (oldest first) for better chronological processing
        mod_comment_files.sort()

        logger.info(f"Found {len(mod_comment_files)} mod comment files to process")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Phase 1: Process all RC files in parallel, writing to temp subdirs
        process_args = [(file_path, target_subreddits) for file_path in mod_comment_files]

        logger.info("🚀 Phase 1: Processing RC files to temp subdirs...")
        results = process_files_parallel(process_args, process_single_file, PROCESSES, logger)

        # Check for processing errors
        failed_files = [r for r in results if r.get("error")]
        total_lines = sum(r.get("total_comments", 0) for r in results if not r.get("error"))
        total_filtered = sum(r.get("filtered_comments", 0) for r in results if not r.get("error"))

        if failed_files:
            logger.warning(f"⚠️  {len(failed_files)} files failed processing:")
            for result in failed_files:
                logger.warning(f"  {os.path.basename(result['file'])}: {result.get('error', 'Unknown error')}")

        # Collect subreddits that have temp data
        subreddits_with_data = set()
        for result in results:
            if not result.get("error"):
                subreddits_with_data.update(result.get("written_files", {}).keys())

        if not subreddits_with_data:
            logger.error("❌ No subreddits have temp data!")
            log_stage_end(logger, 3, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"✅ Phase 1 complete: {len(subreddits_with_data)} subreddits have temp data")

        # Phase 2: Consolidate temp files per subreddit in parallel
        consolidate_args = [(subreddit, target_subreddits) for subreddit in subreddits_with_data]

        logger.info(f"🚀 Phase 2: Consolidating {len(subreddits_with_data)} subreddits...")
        consolidate_results = process_files_parallel(consolidate_args, consolidate_subreddit, PROCESSES, logger)

        # Collect final statistics
        successful_subreddits = 0
        total_final_comments = 0
        total_file_size = 0
        final_subreddit_stats = {}
        failed_consolidations = []

        for result in consolidate_results:
            if result.get("error"):
                failed_consolidations.append(result)
            else:
                subreddit = result["subreddit"]
                successful_subreddits += 1
                total_final_comments += result["comments"]
                total_file_size += result["file_size"]

                final_subreddit_stats[subreddit] = {
                    "comments": result["comments"],
                    "file_size": result["file_size"],
                    "output_file": result["output_file"]
                }

        # Phase 3: Cleanup temp files
        logger.info(f"🚀 Phase 3: Cleaning up temp files...")
        cleanup_temp_files(logger)

        # Calculate final statistics
        elapsed = time.time() - start_time

        # Save summary statistics
        summary = {
            'total_files_processed': len(mod_comment_files),
            'total_lines_processed': total_lines,
            'total_comments_filtered': total_filtered,
            'final_comments_in_output': total_final_comments,
            'target_subreddits_count': len(target_subreddits),
            'subreddits_with_data': successful_subreddits,
            'total_output_size_bytes': total_file_size,
            'processing_time_seconds': elapsed,
            'failed_files': len(failed_files),
            'failed_consolidations': len(failed_consolidations),
            'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'subreddit_details': {sub: {"comments": stats["comments"], "file_size": stats["file_size"]}
                                 for sub, stats in final_subreddit_stats.items()}
        }

        summary_file = os.path.join(PATHS['data'], 'stage3_filter_and_consolidate_summary.json')
        write_json_file(summary, summary_file)

        logger.info(f"🎉 Stage 3 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Created {successful_subreddits} subreddit files")
        logger.info(f"💬 Total comments: {total_final_comments:,}")
        logger.info(f"💾 Total size: {total_file_size:,} bytes ({total_file_size/1024/1024:.1f} MB)")
        logger.info(f"🚀 Processing rate: {total_lines/elapsed:,.0f} lines/sec")
        logger.info(f"Summary saved to: {summary_file}")

        if failed_consolidations:
            logger.warning(f"⚠️  {len(failed_consolidations)} consolidations failed:")
            for result in failed_consolidations:
                logger.warning(f"  r/{result['subreddit']}: {result.get('error', 'Unknown error')}")

        # Show top 10 subreddits by comment count
        top_subreddits = sorted(final_subreddit_stats.items(), key=lambda x: x[1]["comments"], reverse=True)[:10]
        logger.info(f"🏆 Top 10 subreddits by comment count:")
        for i, (subreddit, stats) in enumerate(top_subreddits):
            count = stats["comments"]
            size = stats["file_size"]
            logger.info(f"  {i+1:2d}. {subreddit}: {count:,} comments ({size:,} bytes)")

        log_stage_end(logger, 3, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 3 execution")
        log_stage_end(logger, 3, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())