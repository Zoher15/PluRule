#!/usr/bin/env python3
"""
Stage 5: Collect and Organize Submission Comments

Collects comments for target submission IDs and organizes them by subreddit.
Uses temp subdirectories approach similar to Stage 3.

Phase 1: Parallel RC file processing - each RC file writes filtered comments
         to temp/{subreddit}/RC_{date}.zst files
Phase 2: Parallel subreddit consolidation - organize comments into nested structure
         and output pickle files per subreddit
Phase 3: Cleanup temp directories

Input:  reddit_comments/RC_*.zst files + subreddit_submission_ids.json
Output: organized_comments/{subreddit}_submission_comments.pkl files
"""

import sys
import os
import time
import pickle
import shutil
from collections import defaultdict
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, DATE_RANGE, create_directories
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, process_zst_file_multi)
from utils.reddit import extract_submission_id, normalize_subreddit_name, validate_comment_structure

def load_submission_ids():
    """Load submission IDs from Stage 4 output."""
    submission_ids_file = os.path.join(PATHS['data'], 'subreddit_submission_ids.json')
    data = read_json_file(submission_ids_file)

    # Create lookup structures
    all_submission_ids = set()
    subreddit_to_ids = {}
    target_subreddits = set()

    for subreddit, submission_ids in data['subreddit_submission_ids'].items():
        normalized_sub = normalize_subreddit_name(subreddit)
        subreddit_to_ids[normalized_sub] = set(submission_ids)
        all_submission_ids.update(submission_ids)
        target_subreddits.add(normalized_sub)

    print(f"📋 Loaded {len(all_submission_ids)} submission IDs from {len(subreddit_to_ids)} subreddits")
    return all_submission_ids, subreddit_to_ids, target_subreddits

def get_rc_files():
    """Get list of RC files in date range."""
    from utils.files import get_files_in_date_range

    return get_files_in_date_range(
        PATHS['reddit_comments'],
        'RC_',
        DATE_RANGE
    )


def process_rc_file(args: tuple) -> Dict[str, Any]:
    """
    Phase 1: Process single RC file and write filtered comments to temp subdirs.
    Uses process_zst_file_multi for efficient streaming and multi-output.

    Args:
        args: (rc_file_path, subreddit_to_ids, target_subreddits, temp_dir)

    Returns:
        Dict with processing statistics
    """
    rc_file_path, subreddit_to_ids, target_subreddits, temp_dir = args

    rc_filename = os.path.basename(rc_file_path)
    rc_date = rc_filename.split('_')[1].split('.')[0]  # Extract YYYY-MM

    def comment_processor(line: str, state: Dict) -> Dict[str, Any]:
        """Process each comment line and route to appropriate subreddit temp file."""
        try:
            comment = json_loads(line)

            # Validate comment structure
            if not validate_comment_structure(comment):
                return {'matched': False}

            # Check if comment is from target subreddit
            subreddit = normalize_subreddit_name(comment.get('subreddit', ''))
            if subreddit not in target_subreddits:
                return {'matched': False}

            # Extract submission ID using utility
            submission_id = extract_submission_id(comment.get('link_id', ''))
            if not submission_id:
                return {'matched': False}

            subreddit_target_ids = subreddit_to_ids.get(subreddit, set())

            if submission_id in subreddit_target_ids:
                # Get original subreddit case for consistency
                original_subreddit = comment.get('subreddit', 'unknown')

                # Determine output file path
                output_file = os.path.join(temp_dir, original_subreddit, f"RC_{rc_date}.zst")

                return {
                    'matched': True,
                    'output_files': [output_file],
                    'data': comment
                }

        except Exception:
            pass  # Skip malformed lines

        return {'matched': False}

    print(f"🔄 Processing {rc_filename}")
    start_time = time.time()

    try:
        # Process with multi-output utility
        stats = process_zst_file_multi(rc_file_path, comment_processor, {})

        elapsed = time.time() - start_time
        subreddits_with_comments = len(stats["output_stats"])

        print(f"✅ {rc_filename}: {stats['lines_processed']:,} lines, {stats['lines_matched']:,} comments -> {subreddits_with_comments} subreddits in {elapsed:.1f}s")

        return {
            'rc_file': rc_filename,
            'total_lines': stats['lines_processed'],
            'matched_comments': stats['lines_matched'],
            'subreddits_with_comments': subreddits_with_comments,
            'processing_time': elapsed,
            'success': True
        }

    except Exception as e:
        print(f"❌ Error processing {rc_filename}: {e}")
        return {
            'rc_file': rc_filename,
            'total_lines': 0,
            'matched_comments': 0,
            'subreddits_with_comments': 0,
            'processing_time': 0,
            'success': False,
            'error': str(e)
        }

def organize_subreddit_comments(args: tuple) -> Dict[str, Any]:
    """
    Phase 2: Organize comments for a single subreddit into nested structure.

    Args:
        args: (subreddit, target_submission_ids, temp_dir, output_dir)

    Returns:
        Dict with organization statistics
    """
    subreddit, target_submission_ids, temp_dir, output_dir = args

    print(f"🔄 Organizing {subreddit} ({len(target_submission_ids)} target submissions)")
    start_time = time.time()

    # Build nested structure: {submission_id: {comment_id: comment_object}}
    submission_comments = defaultdict(dict)
    total_comments = 0
    files_processed = 0

    # Get subreddit's temp directory
    subreddit_temp_dir = os.path.join(temp_dir, subreddit)
    if not os.path.exists(subreddit_temp_dir):
        print(f"⚠️  No temp directory found for {subreddit}")
        return {
            'subreddit': subreddit,
            'comments_organized': 0,
            'submissions_with_comments': 0,
            'files_processed': 0,
            'processing_time': 0,
            'success': False
        }

    # Get all RC files in subreddit's temp dir
    rc_files = []
    for filename in os.listdir(subreddit_temp_dir):
        if filename.startswith('RC_') and filename.endswith('.zst'):
            rc_files.append(os.path.join(subreddit_temp_dir, filename))

    rc_files.sort()  # Process in chronological order

    # Process each RC file for this subreddit
    for rc_file_path in rc_files:
        try:
            for line_data in read_zst_lines(rc_file_path):
                try:
                    comment = json_loads(line_data)

                    # Validate comment structure
                    if not validate_comment_structure(comment):
                        continue

                    # Extract submission ID using utility
                    submission_id = extract_submission_id(comment.get('link_id', ''))
                    if not submission_id:
                        continue

                    # Check if this submission is in our target list
                    if submission_id in target_submission_ids:
                        comment_id = comment.get('id', f'unknown_{total_comments}')
                        submission_comments[submission_id][comment_id] = comment
                        total_comments += 1

                except Exception:
                    continue

            files_processed += 1

        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(rc_file_path)} for {subreddit}: {e}")
            continue

    # Write pickle file for this subreddit
    output_file = os.path.join(output_dir, f"{subreddit}_submission_comments.pkl")

    # Convert defaultdict to regular dict for pickling
    output_data = {submission_id: dict(comments) for submission_id, comments in submission_comments.items()}

    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - start_time
    total_submissions = len(submission_comments)
    avg_comments_per_submission = total_comments / total_submissions if total_submissions > 0 else 0

    print(f"✅ {subreddit}: {total_comments:,} comments across {total_submissions} submissions "
          f"(avg {avg_comments_per_submission:.1f} comments/submission) in {elapsed:.1f}s")

    return {
        'subreddit': subreddit,
        'comments_organized': total_comments,
        'submissions_with_comments': total_submissions,
        'files_processed': files_processed,
        'processing_time': elapsed,
        'success': True
    }

def main():
    """Main function to orchestrate the three-phase process."""
    print("🚀 Starting Stage 5: Collect and Organize Submission Comments")
    overall_start = time.time()

    # Create directories
    create_directories()

    # Load submission IDs from Stage 4
    _, subreddit_to_ids, target_subreddits = load_submission_ids()

    # Get RC files to process
    rc_files = get_rc_files()
    if not rc_files:
        print("❌ No RC files found to process")
        return

    # Setup temp directory
    temp_dir = os.path.join(PATHS['submission_comments'], 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Phase 1: Process RC files in parallel
    print(f"\n📁 Phase 1: Processing {len(rc_files)} RC files with {PROCESSES} processes")
    phase1_start = time.time()

    rc_args = [(rc_file, subreddit_to_ids, target_subreddits, temp_dir) for rc_file in rc_files]
    rc_results = process_files_parallel(process_rc_file, rc_args, max_workers=PROCESSES)

    phase1_elapsed = time.time() - phase1_start
    successful_rc = [r for r in rc_results if r['success']]
    total_lines_processed = sum(r['total_lines'] for r in successful_rc)
    total_comments_collected = sum(r['matched_comments'] for r in successful_rc)

    print(f"✅ Phase 1 complete in {phase1_elapsed:.1f}s")
    print(f"   📊 {len(successful_rc)}/{len(rc_files)} files processed successfully")
    print(f"   📊 {total_lines_processed:,} lines processed, {total_comments_collected:,} comments collected")

    # Phase 2: Organize comments by subreddit in parallel
    print(f"\n🗂️  Phase 2: Organizing comments for {len(subreddit_to_ids)} subreddits")
    phase2_start = time.time()

    subreddit_args = [(subreddit, target_ids, temp_dir, PATHS['organized_comments'])
                      for subreddit, target_ids in subreddit_to_ids.items()]
    org_results = process_files_parallel(organize_subreddit_comments, subreddit_args, max_workers=PROCESSES)

    phase2_elapsed = time.time() - phase2_start
    successful_orgs = [r for r in org_results if r['success']]
    total_comments_organized = sum(r['comments_organized'] for r in successful_orgs)
    total_submissions = sum(r['submissions_with_comments'] for r in successful_orgs)

    print(f"✅ Phase 2 complete in {phase2_elapsed:.1f}s")
    print(f"   📊 {len(successful_orgs)}/{len(subreddit_to_ids)} subreddits organized successfully")
    print(f"   📊 {total_comments_organized:,} comments organized across {total_submissions:,} submissions")

    # Phase 3: Cleanup temp directories
    print(f"\n🧹 Phase 3: Cleaning up temp directories")
    phase3_start = time.time()

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"   🗑️  Removed temp directory: {temp_dir}")

    phase3_elapsed = time.time() - phase3_start
    print(f"✅ Phase 3 complete in {phase3_elapsed:.1f}s")

    # Write summary statistics
    stats_file = os.path.join(PATHS['organized_comments'], 'stage5_submission_comment_organization_stats.json')
    stats_data = {
        'summary': {
            'total_subreddits': len(org_results),
            'successful_subreddits': len(successful_orgs),
            'total_comments_organized': total_comments_organized,
            'total_submissions_with_comments': total_submissions,
            'avg_comments_per_submission': round(total_comments_organized / total_submissions, 2) if total_submissions > 0 else 0,
            'phase1_time': round(phase1_elapsed, 1),
            'phase2_time': round(phase2_elapsed, 1),
            'phase3_time': round(phase3_elapsed, 1),
            'total_time': round(time.time() - overall_start, 1),
            'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'subreddit_stats': [
            {
                'subreddit': r['subreddit'],
                'comments_organized': r['comments_organized'],
                'submissions_with_comments': r['submissions_with_comments'],
                'files_processed': r['files_processed'],
                'processing_time': round(r['processing_time'], 2),
                'avg_comments_per_submission': round(r['comments_organized'] / r['submissions_with_comments'], 2) if r['submissions_with_comments'] > 0 else 0
            }
            for r in sorted(successful_orgs, key=lambda x: x['comments_organized'], reverse=True)
        ],
        'failed_subreddits': [
            {
                'subreddit': r['subreddit'],
                'reason': 'processing_failed'
            }
            for r in org_results if not r['success']
        ]
    }

    write_json_file(stats_file, stats_data)

    # Final summary
    overall_elapsed = time.time() - overall_start
    print(f"\n🎉 Stage 5 Complete!")
    print(f"   ⏱️  Total time: {overall_elapsed:.1f}s")
    print(f"   📁 RC files processed: {len(successful_rc)}/{len(rc_files)}")
    print(f"   🗂️  Subreddits organized: {len(successful_orgs)}/{len(subreddit_to_ids)}")
    print(f"   💬 Comments organized: {total_comments_organized:,}")
    print(f"   📝 Submissions with comments: {total_submissions:,}")
    print(f"   📊 Average comments per submission: {total_comments_organized/total_submissions:.1f}" if total_submissions > 0 else "   📊 No submissions found")
    print(f"   📈 Statistics: {stats_file}")

    # Show failed subreddits if any
    failed_subreddits = [r for r in org_results if not r['success']]
    if failed_subreddits:
        print(f"\n⚠️  Failed subreddits ({len(failed_subreddits)}):")
        for r in failed_subreddits[:10]:  # Show first 10
            print(f"     {r['subreddit']}")
        if len(failed_subreddits) > 10:
            print(f"     ... and {len(failed_subreddits) - 10} more")

if __name__ == "__main__":
    main()