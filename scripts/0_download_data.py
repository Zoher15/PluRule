#!/usr/bin/env python3
"""
Stage 0: Download Reddit Data from Internet Archive

Downloads Reddit comment and submission data from Internet Archive
for the specified date range. Handles the different URL patterns
for different time periods.

URL Patterns:
- 2005-12 to 2022-12: https://archive.org/download/pushshift_reddit_200506_to_202212/reddit/comments/RC_YYYY-MM.zst
- 2023-01: https://archive.org/download/pushshift_reddit_202301/reddit/comments/RC_2023-01.zst
- 2023-02: https://archive.org/download/pushshift_reddit_202302/reddit/comments/RC_2023-02.zst

Input:  None (downloads from internet)
Output: RC_*.zst and RS_*.zst files in reddit_data directory
"""

import sys
import os
import time
import requests
from datetime import datetime, timedelta
from typing import List, Tuple
from urllib.parse import urljoin

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, DATE_RANGE, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import process_files_parallel, ensure_directory


def generate_download_urls(date_range: Tuple[str, str], logger=None) -> List[Tuple[str, str, str]]:
    """
    Generate download URLs for the given date range.

    Returns:
        List of tuples: (url, filename, file_type)
        where file_type is 'comments' or 'submissions'
    """
    start_date, end_date = date_range

    # Parse dates
    start_year, start_month = map(int, start_date.split('-'))
    end_year, end_month = map(int, end_date.split('-'))

    download_urls = []

    # Generate all months in range
    current_date = datetime(start_year, start_month, 1)
    end_date_obj = datetime(end_year, end_month, 1)

    while current_date <= end_date_obj:
        year = current_date.year
        month = current_date.month
        date_str = f"{year}-{month:02d}"

        # Determine URL pattern based on date
        if year < 2023:
            # 2005-12 to 2022-12 pattern
            base_url = "https://archive.org/download/pushshift_reddit_200506_to_202212/reddit"
        elif year == 2023 and month == 1:
            # 2023-01 pattern
            base_url = "https://archive.org/download/pushshift_reddit_202301/reddit"
        elif year == 2023 and month == 2:
            # 2023-02 pattern
            base_url = "https://archive.org/download/pushshift_reddit_202302/reddit"
        else:
            if logger:
                logger.warning(f"⚠️  Warning: No known URL pattern for {date_str}")
            current_date += timedelta(days=32)
            current_date = current_date.replace(day=1)
            continue

        # Add comment and submission URLs
        for file_type, prefix in [('comments', 'RC'), ('submissions', 'RS')]:
            filename = f"{prefix}_{date_str}.zst"
            url = f"{base_url}/{file_type}/{filename}"

            download_urls.append((url, filename, file_type))

        # Move to next month
        current_date += timedelta(days=32)
        current_date = current_date.replace(day=1)

    return download_urls


def download_file(args: Tuple[str, str, str]) -> dict:
    """
    Download a single file from Internet Archive.

    Args:
        args: Tuple of (url, filename, file_type)

    Returns:
        Dictionary with download statistics
    """
    url, filename, file_type = args

    # Create worker logger
    worker_logger = get_stage_logger(0, "download_data")

    # Determine output directory
    if file_type == 'comments':
        output_dir = PATHS['reddit_comments']
    else:  # submissions
        output_dir = PATHS['reddit_submissions']

    output_path = os.path.join(output_dir, filename)

    # Skip if file already exists
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        worker_logger.info(f"✓ {filename} already exists ({file_size / (1024**3):.1f} GB)")
        return {
            "filename": filename,
            "url": url,
            "status": "skipped",
            "size_bytes": file_size,
            "output_path": output_path
        }

    # Ensure output directory exists
    ensure_directory(output_path)

    worker_logger.info(f"📥 Downloading {filename}...")
    start_time = time.time()

    try:
        # Start download with streaming
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))

        downloaded_bytes = 0
        chunk_size = 8192 * 16  # 128KB chunks

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    # Progress update every 100MB
                    if downloaded_bytes % (100 * 1024 * 1024) == 0:
                        if total_size > 0:
                            progress = (downloaded_bytes / total_size) * 100
                            worker_logger.info(f"  {filename}: {downloaded_bytes / (1024**3):.1f} GB / "
                                  f"{total_size / (1024**3):.1f} GB ({progress:.1f}%)")
                        else:
                            worker_logger.info(f"  {filename}: {downloaded_bytes / (1024**3):.1f} GB downloaded")

        elapsed = time.time() - start_time
        download_speed = downloaded_bytes / elapsed / (1024**2)  # MB/s

        worker_logger.info(f"✅ {filename} completed: {downloaded_bytes / (1024**3):.1f} GB "
              f"in {elapsed:.1f}s ({download_speed:.1f} MB/s)")

        return {
            "filename": filename,
            "url": url,
            "status": "success",
            "size_bytes": downloaded_bytes,
            "download_time": elapsed,
            "download_speed_mbps": download_speed,
            "output_path": output_path
        }

    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        worker_logger.error(f"❌ {filename} failed after {elapsed:.1f}s: {e}")

        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)

        return {
            "filename": filename,
            "url": url,
            "status": "failed",
            "error": str(e),
            "download_time": elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time
        worker_logger.error(f"❌ {filename} failed with unexpected error after {elapsed:.1f}s: {e}")

        # Clean up partial file
        if os.path.exists(output_path):
            os.remove(output_path)

        return {
            "filename": filename,
            "url": url,
            "status": "error",
            "error": str(e),
            "download_time": elapsed
        }


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(0, "download_data")
    log_stage_start(logger, 0, "Download Reddit Data from Internet Archive")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Ensure reddit data directories exist
        os.makedirs(PATHS['reddit_comments'], exist_ok=True)
        os.makedirs(PATHS['reddit_submissions'], exist_ok=True)

        # Generate download URLs
        logger.info(f"Generating download URLs for date range: {DATE_RANGE[0]} to {DATE_RANGE[1]}")
        download_urls = generate_download_urls(DATE_RANGE, logger)

        total_files = len(download_urls)
        comment_files = len([url for url in download_urls if url[2] == 'comments'])
        submission_files = len([url for url in download_urls if url[2] == 'submissions'])

        logger.info(f"Found {total_files} files to download:")
        logger.info(f"  Comment files (RC_*): {comment_files}")
        logger.info(f"  Submission files (RS_*): {submission_files}")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Download files in parallel
        logger.info("🚀 Starting parallel downloads...")
        results = process_files_parallel(download_urls, download_file, PROCESSES, logger)

        # Analyze results
        successful = len([r for r in results if r.get('status') == 'success'])
        skipped = len([r for r in results if r.get('status') == 'skipped'])
        failed = len([r for r in results if r.get('status') in ['failed', 'error']])

        total_downloaded_bytes = sum(r.get('size_bytes', 0) for r in results
                                    if r.get('status') in ['success', 'skipped'])

        elapsed = time.time() - start_time

        # Print summary
        logger.info(f"Stage 0 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Files downloaded: {successful}")
        logger.info(f"Files skipped (already exist): {skipped}")
        logger.info(f"Files failed: {failed}")
        logger.info(f"Total data: {total_downloaded_bytes / (1024**3):.1f} GB")

        if failed > 0:
            logger.warning(f"Failed downloads:")
            for result in results:
                if result.get('status') in ['failed', 'error']:
                    logger.warning(f"  {result['filename']}: {result.get('error', 'Unknown error')}")

        # Show download locations
        logger.info(f"Download locations:")
        logger.info(f"  Comments: {PATHS['reddit_comments']}")
        logger.info(f"  Submissions: {PATHS['reddit_submissions']}")

        # Save download log
        from utils.files import write_json_file

        download_log = {
            'date_range': DATE_RANGE,
            'total_files': total_files,
            'successful_downloads': successful,
            'skipped_files': skipped,
            'failed_downloads': failed,
            'total_size_gb': total_downloaded_bytes / (1024**3),
            'download_time_seconds': elapsed,
            'download_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results': results
        }

        log_file = os.path.join(PATHS['logs'], 'stage0_download_log.json')
        write_json_file(download_log, log_file)
        logger.info(f"Download log saved to: {log_file}")

        log_stage_end(logger, 0, success=(failed == 0), elapsed_time=elapsed)
        return 0 if failed == 0 else 1

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 0 execution")
        log_stage_end(logger, 0, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())