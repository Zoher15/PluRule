#!/usr/bin/env python3
"""
Stage 8: Collect Media for Submissions

Downloads media files for submissions that have media content.
Uses hybrid approach: requests for direct downloads, yt-dlp for complex sites.

Input:
- submissions/{subreddit}_submissions.zst (from Stage 7)
- stage7_submission_collection_stats.json (for qualified subreddits)

Output:
- media/{subreddit}/{submission_id}.{ext} (downloaded media files)
- stage8_media_collection_stats.json
"""

import sys
import os
import time
import urllib.parse
import requests
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, ensure_directory, get_file_size_gb)
from utils.stats import rank_by_score
from utils.reddit import has_media, validate_submission_structure, normalize_subreddit_name


# User agent to avoid blocks
USER_AGENT = "reddit_research_media_collector/1.0"


def is_og_preview_image(url: str) -> bool:
    """
    Determine if a URL is likely an Open Graph preview image.
    These are usually low-quality thumbnails, not the actual content.
    """
    og_domains = ['external-preview.redd.it', 'preview.redd.it']
    parsed = urllib.parse.urlparse(url)
    return any(domain in parsed.netloc for domain in og_domains)


def create_requests_session() -> requests.Session:
    """Create a robust requests session with retry logic."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Mount adapters for HTTP and HTTPS
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set user agent
    session.headers.update({'User-Agent': USER_AGENT})

    return session


def get_file_extension_from_url(url: str) -> str:
    """Extract file extension from URL, with fallback logic."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()

    # Common extensions
    image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    video_exts = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv']
    audio_exts = ['.mp3', '.wav', '.ogg', '.flac']

    for ext in image_exts + video_exts + audio_exts:
        if path.endswith(ext):
            return ext

    # Domain-based fallbacks
    domain = parsed.netloc.lower()
    if 'imgur.com' in domain:
        return '.jpg'  # Imgur default
    elif 'i.redd.it' in domain:
        return '.jpg'  # Reddit image default
    elif 'v.redd.it' in domain:
        return '.mp4'  # Reddit video default
    elif 'youtube.com' in domain or 'youtu.be' in domain:
        return '.mp4'  # YouTube default

    return '.unknown'


def download_with_requests(url: str, output_path: str, session: requests.Session, logger=None) -> Dict[str, Any]:
    """
    Download file using requests with proper error handling.

    Returns:
        Dict with download result information
    """
    try:
        logger.debug(f"    📥 Downloading: {url}")

        # Stream download for large files
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Ensure output directory exists
        ensure_directory(output_path)

        # Download file in chunks
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(output_path)

        return {
            'success': True,
            'method': 'requests',
            'file_size': file_size,
            'url': url,
            'output_path': output_path
        }

    except Exception as e:
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)

        return {
            'success': False,
            'method': 'requests',
            'error': str(e),
            'url': url,
            'output_path': output_path
        }


def download_with_ytdlp(url: str, output_dir: str, submission_id: str, logger) -> Dict[str, Any]:
    """
    Download file using yt-dlp for complex sites.

    Returns:
        Dict with download result information
    """
    try:
        logger.debug(f"    📹 yt-dlp downloading: {url}")

        # Ensure output directory exists
        ensure_directory(os.path.join(output_dir, "dummy.txt"))

        # yt-dlp command
        cmd = [
            'yt-dlp',
            '--no-playlist',
            '--output', f'{output_dir}/{submission_id}.%(ext)s',
            '--user-agent', USER_AGENT,
            url
        ]

        # Run yt-dlp
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            # Find downloaded file
            downloaded_files = []
            for filename in os.listdir(output_dir):
                if filename.startswith(submission_id):
                    file_path = os.path.join(output_dir, filename)
                    downloaded_files.append(file_path)

            if downloaded_files:
                output_path = downloaded_files[0]  # Take first match
                file_size = os.path.getsize(output_path)

                return {
                    'success': True,
                    'method': 'yt-dlp',
                    'file_size': file_size,
                    'url': url,
                    'output_path': output_path
                }

        return {
            'success': False,
            'method': 'yt-dlp',
            'error': result.stderr or 'Unknown yt-dlp error',
            'url': url,
            'output_path': None
        }

    except Exception as e:
        return {
            'success': False,
            'method': 'yt-dlp',
            'error': str(e),
            'url': url,
            'output_path': None
        }


def download_submission_media(submission: Dict[str, Any], media_dir: str, session: requests.Session, logger) -> Dict[str, Any]:
    """Download media for a single submission."""
    submission_id = submission.get('id', 'unknown')
    url = submission.get('url', '')

    if not url:
        return {
            'submission_id': submission_id,
            'success': False,
            'error': 'No URL found',
            'url': url
        }

    # Skip OG preview images
    if is_og_preview_image(url):
        return {
            'submission_id': submission_id,
            'success': False,
            'error': 'Skipped OG preview image',
            'url': url
        }

    # Determine download strategy based on URL
    domain = urllib.parse.urlparse(url).netloc.lower()

    # YouTube and complex video sites - use yt-dlp
    if any(site in domain for site in ['youtube.com', 'youtu.be', 'v.redd.it']):
        result = download_with_ytdlp(url, media_dir, submission_id, logger)
    else:
        # Direct downloads - use requests
        extension = get_file_extension_from_url(url)
        output_path = os.path.join(media_dir, f"{submission_id}{extension}")
        result = download_with_requests(url, output_path, session, logger)

    # Add submission info to result
    result['submission_id'] = submission_id
    return result


def process_subreddit_media(args: tuple) -> Dict[str, Any]:
    """Process media downloads for a single subreddit."""
    subreddit_name, = args

    # Create worker logger (same function works in worker processes)
    worker_logger = get_stage_logger(8, "collect_media")

    # Normalize subreddit name for consistency
    normalized_subreddit = normalize_subreddit_name(subreddit_name)

    worker_logger.info(f"🔄 Processing media for {subreddit_name}")
    start_time = time.time()

    # File paths (use original name for file lookup, normalized for directory)
    submissions_file = os.path.join(PATHS['submissions'], f"{subreddit_name}_submissions.zst")
    subreddit_media_dir = os.path.join(PATHS['media'], normalized_subreddit)

    if not os.path.exists(submissions_file):
        return {
            'subreddit': subreddit_name,
            'success': False,
            'error': f'Submissions file not found: {submissions_file}',
            'media_downloaded': 0,
            'total_submissions': 0
        }

    try:
        # Create requests session
        session = create_requests_session()

        # Load submissions
        submission_lines = read_zst_lines(submissions_file)
        submissions = []

        for line in submission_lines:
            if line.strip():
                submission = json_loads(line)
                # Validate submission structure before processing
                if validate_submission_structure(submission):
                    submissions.append(submission)

        worker_logger.info(f"  📊 Loaded {len(submissions)} submissions for {subreddit_name}")

        # Filter for media submissions
        media_submissions = []
        for submission in submissions:
            if has_media(submission):
                media_submissions.append(submission)

        worker_logger.info(f"  🎬 Found {len(media_submissions)} media submissions in {subreddit_name}")

        if not media_submissions:
            return {
                'subreddit': subreddit_name,
                'success': True,
                'media_downloaded': 0,
                'total_submissions': len(submissions),
                'media_submissions': 0,
                'processing_time': time.time() - start_time
            }

        # Download media files
        download_results = []
        successful_downloads = 0
        total_size = 0

        for submission in media_submissions:
            result = download_submission_media(submission, subreddit_media_dir, session, worker_logger)
            download_results.append(result)

            if result.get('success', False):
                successful_downloads += 1
                total_size += result.get('file_size', 0)

        elapsed = time.time() - start_time
        success_rate = (successful_downloads / len(media_submissions)) * 100 if media_submissions else 0

        worker_logger.info(f"  ✅ {subreddit_name}: {successful_downloads}/{len(media_submissions)} downloads successful ({success_rate:.1f}%) in {elapsed:.1f}s")

        # Group failures by error type for debugging
        failure_reasons = {}
        for result in download_results:
            if not result.get('success', False):
                error = result.get('error', 'Unknown error')
                failure_reasons[error] = failure_reasons.get(error, 0) + 1

        return {
            'subreddit': subreddit_name,
            'success': True,
            'media_downloaded': successful_downloads,
            'total_submissions': len(submissions),
            'media_submissions': len(media_submissions),
            'success_rate': success_rate,
            'total_size_bytes': total_size,
            'processing_time': elapsed,
            'failure_reasons': failure_reasons,
            'download_details': download_results
        }

    except Exception as e:
        worker_logger.error(f"❌ Error processing {subreddit_name}: {e}")
        return {
            'subreddit': subreddit_name,
            'success': False,
            'error': str(e),
            'media_downloaded': 0,
            'total_submissions': 0,
            'processing_time': time.time() - start_time
        }


def load_qualified_subreddits(logger) -> List[str]:
    """Load subreddits that have submissions from Stage 7."""
    summary_file = os.path.join(PATHS['data'], 'stage7_submission_collection_stats.json')

    try:
        summary = read_json_file(summary_file)

        # Get subreddits with successful submission collection
        qualified_subreddits = []
        for subreddit_stat in summary.get('subreddit_stats', []):
            if subreddit_stat.get('submissions_collected', 0) > 0:
                qualified_subreddits.append(subreddit_stat['subreddit'])

        logger.info(f"Loaded {len(qualified_subreddits)} subreddits with submissions from Stage 7")
        return qualified_subreddits

    except Exception as e:
        log_error_and_continue(logger, e, "loading Stage 7 summary")
        return []


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(8, "collect_media")
    log_stage_start(logger, 8, "Collect Media for Submissions")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Check if yt-dlp is available
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
            logger.info("✅ yt-dlp found and working")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️  yt-dlp not found - video downloads may fail")
            logger.warning("Install with: pip install yt-dlp")

        # Load qualified subreddits from Stage 7
        qualified_subreddits = load_qualified_subreddits(logger)

        if not qualified_subreddits:
            logger.error("❌ No qualified subreddits found!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"📊 Processing media for {len(qualified_subreddits)} subreddits")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Process subreddits in parallel
        subreddit_args = [(subreddit,) for subreddit in qualified_subreddits]
        results = process_files_parallel(subreddit_args, process_subreddit_media, PROCESSES, logger)

        # Collect statistics
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]

        total_media_downloaded = sum(r.get('media_downloaded', 0) for r in successful_results)
        total_submissions = sum(r.get('total_submissions', 0) for r in successful_results)
        total_media_submissions = sum(r.get('media_submissions', 0) for r in successful_results)
        total_size = sum(r.get('total_size_bytes', 0) for r in successful_results)
        total_size_gb = total_size / (1024**3)

        # Calculate overall success rate
        overall_success_rate = (total_media_downloaded / total_media_submissions) * 100 if total_media_submissions > 0 else 0

        elapsed = time.time() - start_time

        # Aggregate failure reasons across all subreddits
        all_failure_reasons = {}
        for result in successful_results:
            for reason, count in result.get('failure_reasons', {}).items():
                all_failure_reasons[reason] = all_failure_reasons.get(reason, 0) + count

        # Create summary statistics
        summary = {
            'summary': {
                'total_subreddits_processed': len(qualified_subreddits),
                'successful_subreddits': len(successful_results),
                'failed_subreddits': len(failed_results),
                'total_submissions_processed': total_submissions,
                'total_media_submissions': total_media_submissions,
                'total_media_downloaded': total_media_downloaded,
                'overall_success_rate': overall_success_rate,
                'total_size_bytes': total_size,
                'total_size_gb': total_size_gb,
                'processing_time_seconds': elapsed,
                'downloads_per_second': total_media_downloaded / elapsed if elapsed > 0 else 0,
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'failure_reasons': all_failure_reasons
            },
            'subreddit_stats': rank_by_score(
                [
                    {
                        'subreddit': r['subreddit'],
                        'media_downloaded': r.get('media_downloaded', 0),
                        'media_submissions': r.get('media_submissions', 0),
                        'success_rate': r.get('success_rate', 0),
                        'size_gb': r.get('total_size_bytes', 0) / (1024**3),
                        'processing_time': r.get('processing_time', 0)
                    }
                    for r in successful_results
                ],
                'media_downloaded',
                ascending=False  # Higher download count = better rank
            ),
            'failed_subreddits': [
                {
                    'subreddit': r['subreddit'],
                    'error': r.get('error', 'Unknown error')
                }
                for r in failed_results
            ]
        }

        # Save summary
        summary_file = os.path.join(PATHS['data'], 'stage8_media_collection_stats.json')
        write_json_file(summary, summary_file)

        # Log comprehensive summary
        summary_stats = {
            'processed_subreddits': f"{len(successful_results)}/{len(qualified_subreddits)}",
            'media_downloaded': total_media_downloaded,
            'media_submissions': total_media_submissions,
            'success_rate': f"{overall_success_rate:.1f}%",
            'total_size_gb': f"{total_size_gb:.2f} GB",
            'download_rate': f"{total_media_downloaded/elapsed:.1f} files/sec"
        }
        log_stats(logger, summary_stats, "Stage 8 Results")

        if failed_results:
            logger.warning(f"⚠️  Failed subreddits: {len(failed_results)}")
            for result in failed_results[:5]:  # Show first 5
                logger.warning(f"  {result['subreddit']}: {result.get('error', 'Unknown error')}")

        # Show top failure reasons
        if all_failure_reasons:
            logger.info("🔍 Top failure reasons:")
            sorted_failures = sorted(all_failure_reasons.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_failures[:3]:
                logger.info(f"  {reason}: {count:,} failures")

        logger.info(f"Summary saved to: {summary_file}")
        log_stage_end(logger, 8, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 8 execution")
        log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())