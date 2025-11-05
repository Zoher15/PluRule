#!/usr/bin/env python3
"""
Stage 8: Collect Media for Submissions

Downloads media files (images, thumbnails) for submissions to provide visual context.
Uses priority-based extraction with robust error handling and progress tracking.

Priority Hierarchy (early stopping):
1. media_metadata - Gallery/inline images (1-N items)
2. url field - Direct image posts (1 item)
3. oembed - Video thumbnails (1 item)
4. preview - Reddit cached images (fallback, 1 item)

Input:
- data/submissions/{subreddit}_submissions.zst (from Stage 7)

Output:
- media/{subreddit}/{submission_id}_{media_id}.{ext}
- data/stage8_media_collection_stats.json
- data/stage8_successful_submission_ids.json
"""

import sys
import os
import time
import urllib.parse
import requests
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES, MIN_TEST_THREAD_PAIRS
from utils.logging import get_stage_logger, log_stage_start, log_stage_end
from utils.files import read_zst_lines, json_loads, write_json_file, process_files_parallel, ensure_directory, read_json_file

USER_AGENT = "reddit_research_media_collector/1.0"

# Download settings
DOWNLOAD_TIMEOUT = 15  # Reduced from 30s for faster failures
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max
REQUEST_DELAY = 0.1  # Small delay between downloads to be respectful

# Media configuration
VIDEO_DOMAINS = frozenset([
    'v.redd.it', 'youtube.com', 'youtu.be', 'vimeo.com',
    'streamable.com', 'twitch.tv', 'clips.twitch.tv',
    'tiktok.com', 'instagram.com', 'dailymotion.com'
])

EXTENSIONLESS_MEDIA_HOSTS = frozenset([
    'imgur.com', 'i.imgur.com', 'giphy.com', 'gfycat.com'
])

IMAGE_EXTENSIONS = frozenset(['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'])

VALID_CONTENT_TYPES = frozenset([
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'
])

CONTENT_TYPE_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp'
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_qualified_subreddits_from_stage6(logger) -> List[Dict[str, Any]]:
    """Load subreddits with >= MIN_TEST_THREAD_PAIRS successful thread pairs."""
    summary_file = os.path.join(PATHS['data'], 'stage6_trees_and_threads_summary.json')

    if not os.path.exists(summary_file):
        logger.error(f"❌ Stage 6 summary not found: {summary_file}")
        return []

    try:
        summary = read_json_file(summary_file)
        qualified = [
            stat for stat in summary.get('subreddit_stats', [])
            if stat.get('successful_pairs', 0) >= MIN_TEST_THREAD_PAIRS
        ]
        logger.info(f"Loaded {len(qualified)} qualified subreddits (>= {MIN_TEST_THREAD_PAIRS} thread pairs)")
        return qualified
    except Exception as e:
        logger.error(f"❌ Error loading Stage 6 summary: {e}")
        return []


def extract_extension_from_url(url: str) -> Optional[str]:
    """Extract file extension from URL (without dot)."""
    try:
        path = urllib.parse.urlparse(url).path.lower()
        if '.' in path:
            ext = path.split('.')[-1]
            if ext in IMAGE_EXTENSIONS:
                return ext
    except Exception:
        pass
    return None


def is_video_domain(url: str) -> bool:
    """Check if URL is from a video platform."""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        return any(vd in domain for vd in VIDEO_DOMAINS)
    except Exception:
        return False


def is_likely_media_url(url: str) -> bool:
    """Check if URL likely points to downloadable media."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Skip videos and reddit gallery links
        if is_video_domain(url) or 'reddit.com/gallery/' in url:
            return False

        # Skip reddit.com unless it's i.redd.it
        if 'reddit.com' in domain and 'i.redd.it' not in domain:
            return False

        # Check extension first
        if '.' in path and path.split('.')[-1] in IMAGE_EXTENSIONS:
            return True

        # Check known extensionless hosts
        return any(host in domain for host in EXTENSIONLESS_MEDIA_HOSTS)

    except Exception:
        return False


def sanitize_media_id(media_id: str, max_length: int = 50) -> str:
    """Sanitize media ID for filenames."""
    return media_id.replace('|', '_').replace('/', '_').replace('\\', '_')[:max_length]


def categorize_error(error_msg: str) -> str:
    """Categorize error message."""
    lower = error_msg.lower()

    if '404' in error_msg or 'not found' in lower:
        return '404_not_found'
    elif '403' in error_msg or 'forbidden' in lower:
        return '403_forbidden'
    elif '429' in error_msg or 'too many' in lower:
        return '429_rate_limited'
    elif 'timeout' in lower:
        return 'timeout'
    elif 'connection' in lower:
        return 'connection_error'
    elif 'ssl' in lower or 'certificate' in lower:
        return 'ssl_error'
    elif 'content-type' in lower:
        return 'invalid_content_type'
    else:
        return 'other_error'


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def create_session() -> requests.Session:
    """Create requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,  # Reduced retries
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': USER_AGENT})
    return session


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file(url: str, output_path: str, session: requests.Session) -> Dict[str, Any]:
    """
    Download and validate file with strict timeout.

    Returns: {'success': bool, 'file_size': int, 'extension': str, 'error': str}
    """
    try:
        # Download with strict timeout
        response = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        # Validate Content-Type
        content_type = response.headers.get('Content-Type', '').lower().split(';')[0].strip()
        if content_type not in VALID_CONTENT_TYPES:
            return {'success': False, 'error': f'Invalid Content-Type: {content_type}'}

        # Get extension from Content-Type
        extension = CONTENT_TYPE_TO_EXT.get(content_type, '.jpg')

        # Write file with size limit
        ensure_directory(output_path)
        file_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_size += len(chunk)
                    if file_size > MAX_FILE_SIZE:
                        os.remove(output_path)
                        return {'success': False, 'error': 'File too large'}
                    f.write(chunk)

        return {
            'success': True,
            'file_size': file_size,
            'extension': extension
        }

    except requests.exceptions.Timeout:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': 'Timeout'}

    except requests.exceptions.HTTPError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': str(e)}

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': f'Download error: {str(e)[:50]}'}


# ============================================================================
# MEDIA EXTRACTION
# ============================================================================

def is_video_submission(submission: Dict) -> bool:
    """Check if submission is video content."""
    if submission.get('is_video'):
        return True

    url = submission.get('url', '')
    if url and is_video_domain(url):
        return True

    media_metadata = submission.get('media_metadata')
    if media_metadata:
        for info in media_metadata.values():
            if info.get('e') in ['Video', 'RedditVideo']:
                return True

    return False


def make_media_item(url: str, media_id: str, source: str, index: int = None) -> Dict:
    """Create media item dict."""
    url = url.replace('&amp;', '&')
    item = {
        'url': url,
        'media_id': media_id,
        'source': source,
        'extension_hint': extract_extension_from_url(url)
    }
    if index is not None:
        item['index'] = index
    return item


def extract_media_metadata_urls(submission: Dict) -> List[Dict]:
    """Extract URLs from media_metadata (galleries)."""
    media_metadata = submission.get('media_metadata', {})
    if not media_metadata:
        return []

    urls = []
    for idx, (media_id, info) in enumerate(media_metadata.items()):
        media_type = info.get('e')

        if media_type == 'Image' and 's' in info and 'u' in info['s']:
            urls.append(make_media_item(info['s']['u'], media_id, 'media_metadata', idx))

    return urls


def extract_url_field(submission: Dict) -> List[Dict]:
    """Extract URL from direct url field."""
    url = submission.get('url', '')
    if url and is_likely_media_url(url):
        return [make_media_item(url, 'direct', 'url')]
    return []


def extract_oembed_url(submission: Dict) -> List[Dict]:
    """Extract oembed thumbnail."""
    media = submission.get('media') or submission.get('secure_media')
    if media and 'oembed' in media:
        thumbnail_url = media['oembed'].get('thumbnail_url')
        if thumbnail_url:
            return [make_media_item(thumbnail_url, 'oembed', 'oembed')]
    return []


def extract_preview_url(submission: Dict) -> List[Dict]:
    """Extract preview URL (fallback)."""
    preview = submission.get('preview')
    if not preview or submission.get('is_self'):
        return []

    try:
        url = preview['images'][0]['source'].get('url')
        if url:
            return [make_media_item(url, 'preview', 'preview')]
    except (KeyError, TypeError, IndexError):
        pass

    return []


def extract_download_urls(submission: Dict) -> Tuple[List[Dict], Optional[str]]:
    """Extract URLs using priority hierarchy (early stopping)."""
    for extractor, source in [
        (extract_media_metadata_urls, 'media_metadata'),
        (extract_url_field, 'url'),
        (extract_oembed_url, 'oembed'),
        (extract_preview_url, 'preview')
    ]:
        urls = extractor(submission)
        if urls:
            return urls, source
    return [], None


# ============================================================================
# SUBMISSION PROCESSING
# ============================================================================

def download_submission_media(submission: Dict, media_dir: str, session: requests.Session) -> Dict[str, Any]:
    """Download all media for a submission."""
    submission_id = submission.get('id', 'unknown')

    # Skip NSFW
    if submission.get('over_18') or submission.get('over18'):
        return {
            'submission_id': submission_id,
            'status': 'skipped_nsfw',
            'files_downloaded': 0,
            'errors': []
        }

    # Skip crossposts
    if submission.get('crosspost_parent_list') or submission.get('crosspost_parent'):
        return {
            'submission_id': submission_id,
            'status': 'skipped_crosspost',
            'files_downloaded': 0,
            'errors': []
        }

    # Extract URLs
    urls, source = extract_download_urls(submission)
    is_video = is_video_submission(submission)

    if not urls:
        return {
            'submission_id': submission_id,
            'status': 'no_media',
            'files_downloaded': 0,
            'is_video': is_video,
            'errors': []
        }

    # Download each URL
    successful = 0
    errors = []

    for url_info in urls:
        url = url_info['url']
        media_id = url_info['media_id']
        extension_hint = url_info.get('extension_hint')

        # Generate filename
        if media_id in ['direct', 'oembed', 'preview']:
            filename_base = f"{submission_id}_{media_id}"
        else:
            idx = url_info.get('index', 0)
            safe_id = sanitize_media_id(media_id)
            filename_base = f"{submission_id}_{idx}_{safe_id}"

        # Check cache
        if extension_hint:
            cached_path = os.path.join(media_dir, f"{filename_base}.{extension_hint}")
            if os.path.exists(cached_path):
                successful += 1
                continue

        # Download
        temp_path = os.path.join(media_dir, f"{filename_base}.tmp")
        result = download_file(url, temp_path, session)

        if result['success']:
            # Rename with correct extension
            ext = result['extension']
            final_path = os.path.join(media_dir, f"{filename_base}{ext}")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            successful += 1
        else:
            errors.append(result['error'])

        # Small delay between downloads
        time.sleep(REQUEST_DELAY)

    # Determine status
    expected = len(urls)
    if successful == expected:
        status = 'complete'
    elif successful > 0:
        status = 'partial'
    else:
        status = 'failed'

    return {
        'submission_id': submission_id,
        'status': status,
        'files_downloaded': successful,
        'source': source,
        'is_video': is_video,
        'errors': errors
    }


# ============================================================================
# SUBREDDIT PROCESSING
# ============================================================================

def process_subreddit(args: Tuple) -> Dict[str, Any]:
    """Process all submissions for a subreddit with progress tracking."""
    subreddit, = args
    logger = get_stage_logger(8, "collect_media", worker_identifier=f"subreddits/{subreddit}")
    logger.info(f"🔄 Processing r/{subreddit}")

    start_time = time.time()

    submissions_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")
    media_dir = os.path.join(PATHS['media'], subreddit)

    if not os.path.exists(submissions_file):
        logger.warning(f"⚠️  Submissions file not found")
        return {'subreddit': subreddit, 'error': 'submissions_file_not_found'}

    try:
        session = create_session()

        # Process submissions with streaming aggregation
        status_counts = defaultdict(int)
        error_counts = defaultdict(int)
        successful_ids = []
        total_files = 0
        submission_count = 0

        for line in read_zst_lines(submissions_file):
            if not line.strip():
                continue

            submission = json_loads(line)
            result = download_submission_media(submission, media_dir, session)

            # Aggregate incrementally
            status_counts[result['status']] += 1
            total_files += result['files_downloaded']
            submission_count += 1

            # Track successful IDs
            if result['status'] in ['complete', 'no_media']:
                successful_ids.append(result['submission_id'])

            # Track errors
            for error in result.get('errors', []):
                error_counts[categorize_error(error)] += 1

            # Progress logging every 100 submissions
            if submission_count % 100 == 0:
                logger.info(f"  Progress: {submission_count} submissions processed, {total_files} files downloaded")

        elapsed = time.time() - start_time
        logger.info(f"✅ Processed {submission_count} submissions, downloaded {total_files} files in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'submission_count': submission_count,
            'total_files': total_files,
            'status_counts': dict(status_counts),
            'error_counts': dict(error_counts),
            'successful_ids': successful_ids,
            'processing_time': elapsed
        }

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'subreddit': subreddit, 'error': str(e)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(8, "collect_media")
    log_stage_start(logger, 8, "Collect Media for Submissions")
    start_time = time.time()

    try:
        # Load qualified subreddits
        qualified_stats = load_qualified_subreddits_from_stage6(logger)
        if not qualified_stats:
            logger.error("❌ No qualified subreddits found!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        subreddits = [s['subreddit'] for s in qualified_stats]
        logger.info(f"Processing {len(subreddits)} subreddits")

        # Cap processes for network-heavy operations
        logger.info(f"Using {PROCESSES} parallel processes")

        # Process in parallel
        args = [(s,) for s in subreddits]
        results = process_files_parallel(args, process_subreddit, PROCESSES, logger)

        # Separate valid and error results
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]

        # Aggregate statistics
        total_submissions = sum(r.get('submission_count', 0) for r in valid_results)
        total_files = sum(r.get('total_files', 0) for r in valid_results)

        # Aggregate status counts
        global_status_counts = defaultdict(int)
        for r in valid_results:
            for status, count in r.get('status_counts', {}).items():
                global_status_counts[status] += count

        # Aggregate error counts
        global_error_counts = defaultdict(int)
        for r in valid_results:
            for error, count in r.get('error_counts', {}).items():
                global_error_counts[error] += count

        # Collect successful IDs
        successful_ids_by_subreddit = {
            r['subreddit']: r.get('successful_ids', [])
            for r in valid_results
        }
        total_successful_ids = sum(len(ids) for ids in successful_ids_by_subreddit.values())

        elapsed = time.time() - start_time

        # Create summary
        summary = {
            'summary': {
                'total_subreddits': len(subreddits),
                'successful_subreddits': len(valid_results),
                'failed_subreddits': len(error_results),
                'total_submissions': total_submissions,
                'total_files_downloaded': total_files,
                'total_successful_submissions': total_successful_ids,
                'processing_time_seconds': round(elapsed, 1),
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'status_breakdown': dict(global_status_counts),
            'error_breakdown': dict(sorted(global_error_counts.items(), key=lambda x: -x[1])[:20]),
            'per_subreddit': [
                {
                    'subreddit': r['subreddit'],
                    'submissions': r.get('submission_count', 0),
                    'files_downloaded': r.get('total_files', 0),
                    'successful_submissions': len(r.get('successful_ids', [])),
                    'processing_time': round(r.get('processing_time', 0), 2)
                }
                for r in sorted(valid_results, key=lambda x: x.get('total_files', 0), reverse=True)
            ],
            'failed_subreddits': [
                {'subreddit': r['subreddit'], 'error': r.get('error', 'unknown')}
                for r in error_results
            ]
        }

        # Save statistics
        stats_file = os.path.join(PATHS['data'], 'stage8_media_collection_stats.json')
        write_json_file(summary, stats_file, pretty=True)

        # Save successful IDs
        successful_ids_output = {
            'metadata': {
                'total_subreddits': len(successful_ids_by_subreddit),
                'total_successful_submissions': total_successful_ids,
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'criteria': 'Submissions with status: complete or no_media'
            },
            'subreddit_submission_ids': successful_ids_by_subreddit
        }

        ids_file = os.path.join(PATHS['data'], 'stage8_successful_submission_ids.json')
        write_json_file(successful_ids_output, ids_file, pretty=True)

        # Log results
        logger.info(f"\n🎉 Stage 8 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Subreddits: {len(valid_results)}/{len(subreddits)}")
        logger.info(f"Submissions: {total_submissions:,}")
        logger.info(f"Files downloaded: {total_files:,}")
        logger.info(f"Successful submissions: {total_successful_ids:,}")

        logger.info(f"\n📊 Status Breakdown:")
        for status, count in sorted(global_status_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {status}: {count:,}")

        if global_error_counts:
            logger.info(f"\n⚠️  Top Errors:")
            for error, count in list(sorted(global_error_counts.items(), key=lambda x: -x[1]))[:5]:
                logger.info(f"  {error}: {count:,}")

        logger.info(f"\nResults: {stats_file}")
        logger.info(f"Successful IDs: {ids_file}")

        if error_results:
            logger.warning(f"\n❌ Failed subreddits: {len(error_results)}")
            for r in error_results[:10]:
                logger.warning(f"  {r['subreddit']}: {r.get('error', 'unknown')}")

        log_stage_end(logger, 8, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"❌ Stage 8 failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
