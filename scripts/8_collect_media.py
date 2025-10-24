#!/usr/bin/env python3
"""
Stage 8: Collect Media for Submissions

Downloads media files for submissions using priority-based collection strategy.
Submission-centric approach: process each submission once, track detailed statistics.

Priority Hierarchy (early stopping):
1. media_metadata - Gallery/inline images (original source, 1-N items)
2. url field - Direct image posts (original source, 1 item)
3. oembed - Video thumbnails from YouTube/Vimeo (original source, 1 item)
4. preview - Reddit's cached images (fallback, 1 item, first only)

Input:
- submissions/{subreddit}_submissions.zst (from Stage 7)

Output:
- media/{subreddit}/{submission_id}_{media_id}.{ext} (downloaded media files)
- data/stage8_media_collection_stats.json (detailed statistics)
- data/stage8_successful_submission_ids.json (for Stage 9 filtering)
"""

import sys
import os
import time
import urllib.parse
import requests
import subprocess
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, PROCESSES
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_stats
from utils.files import (read_zst_lines, json_loads, write_json_file,
                        process_files_parallel, ensure_directory,
                        load_qualified_subreddits_from_stage6)

# User agent to avoid blocks
USER_AGENT = "reddit_research_media_collector/1.0"

# ============================================================================
# Constants
# ============================================================================

# Video platforms to skip (not downloadable image content)
VIDEO_DOMAINS = frozenset([
    'v.redd.it', 'youtube.com', 'youtu.be', 'vimeo.com',
    'streamable.com', 'twitch.tv', 'clips.twitch.tv',
    'tiktok.com', 'instagram.com', 'dailymotion.com'
])

# Extension-less media hosts (need domain check for URLs like imgur.com/abc123)
EXTENSIONLESS_MEDIA_HOSTS = frozenset([
    'imgur.com', 'i.imgur.com', 'giphy.com', 'gfycat.com'
])

# Valid image extensions
IMAGE_EXTENSIONS = frozenset(['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'])

# Valid content types for downloaded files
VALID_CONTENT_TYPES = frozenset([
    'image/jpeg', 'image/png', 'image/gif', 'image/webp',
    'image/bmp', 'video/mp4', 'application/octet-stream'
])

# Content-Type to extension mapping (source of truth for extensions)
CONTENT_TYPE_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'video/mp4': '.mp4',
    'application/octet-stream': '.jpg'  # Fallback for unspecified binary
}


# ============================================================================
# URL Helper Functions
# ============================================================================

def extract_extension_from_url(url: str) -> Optional[str]:
    """
    Extract file extension from URL, handling query parameters.
    Returns lowercase extension without dot, or None if not found.

    Note: Returns None instead of guessing - let download_file determine
    the actual extension from Content-Type header.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()

    # Get extension from path (not query string)
    if '.' in path:
        ext = path.split('.')[-1]
        # Validate it's actually an image extension
        if ext in IMAGE_EXTENSIONS:
            return ext

    return None  # Let download_file determine from Content-Type


def is_video_domain(url: str) -> bool:
    """Check if URL is from a known video platform."""
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        return any(vd in domain for vd in VIDEO_DOMAINS)
    except (ValueError, AttributeError):
        return False


def is_likely_media_url(url: str) -> bool:
    """
    Check if URL likely points to downloadable media.
    Uses extension-first approach with domain fallback.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        # Skip video domains
        if is_video_domain(url):
            return False

        # Skip reddit.com gallery links (handled by media_metadata)
        if 'reddit.com/gallery/' in url or ('reddit.com' in domain and 'i.redd.it' not in domain):
            return False

        # Check extension first (most reliable)
        if '.' in path:
            ext = path.split('.')[-1]
            if ext in IMAGE_EXTENSIONS:
                return True

        # Fallback: check if it's a known extension-less host
        return any(host in domain for host in EXTENSIONLESS_MEDIA_HOSTS)

    except (ValueError, AttributeError):
        return False


def sanitize_media_id(media_id: str, max_length: int = 50) -> str:
    """Sanitize media ID for use in filenames."""
    return media_id.replace('|', '_').replace('/', '_').replace('\\', '_')[:max_length]


# ============================================================================
# Statistics Helper Functions
# ============================================================================

def with_percentages(counts: Dict[str, int], total: int) -> Dict[str, Dict]:
    """Add percentages to count dictionary."""
    return {
        k: {
            'count': v,
            'percentage': round(100 * v / total, 2) if total > 0 else 0.0
        }
        for k, v in counts.items()
    }


def count_by_field(items: List[Dict], field: str, filter_status: str = None) -> Dict[str, int]:
    """Count occurrences of field values, optionally filtered by status."""
    filtered = [i for i in items if not filter_status or i.get('status') == filter_status]
    counts = defaultdict(int)
    for item in filtered:
        if value := item.get(field):
            counts[value] += 1
    return dict(counts)


def count_where(items: List[Dict], field: str, value: Any, filter_status: str = None) -> int:
    """Count items where field equals value, optionally filtered by status."""
    return sum(1 for i in items if (not filter_status or i.get('status') == filter_status)
               and i.get(field) == value)


def aggregate_nested(results: List[Dict], key: str) -> Dict[str, int]:
    """Aggregate counts from nested dictionaries in results."""
    aggregated = defaultdict(int)
    for result in results:
        for item, count in result.get(key, {}).items():
            aggregated[item] += count
    return dict(aggregated)


def categorize_error(error_msg: str) -> str:
    """Categorize error message into standard types."""
    lower = error_msg.lower()

    # HTTP status codes
    if '404' in error_msg or 'not found' in lower:
        return '404 Not Found'
    elif '403' in error_msg or 'forbidden' in lower:
        return '403 Forbidden'
    elif '429' in error_msg or 'too many requests' in lower:
        return '429 Rate Limited'
    elif '500' in error_msg or '502' in error_msg or '503' in error_msg:
        return '5xx Server Error'

    # Network errors
    elif 'timeout' in lower:
        return 'Timeout'
    elif 'connection' in lower or 'connectionerror' in lower:
        return 'Connection Error'
    elif 'ssl' in lower or 'certificate' in lower:
        return 'SSL/Certificate Error'

    # Content validation errors
    elif 'invalid content type' in lower:
        return 'Invalid Content-Type'
    elif 'invalid file type' in lower:
        return 'Invalid File Type (validation)'

    # Generic fallback
    else:
        return error_msg[:100]


# ============================================================================
# Download Functions
# ============================================================================

def create_session() -> requests.Session:
    """Create a robust requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def download_file(url: str, output_path: str, session: requests.Session) -> Dict[str, Any]:
    """
    Download and validate file.

    Returns:
        {'success': bool, 'file_size': int, 'actual_extension': str, 'error': str}
    """
    try:
        # Download
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Determine actual extension from Content-Type (source of truth)
        content_type = response.headers.get('Content-Type', '').lower()
        actual_extension = None

        if content_type:
            # Split to handle "image/jpeg; charset=utf-8" style headers
            base_type = content_type.split(';')[0].strip()

            # Validate Content-Type
            if base_type not in VALID_CONTENT_TYPES and not base_type.startswith('image/'):
                return {'success': False, 'error': f'Invalid Content-Type: {base_type}'}

            # Get actual extension from Content-Type
            actual_extension = CONTENT_TYPE_TO_EXT.get(base_type)

        # Write file (will be renamed later with correct extension)
        ensure_directory(output_path)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(output_path)

        # Validate with file command
        try:
            file_type = subprocess.check_output(
                ['file', '--brief', '--mime-type', output_path],
                stderr=subprocess.DEVNULL,
                universal_newlines=True
            ).strip()

            # Accept images and video/mp4 (for animated content)
            if not (file_type.startswith('image/') or file_type == 'video/mp4'):
                os.remove(output_path)
                return {'success': False, 'error': f'Invalid File Type (validation): {file_type}'}

            # If Content-Type didn't give us extension, use file command result
            if not actual_extension:
                actual_extension = CONTENT_TYPE_TO_EXT.get(file_type, '.jpg')

        except (subprocess.CalledProcessError, FileNotFoundError):
            # If file command fails, use Content-Type or fallback
            if not actual_extension:
                actual_extension = '.jpg'

        return {
            'success': True,
            'file_size': file_size,
            'actual_extension': actual_extension
        }

    except requests.exceptions.Timeout:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': 'Timeout'}

    except requests.exceptions.ConnectionError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': f'Connection Error: {str(e)[:50]}'}

    except requests.exceptions.HTTPError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': str(e)}

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': str(e)}


# ============================================================================
# Media Extraction Functions
# ============================================================================

def is_video_submission(submission: Dict) -> bool:
    """Check if submission is a video submission."""
    if submission.get('is_video'):
        return True

    url = submission.get('url', '')
    if url and is_video_domain(url):
        return True

    if submission.get('media_metadata'):
        for media_info in submission['media_metadata'].values():
            if media_info.get('e') in ['Video', 'RedditVideo']:
                return True

    return False


def make_media_item(url: str, media_id: str, source: str, index: int = None) -> Dict:
    """Create standardized media item dict with optional extension hint."""
    url = url.replace('&amp;', '&')
    ext = extract_extension_from_url(url)
    item = {
        'url': url,
        'media_id': media_id,
        'source': source,
        'extension_hint': f'.{ext}' if ext else None  # Hint from URL, not truth
    }
    if index is not None:
        item['index'] = index
    return item


def extract_media_metadata_urls(submission: Dict) -> List[Dict]:
    """Extract URLs from media_metadata field (galleries)."""
    media_metadata = submission.get('media_metadata')
    if not media_metadata:
        return []

    urls = []
    for idx, (media_id, info) in enumerate(media_metadata.items()):
        media_type = info.get('e')

        if media_type == 'Image' and 's' in info and 'u' in info['s']:
            urls.append(make_media_item(info['s']['u'], media_id, 'media_metadata', idx))
        elif media_type == 'AnimatedImage' and 's' in info and 'mp4' in info['s']:
            item = make_media_item(info['s']['mp4'], media_id, 'media_metadata', idx)
            item['extension_hint'] = '.mp4'  # AnimatedImage is always mp4
            urls.append(item)

    return urls


def extract_url_field(submission: Dict) -> List[Dict]:
    """Extract URL from direct url field using smart detection."""
    url = submission.get('url', '')
    return [make_media_item(url, 'direct', 'url')] if url and is_likely_media_url(url) else []


def extract_oembed_url(submission: Dict) -> List[Dict]:
    """Extract oembed thumbnail URL."""
    media = submission.get('media') or submission.get('secure_media')
    if media and 'oembed' in media and media['oembed'].get('thumbnail_url'):
        return [make_media_item(media['oembed']['thumbnail_url'], 'oembed', 'oembed')]
    return []


def extract_preview_url(submission: Dict) -> List[Dict]:
    """Extract preview URL (fallback for link posts only, first image only)."""
    preview = submission.get('preview')
    if not preview or submission.get('is_self'):
        return []

    try:
        img = preview['images'][0]
        url = img['source'].get('url')
        return [make_media_item(url, 'preview', 'preview')] if url else []
    except (KeyError, TypeError, IndexError):
        return []


def extract_download_urls(submission: Dict) -> Tuple[List[Dict], Optional[str]]:
    """Extract downloadable URLs using priority hierarchy (early stopping)."""
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
# Submission Processing
# ============================================================================

def make_skip_result(submission_id: str, status: str, is_video: bool = False) -> Dict[str, Any]:
    """Create standardized skip result dict."""
    return {
        'submission_id': submission_id,
        'status': status,
        'files_downloaded': 0,
        'total_size': 0,
        'source': None,
        'has_multiple': False,
        'is_video': is_video,
        'errors': []
    }


def download_submission_media(submission: Dict, media_dir: str, session: requests.Session) -> Dict[str, Any]:
    """
    Download all media for a single submission.

    Returns:
        Result dict with status, counts, and metadata
    """
    submission_id = submission.get('id', 'unknown')

    # Check NSFW
    if submission.get('over_18') or submission.get('over18'):
        return make_skip_result(submission_id, 'skipped_nsfw')

    # Check crosspost
    if submission.get('crosspost_parent_list') or submission.get('crosspost_parent'):
        return make_skip_result(submission_id, 'skipped_crosspost')

    # Skip self posts that only contain a URL (no real text content)
    if submission.get('is_self'):
        selftext = submission.get('selftext', '').strip()
        if selftext and len(selftext) < 500 and selftext.count('\n') == 0:
            try:
                parsed = urllib.parse.urlparse(selftext)
                if parsed.scheme and parsed.netloc:
                    return make_skip_result(submission_id, 'skipped_url_only_selfpost')
            except (ValueError, AttributeError):
                pass

    # Extract URLs
    urls, source = extract_download_urls(submission)
    is_video = is_video_submission(submission)
    has_multiple = len(urls) > 1

    if not urls:
        return make_skip_result(submission_id, 'no_media', is_video)

    # Download each URL
    successful = 0
    failed = 0
    total_size = 0
    errors = []

    for url_info in urls:
        url = url_info['url']
        media_id = url_info['media_id']
        extension_hint = url_info.get('extension_hint')
        media_source = url_info['source']

        # Generate base filename (without extension)
        if media_id in ['direct', 'oembed', 'preview']:
            filename_base = f"{submission_id}_{media_source}"
        else:
            # Gallery item: include index for ordering
            idx = url_info.get('index', 0)
            safe_id = sanitize_media_id(media_id)
            filename_base = f"{submission_id}_{idx}_{safe_id}_{media_source}"

        # Check if already cached (try with hint extension or common extensions)
        cached_path = None
        if extension_hint:
            potential_path = os.path.join(media_dir, f"{filename_base}{extension_hint}")
            if os.path.exists(potential_path):
                cached_path = potential_path

        if cached_path:
            successful += 1
            total_size += os.path.getsize(cached_path)
            continue

        # Download to temporary path first
        temp_path = os.path.join(media_dir, f"{filename_base}.tmp")
        result = download_file(url, temp_path, session)

        if result['success']:
            # Rename to final path with actual extension
            actual_ext = result['actual_extension']
            final_path = os.path.join(media_dir, f"{filename_base}{actual_ext}")

            # Rename temp file to final filename with correct extension
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)

            successful += 1
            total_size += result['file_size']
        else:
            failed += 1
            errors.append(result['error'])

    # Determine status (strict: all or nothing)
    expected = len(urls)
    if successful == expected and failed == 0:
        status = 'complete'
    elif successful > 0 and failed > 0:
        status = 'partial'
    else:
        status = 'failed'

    return {
        'submission_id': submission_id,
        'status': status,
        'files_downloaded': successful,
        'total_size': total_size,
        'source': source,
        'has_multiple': has_multiple,
        'is_video': is_video,
        'errors': errors
    }


# ============================================================================
# Subreddit Processing
# ============================================================================

def process_subreddit(args: Tuple) -> Dict[str, Any]:
    """Process all submissions for a subreddit."""
    subreddit, = args

    logger = get_stage_logger(8, "collect_media", worker_identifier=f"subreddits/{subreddit}")
    logger.info(f"🔍 Processing r/{subreddit}")

    start_time = time.time()

    # Load submissions
    submissions_file = os.path.join(PATHS['submissions'], f"{subreddit}_submissions.zst")
    media_dir = os.path.join(PATHS['media'], subreddit)

    if not os.path.exists(submissions_file):
        logger.warning(f"  ⚠️  Submissions file not found")
        return {'subreddit': subreddit, 'error': 'submissions_file_not_found'}

    try:
        # Stream and process submissions (memory-efficient)
        session = create_session()
        results = []
        submission_count = 0

        for line in read_zst_lines(submissions_file):
            if line.strip():
                submission = json_loads(line)
                result = download_submission_media(submission, media_dir, session)
                results.append(result)
                submission_count += 1

        logger.info(f"  📊 Processed {submission_count} submissions")

        # Aggregate stats
        total_files = sum(r['files_downloaded'] for r in results)
        total_size = sum(r['total_size'] for r in results)

        # Collect successful submission IDs (complete + no_media)
        successful_ids = [r['submission_id'] for r in results
                         if r['status'] in ['complete', 'no_media']]

        # Aggregate error reasons
        error_reasons = defaultdict(int)
        for r in results:
            if r['status'] in ['failed', 'partial']:
                for error in r.get('errors', []):
                    error_reasons[categorize_error(error)] += 1

        elapsed = time.time() - start_time
        logger.info(f"  ✅ Processed {submission_count} submissions, "
                   f"downloaded {total_files} files in {elapsed:.1f}s")

        return {
            'subreddit': subreddit,
            'results': results,
            'total_files': total_files,
            'total_size': total_size,
            'successful_ids': successful_ids,
            'error_reasons': dict(error_reasons),
            'processing_time': elapsed
        }

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        return {'subreddit': subreddit, 'error': str(e)}


# ============================================================================
# Statistics Formatting
# ============================================================================

def build_media_characteristics(results: List[Dict], total_subs: int, complete_count: int) -> Dict:
    """Build media characteristics stats dict."""
    multiple = count_where(results, 'has_multiple', True, filter_status='complete')
    videos = count_where(results, 'is_video', True)
    return {
        'has_multiple_media': {
            'count': multiple,
            'percentage': round(100 * multiple / complete_count, 2) if complete_count > 0 else 0.0
        },
        'is_video_submission': {
            'count': videos,
            'percentage': round(100 * videos / total_subs, 2) if total_subs > 0 else 0.0
        }
    }


def format_subreddit_stats(subreddit_result: Dict) -> Dict:
    """Format statistics for a single subreddit."""
    results = subreddit_result.get('results', [])
    total_subs = len(results)
    status_counts = count_by_field(results, 'status')
    complete_count = status_counts.get('complete', 0)
    source_counts = count_by_field(results, 'source', filter_status='complete')

    return {
        'subreddit': subreddit_result['subreddit'],
        'total_submissions': total_subs,
        'total_files_downloaded': subreddit_result.get('total_files', 0),
        'size_gb': subreddit_result.get('total_size', 0) / (1024**3),
        'status_breakdown': with_percentages(status_counts, total_subs),
        'media_sources': with_percentages(source_counts, complete_count),
        'media_characteristics': build_media_characteristics(results, total_subs, complete_count),
        'processing_time': subreddit_result.get('processing_time', 0)
    }


def format_global_stats(valid_results: List[Dict]) -> Dict:
    """Format global statistics from all subreddits."""
    all_results = [r for sr in valid_results for r in sr.get('results', [])]
    total_subs = len(all_results)
    total_files = sum(sr.get('total_files', 0) for sr in valid_results)
    total_size = sum(sr.get('total_size', 0) for sr in valid_results)

    status_counts = count_by_field(all_results, 'status')
    complete_count = status_counts.get('complete', 0)
    source_counts = count_by_field(all_results, 'source', filter_status='complete')
    global_errors = aggregate_nested(valid_results, 'error_reasons')

    return {
        'total_submissions': total_subs,
        'total_files_downloaded': total_files,
        'total_size_bytes': total_size,
        'total_size_gb': total_size / (1024**3),
        'status_breakdown': with_percentages(status_counts, total_subs),
        'media_sources': with_percentages(source_counts, complete_count),
        'media_characteristics': build_media_characteristics(all_results, total_subs, complete_count),
        'error_reasons': dict(sorted(global_errors.items(), key=lambda x: x[1], reverse=True)[:10])
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution function."""
    logger = get_stage_logger(8, "collect_media")
    log_stage_start(logger, 8, "Collect Media for Submissions")

    start_time = time.time()

    try:
        # Get qualified subreddits
        qualified_stats = load_qualified_subreddits_from_stage6(logger)
        if not qualified_stats:
            logger.error("❌ No qualified subreddits found!")
            log_stage_end(logger, 8, success=False, elapsed_time=time.time() - start_time)
            return 1

        subreddits = [s['subreddit'] for s in qualified_stats]
        logger.info(f"Processing {len(subreddits)} subreddits")
        logger.info(f"Using {PROCESSES} parallel processes")

        # Process in parallel
        args = [(s,) for s in subreddits]
        results = process_files_parallel(args, process_subreddit, PROCESSES, logger)

        logger.info("Aggregating statistics...")

        # Separate valid and error results
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]

        # Format statistics
        global_stats = format_global_stats(valid_results)
        per_subreddit_stats = [format_subreddit_stats(r) for r in valid_results]

        elapsed = time.time() - start_time

        # Create summary
        summary = {
            'summary': {
                **global_stats,
                'total_subreddits': len(subreddits),
                'successful_subreddits': len(valid_results),
                'failed_subreddits': len(error_results),
                'processing_time_seconds': elapsed,
                'files_per_second': global_stats['total_files_downloaded'] / elapsed if elapsed > 0 else 0,
                'collection_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'per_subreddit_summary': per_subreddit_stats,
            'failed_subreddits': [{'subreddit': r['subreddit'], 'error': r['error']}
                                 for r in error_results]
        }

        # Save statistics
        output_file = os.path.join(PATHS['data'], 'stage8_media_collection_stats.json')
        write_json_file(summary, output_file, pretty=True)

        # Log summary
        log_stats(logger, {
            'subreddits_processed': f"{len(valid_results)}/{len(subreddits)}",
            'total_submissions': global_stats['total_submissions'],
            'files_downloaded': global_stats['total_files_downloaded'],
            'size_gb': f"{global_stats['total_size_gb']:.2f} GB",
            'media_downloaded': global_stats['status_breakdown']['complete']['count'],
            'no_media': global_stats['status_breakdown'].get('no_media', {}).get('count', 0),
            'partial_media': global_stats['status_breakdown'].get('partial', {}).get('count', 0),
            'failed_media': global_stats['status_breakdown'].get('failed', {}).get('count', 0),
            'skipped_nsfw': global_stats['status_breakdown'].get('skipped_nsfw', {}).get('count', 0),
            'skipped_crosspost': global_stats['status_breakdown'].get('skipped_crosspost', {}).get('count', 0),
            'skipped_url_only_selfpost': global_stats['status_breakdown'].get('skipped_url_only_selfpost', {}).get('count', 0)
        }, "Stage 8 Results")

        # Show media sources
        logger.info("\n📊 Media Sources (for complete submissions):")
        for source, data in global_stats['media_sources'].items():
            logger.info(f"  {source}: {data['count']:,} ({data['percentage']}%)")

        # Show top errors
        if global_stats['error_reasons']:
            logger.info("\n🔍 Top failure reasons:")
            for reason, count in list(global_stats['error_reasons'].items())[:5]:
                logger.info(f"  {reason}: {count:,} failures")

        # Save successful submission IDs
        logger.info("\n📝 Creating successful submission IDs list...")
        successful_ids_by_subreddit = {r['subreddit']: r.get('successful_ids', [])
                                       for r in valid_results if r.get('successful_ids')}

        total_successful = sum(len(ids) for ids in successful_ids_by_subreddit.values())

        successful_ids_output = {
            'metadata': {
                'total_subreddits': len(successful_ids_by_subreddit),
                'total_successful_submissions': total_successful,
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'criteria': 'Submissions with status: complete or no_media ONLY'
            },
            'subreddit_submission_ids': successful_ids_by_subreddit
        }

        ids_file = os.path.join(PATHS['data'], 'stage8_successful_submission_ids.json')
        write_json_file(successful_ids_output, ids_file, pretty=True)

        logger.info(f"✅ Saved {total_successful} successful submission IDs " +
                   f"across {len(successful_ids_by_subreddit)} subreddits")
        logger.info(f"\nResults saved to: {output_file}")

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
