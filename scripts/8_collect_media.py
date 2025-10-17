#!/usr/bin/env python3
"""
Stage 8: Collect Media for Submissions

Downloads media files for submissions using priority-based collection strategy.
Submission-centric approach: process each submission once, track detailed statistics.

Priority Hierarchy (early stopping):
1. media_metadata - Gallery/inline images (original source, 1-N items)
2. url field - Direct image posts (original source, 1 item)
3. oembed - Video thumbnails from YouTube/Vimeo (original source, 1 item)
4. preview - Reddit's cached images (fallback, 1 item)

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
from typing import Dict, List, Any, Tuple
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
# Significant Helper Functions
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
    filtered = items if not filter_status else [i for i in items if i.get('status') == filter_status]
    counts = defaultdict(int)
    for item in filtered:
        value = item.get(field)
        if value:
            counts[value] += 1
    return dict(counts)


def count_where(items: List[Dict], field: str, value: Any, filter_status: str = None) -> int:
    """Count items where field equals value, optionally filtered by status."""
    filtered = items if not filter_status else [i for i in items if i.get('status') == filter_status]
    return sum(1 for item in filtered if item.get(field) == value)


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
    if '404' in error_msg or 'not found' in lower:
        return '404 Not Found'
    elif '403' in error_msg or 'forbidden' in lower:
        return '403 Forbidden'
    elif 'timeout' in lower:
        return 'Timeout'
    elif 'connection' in lower:
        return 'Connection Error'
    elif 'invalid' in lower:
        return 'Invalid file type'
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
        {'success': bool, 'file_size': int, 'error': str}
    """
    try:
        # Download
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Check Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        if content_type and not any(t in content_type for t in ['image/', 'video/mp4', 'application/octet-stream']):
            return {'success': False, 'error': f'Invalid content type: {content_type}'}

        # Write file
        ensure_directory(output_path)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size = os.path.getsize(output_path)

        # Validate with file command
        try:
            file_type = subprocess.check_output(['file', '--brief', '--mime-type', output_path],
                                               universal_newlines=True).strip()
            if not (file_type.startswith('image/') or file_type == 'video/mp4'):
                os.remove(output_path)
                return {'success': False, 'error': f'Invalid file type: {file_type}'}
        except subprocess.CalledProcessError:
            pass  # Keep file if validation fails but Content-Type was good

        return {'success': True, 'file_size': file_size}

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
    if url:
        domain = urllib.parse.urlparse(url).netloc.lower()
        video_domains = ['v.redd.it', 'youtube.com', 'youtu.be', 'vimeo.com',
                         'streamable.com', 'twitch.tv']
        if any(vd in domain for vd in video_domains):
            return True

    if submission.get('media_metadata'):
        for media_info in submission['media_metadata'].values():
            if media_info.get('e') in ['Video', 'RedditVideo']:
                return True

    return False


def extract_media_metadata_urls(submission: Dict) -> List[Dict]:
    """Extract URLs from media_metadata field (galleries)."""
    urls = []
    if not submission.get('media_metadata'):
        return urls

    for idx, (media_id, media_info) in enumerate(submission['media_metadata'].items()):
        media_type = media_info.get('e')

        if media_type == 'Image' and 's' in media_info and 'u' in media_info['s']:
            url = media_info['s']['u'].replace('&amp;', '&')
            extension = url.split('.')[-1].lower()
            if extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                extension = 'jpg'
            urls.append({
                'url': url,
                'media_id': media_id,
                'source': 'media_metadata',
                'extension': f'.{extension}',
                'index': idx
            })

        elif media_type == 'AnimatedImage' and 's' in media_info and 'mp4' in media_info['s']:
            url = media_info['s']['mp4'].replace('&amp;', '&')
            urls.append({
                'url': url,
                'media_id': media_id,
                'source': 'media_metadata',
                'extension': '.mp4',
                'index': idx
            })

    return urls


def extract_url_field(submission: Dict) -> List[Dict]:
    """Extract URL from direct url field."""
    url = submission.get('url', '')
    if not url:
        return []

    # Check if video domain (skip)
    domain = urllib.parse.urlparse(url).netloc.lower()
    video_domains = ['v.redd.it', 'youtube.com', 'youtu.be', 'vimeo.com',
                     'streamable.com', 'twitch.tv']
    if any(vd in domain for vd in video_domains):
        return []

    # Check if direct media URL
    known_media_domains = ['i.redd.it', 'i.imgur.com', 'imgur.com', 'giphy.com',
                           'gfycat.com', 'preview.redd.it', 'external-preview.redd.it']
    path = urllib.parse.urlparse(url).path.lower()
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']

    # Skip reddit.com links (galleries)
    is_reddit_link = 'reddit.com' in domain and 'i.redd.it' not in domain

    is_media = (any(d in domain for d in known_media_domains) or
               any(path.endswith(ext) for ext in image_extensions))

    if is_media and not is_reddit_link:
        url = url.replace('&amp;', '&')
        extension = path.split('.')[-1].lower()
        if extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            extension = 'jpg'
        return [{
            'url': url,
            'media_id': 'direct',
            'source': 'url',
            'extension': f'.{extension}'
        }]

    return []


def extract_oembed_url(submission: Dict) -> List[Dict]:
    """Extract oembed thumbnail URL."""
    media = submission.get('media') or submission.get('secure_media')
    if media and 'oembed' in media and media['oembed'].get('thumbnail_url'):
        url = media['oembed']['thumbnail_url'].replace('&amp;', '&')
        extension = url.split('.')[-1].lower()
        if extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            extension = 'jpg'
        return [{
            'url': url,
            'media_id': 'oembed',
            'source': 'oembed',
            'extension': f'.{extension}'
        }]
    return []


def extract_preview_url(submission: Dict) -> List[Dict]:
    """Extract preview URL (fallback for link posts only)."""
    if not submission.get('preview'):
        return []

    # Only use preview for link posts (not text/self posts)
    # Text posts with images use media_metadata (Priority 1)
    if submission.get('is_self'):
        return []

    try:
        preview = submission['preview']
        if 'images' in preview and preview['images']:
            img = preview['images'][0]
            if 'source' in img and img['source'].get('url'):
                url = img['source']['url'].replace('&amp;', '&')
                extension = url.split('.')[-1].lower()
                if extension not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    extension = 'jpg'
                return [{
                    'url': url,
                    'media_id': 'preview',
                    'source': 'preview',
                    'extension': f'.{extension}'
                }]
    except Exception:
        pass

    return []


def extract_download_urls(submission: Dict) -> Tuple[List[Dict], str]:
    """
    Extract downloadable URLs using priority hierarchy.

    Returns:
        (urls, source) - List of URL dicts and source name
    """
    # Priority 1: media_metadata (galleries)
    urls = extract_media_metadata_urls(submission)
    if urls:
        return urls, 'media_metadata'

    # Priority 2: direct URL
    urls = extract_url_field(submission)
    if urls:
        return urls, 'url'

    # Priority 3: oembed thumbnail
    urls = extract_oembed_url(submission)
    if urls:
        return urls, 'oembed'

    # Priority 4: preview (fallback)
    urls = extract_preview_url(submission)
    if urls:
        return urls, 'preview'

    return [], None


# ============================================================================
# Submission Processing
# ============================================================================

def download_submission_media(submission: Dict, media_dir: str, session: requests.Session) -> Dict[str, Any]:
    """
    Download all media for a single submission.

    Returns:
        Result dict with status, counts, and metadata
    """
    submission_id = submission.get('id', 'unknown')

    # Check NSFW
    if submission.get('over_18') or submission.get('over18'):
        return {
            'submission_id': submission_id,
            'status': 'skipped_nsfw',
            'files_downloaded': 0,
            'total_size': 0,
            'source': None,
            'has_multiple': False,
            'is_video': False,
            'errors': []
        }

    # Check crosspost
    if submission.get('crosspost_parent_list') or submission.get('crosspost_parent'):
        return {
            'submission_id': submission_id,
            'status': 'skipped_crosspost',
            'files_downloaded': 0,
            'total_size': 0,
            'source': None,
            'has_multiple': False,
            'is_video': False,
            'errors': []
        }

    # Skip self posts that only contain a URL (no real text content)
    if submission.get('is_self'):
        selftext = submission.get('selftext', '').strip()
        # Check if the entire selftext is just a URL
        if selftext:
            try:
                parsed = urllib.parse.urlparse(selftext)
                # If it parses as a valid URL and the whole text is the URL
                if parsed.scheme and parsed.netloc and selftext == urllib.parse.urlunparse(parsed):
                    return {
                        'submission_id': submission_id,
                        'status': 'skipped_url_only_selfpost',
                        'files_downloaded': 0,
                        'total_size': 0,
                        'source': None,
                        'has_multiple': False,
                        'is_video': False,
                        'errors': []
                    }
            except:
                pass

    # Extract URLs
    urls, source = extract_download_urls(submission)
    is_video = is_video_submission(submission)
    has_multiple = len(urls) > 1

    if not urls:
        return {
            'submission_id': submission_id,
            'status': 'no_media',
            'files_downloaded': 0,
            'total_size': 0,
            'source': None,
            'has_multiple': False,
            'is_video': is_video,
            'errors': []
        }

    # Download each URL
    successful = 0
    failed = 0
    total_size = 0
    errors = []

    for url_info in urls:
        url = url_info['url']
        media_id = url_info['media_id']
        extension = url_info['extension']
        media_source = url_info['source']

        # Generate filename with source suffix
        if media_id in ['direct', 'oembed', 'preview']:
            filename = f"{submission_id}_{media_source}{extension}"
        else:
            # Gallery item: include index for ordering
            idx = url_info.get('index', 0)
            safe_id = media_id.replace('|', '_').replace('/', '_')[:50]
            filename = f"{submission_id}_{idx}_{safe_id}_{media_source}{extension}"

        output_path = os.path.join(media_dir, filename)

        # Skip if cached
        if os.path.exists(output_path):
            successful += 1
            total_size += os.path.getsize(output_path)
            continue

        # Download
        result = download_file(url, output_path, session)
        if result['success']:
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
        # Load all submissions
        submissions = []
        for line in read_zst_lines(submissions_file):
            if line.strip():
                submissions.append(json_loads(line))

        logger.info(f"  📊 Loaded {len(submissions)} submissions")

        # Download media for each submission
        session = create_session()
        results = []

        for submission in submissions:
            result = download_submission_media(submission, media_dir, session)
            results.append(result)

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
        logger.info(f"  ✅ Processed {len(submissions)} submissions, "
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

def format_subreddit_stats(subreddit_result: Dict) -> Dict:
    """Format statistics for a single subreddit."""
    results = subreddit_result.get('results', [])
    total_subs = len(results)

    # Count by status
    status_counts = count_by_field(results, 'status')

    # Count by source (complete only)
    source_counts = count_by_field(results, 'source', filter_status='complete')

    # Count characteristics
    complete_count = status_counts.get('complete', 0)
    multiple = count_where(results, 'has_multiple', True, filter_status='complete')
    videos = count_where(results, 'is_video', True)

    return {
        'subreddit': subreddit_result['subreddit'],
        'total_submissions': total_subs,
        'total_files_downloaded': subreddit_result.get('total_files', 0),
        'size_gb': subreddit_result.get('total_size', 0) / (1024**3),
        'status_breakdown': with_percentages(status_counts, total_subs),
        'media_sources': with_percentages(source_counts, complete_count),
        'media_characteristics': {
            'has_multiple_media': {
                'count': multiple,
                'percentage': round(100 * multiple / complete_count, 2) if complete_count > 0 else 0.0
            },
            'is_video_submission': {
                'count': videos,
                'percentage': round(100 * videos / total_subs, 2) if total_subs > 0 else 0.0
            }
        },
        'processing_time': subreddit_result.get('processing_time', 0)
    }


def format_global_stats(valid_results: List[Dict]) -> Dict:
    """Format global statistics from all subreddits."""
    # Flatten all results
    all_results = []
    for sr in valid_results:
        all_results.extend(sr.get('results', []))

    total_subs = len(all_results)
    total_files = sum(sr.get('total_files', 0) for sr in valid_results)
    total_size = sum(sr.get('total_size', 0) for sr in valid_results)

    # Count by status
    status_counts = count_by_field(all_results, 'status')

    # Count by source (complete only)
    source_counts = count_by_field(all_results, 'source', filter_status='complete')

    # Count characteristics
    complete_count = status_counts.get('complete', 0)
    multiple = count_where(all_results, 'has_multiple', True, filter_status='complete')
    videos = count_where(all_results, 'is_video', True)

    # Aggregate errors
    global_errors = aggregate_nested(valid_results, 'error_reasons')

    return {
        'total_submissions': total_subs,
        'total_files_downloaded': total_files,
        'total_size_bytes': total_size,
        'total_size_gb': total_size / (1024**3),
        'status_breakdown': with_percentages(status_counts, total_subs),
        'media_sources': with_percentages(source_counts, complete_count),
        'media_characteristics': {
            'has_multiple_media': {
                'count': multiple,
                'percentage': round(100 * multiple / complete_count, 2) if complete_count > 0 else 0.0
            },
            'is_video_submission': {
                'count': videos,
                'percentage': round(100 * videos / total_subs, 2) if total_subs > 0 else 0.0
            }
        },
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
