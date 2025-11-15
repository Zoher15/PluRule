#!/usr/bin/env python3
"""
Stage 2: Get SFW Subreddits with Minimum Mod Comments

Takes the subreddit rankings from Stage 1 and selects all SFW subreddits
that have at least MIN_MATCHED_COMMENTS mod comments. Uses Reddit API to check
NSFW status and collect subreddit metadata and community rules.

Supports multiple Reddit API keys for round-robin usage to handle rate limits.
Credentials are loaded from: credentials/reddit_api_keys.json

See credentials/reddit_api_keys.json.template for the expected format.

Input:  subreddit_mod_comment_rankings.json
Output: sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, MIN_MATCHED_COMMENTS, MIN_RULES_FOR_MATCHING, create_directories
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import read_json_file, write_json_file
from utils.reddit import clean_rule_text

# Try to import Reddit API components
try:
    import praw
    from tqdm import tqdm
    REDDIT_API_AVAILABLE = True
except ImportError:
    REDDIT_API_AVAILABLE = False
    # Will log warning from main function when logger is available



def initialize_reddit_clients(logger) -> List[object]:
    """Initialize multiple Reddit API clients from JSON credentials file."""
    if not REDDIT_API_AVAILABLE:
        logger.warning("⚠️  Warning: praw or tqdm not available. Install with: pip install praw tqdm")
        return []

    # Load credentials from JSON file
    credentials_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'credentials',
        'reddit_api_keys.json'
    )

    if not os.path.exists(credentials_file):
        logger.error(f"❌ Credentials file not found: {credentials_file}")
        logger.error("💡 Create a JSON file with your Reddit API keys")
        logger.error("   See credentials/reddit_api_keys.json.template for format")
        return []

    try:
        import json
        with open(credentials_file, 'r') as f:
            cred_data = json.load(f)

        # Support both list format and dict format
        credentials = []
        if isinstance(cred_data, list):
            credentials = cred_data
        elif isinstance(cred_data, dict):
            if 'keys' in cred_data:
                credentials = cred_data['keys']
            else:
                # Single key in dict format
                credentials = [cred_data]

        logger.info(f"📂 Loaded {len(credentials)} API key(s) from {credentials_file}")

    except Exception as e:
        logger.error(f"❌ Failed to load credentials from {credentials_file}: {e}")
        return []

    # Initialize clients from credentials
    clients = []
    for idx, cred in enumerate(credentials, 1):
        try:
            reddit = praw.Reddit(
                client_id=cred['client_id'],
                client_secret=cred['client_secret'],
                user_agent=cred.get('user_agent', f'reddit-mod-pipeline:v1.0:key{idx}')
            )
            # Test the connection
            reddit.user.me()
            clients.append(reddit)
            logger.info(f"✅ Initialized Reddit API client #{idx}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize API key #{idx}: {e}")

    if not clients:
        logger.error("❌ No valid Reddit API keys could be initialized")
        return []

    logger.info(f"🔑 Total active Reddit API clients: {len(clients)}")
    return clients


def get_reddit_client_for_worker(clients: List[object], worker_id: int) -> Optional[object]:
    """Get a Reddit client assigned to a specific worker."""
    if not clients:
        return None
    return clients[worker_id % len(clients)]


def check_nsfw_status(reddit: object, subreddit_name: str, logger) -> bool:
    """Check if a subreddit is NSFW using Reddit API."""
    if not reddit:
        logger.warning(f"⚠️  No Reddit API - skipping r/{subreddit_name} (cannot verify SFW status)")
        return True  # Skip if we can't check (safer)

    try:
        subreddit = reddit.subreddit(subreddit_name)
        return subreddit.over18
    except Exception as e:
        logger.warning(f"⚠️  Error checking r/{subreddit_name}: {e}")
        return True  # Assume NSFW if we can't check (safer)


def extract_subreddit_data(reddit: object, subreddit_name: str, original_rank: int, mod_comment_count: int, logger) -> Optional[Dict[str, Any]]:
    """Extract full subreddit data from Reddit API."""
    if not reddit:
        return None  # No fallback - API is required for accurate data

    try:
        subreddit = reddit.subreddit(subreddit_name)

        # Trigger data fetch
        _ = subreddit.subscribers

        # Get all subreddit attributes
        subreddit_data = vars(subreddit).copy()

        # Add our custom fields
        subreddit_data['mod_comment_rank'] = original_rank
        subreddit_data['mod_comment_count'] = mod_comment_count
        subreddit_data['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Remove PRAW internals
        subreddit_data.pop('_reddit', None)
        subreddit_data.pop('_fetched', None)

        return subreddit_data

    except Exception as e:
        logger.warning(f"⚠️  Error getting data for r/{subreddit_name}: {e}")
        return None


def extract_community_rules(reddit: object, subreddit_name: str, logger) -> List[Dict[str, Any]]:
    """Extract community rules for a subreddit."""
    if not reddit:
        return []

    try:
        subreddit = reddit.subreddit(subreddit_name)
        rules = list(subreddit.rules)

        rules_data = []
        for idx, rule in enumerate(rules):
            rule_data = vars(rule).copy()
            rule_data['subreddit'] = subreddit_name
            rule_data['rule_index'] = idx + 1

            # Add cleaned versions of text fields
            rule_data['short_name_clean'] = clean_rule_text(rule_data.get('short_name', ''))
            rule_data['description_clean'] = clean_rule_text(rule_data.get('description', ''))
            rule_data['violation_reason_clean'] = clean_rule_text(rule_data.get('violation_reason', ''))

            # Add comprehensive text for embedding (used in Stage 4)
            rule_text = ""
            if rule_data['violation_reason_clean']:
                rule_text = f"Violation: {rule_data['violation_reason_clean']}\n"
            rule_text += f"Rule: {rule_data['short_name_clean'].rstrip('.')}. {rule_data['description_clean']}"
            rule_data['rule_comprehensive'] = rule_text.strip()

            # Remove PRAW internals
            rule_data.pop('_reddit', None)
            rule_data.pop('_fetched', None)

            rules_data.append(rule_data)

        return rules_data

    except Exception as e:
        logger.warning(f"⚠️  Error getting rules for r/{subreddit_name}: {e}")
        return []


def process_subreddit_with_retry(reddit: object, subreddit_name: str, original_rank: int, mod_comment_count: int, logger) -> Optional[Dict[str, Any]]:
    """Process a single subreddit with retry logic for rate limiting."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Check if NSFW
            is_nsfw = check_nsfw_status(reddit, subreddit_name, logger)
            if is_nsfw:
                return None  # NSFW, skip

            # Get subreddit data
            subreddit_data = extract_subreddit_data(reddit, subreddit_name, original_rank, mod_comment_count, logger)
            if not subreddit_data:
                return None

            # Get community rules
            rules_data = extract_community_rules(reddit, subreddit_name, logger)

            # Skip subreddits with insufficient rules for semantic matching
            if len(rules_data) < MIN_RULES_FOR_MATCHING:
                logger.info(f"⏭️  Skipping r/{subreddit_name}: only {len(rules_data)} rule(s) (need {MIN_RULES_FOR_MATCHING}+)")
                return None

            # Add SFW rank (will be set by caller)
            return {
                'subreddit': subreddit_data,
                'rules': rules_data
            }

        except Exception as e:
            error_str = str(e)
            # Only retry on rate limiting (429) or server errors (5xx)
            if "429" in error_str or any(code in error_str for code in ["500", "502", "503", "504"]):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5s, 10s, 15s backoff for rate limits
                    logger.warning(f"⚠️  Rate limited/server error for r/{subreddit_name} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # For non-retryable errors or final attempt, log and skip
            logger.warning(f"⚠️  Error processing r/{subreddit_name}: {e}")
            return None

    return None


def process_subreddit_worker(reddit_client: object, ranking_entry: Dict[str, Any], logger, worker_id: int) -> Optional[Dict[str, Any]]:
    """Worker function to process a single subreddit."""
    subreddit_name = ranking_entry['subreddit']
    original_rank = ranking_entry['rank']
    mod_comment_count = ranking_entry['mod_comment_count']

    # Process subreddit with retry logic
    sfw_entry = process_subreddit_with_retry(reddit_client, subreddit_name, original_rank, mod_comment_count, logger)

    if sfw_entry:
        logger.info(f"✅ [Worker {worker_id}] r/{subreddit_name} (mod rank {original_rank}) - {mod_comment_count:,} mod comments, {len(sfw_entry['rules'])} rules")

    return sfw_entry


def collect_sfw_subreddits(rankings_data: Dict[str, Any], reddit_clients: List[object], logger) -> List[Dict[str, Any]]:
    """Collect all SFW subreddits with at least MIN_MATCHED_COMMENTS mod comments in parallel."""
    logger.info(f"🚀 Collecting SFW subreddits with at least {MIN_MATCHED_COMMENTS} mod comments...")

    # Filter rankings to only those meeting the minimum threshold
    eligible_rankings = []
    for ranking_entry in rankings_data['rankings']:
        if ranking_entry['mod_comment_count'] >= MIN_MATCHED_COMMENTS:
            eligible_rankings.append(ranking_entry)
        else:
            # Stop at first subreddit below threshold (rankings are sorted)
            logger.info(f"⏭️  Stopping: r/{ranking_entry['subreddit']} has {ranking_entry['mod_comment_count']} mod comments (below minimum {MIN_MATCHED_COMMENTS})")
            break

    total_to_check = len(eligible_rankings)
    logger.info(f"📊 Found {total_to_check} subreddits to check (>= {MIN_MATCHED_COMMENTS} mod comments)")

    # Use one worker per API key for parallel processing
    num_workers = len(reddit_clients)
    logger.info(f"🔧 Using {num_workers} parallel workers (one per API key)")

    sfw_results = []
    checked_count = 0
    skipped_count = 0

    # Set up progress tracking
    if REDDIT_API_AVAILABLE:
        progress_bar = tqdm(total=total_to_check, desc="Collecting SFW subreddits")
    else:
        progress_bar = None

    # Thread-safe lock for progress tracking
    lock = threading.Lock()

    try:
        # Process subreddits in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_ranking = {}
            for idx, ranking_entry in enumerate(eligible_rankings):
                worker_id = idx % num_workers
                reddit_client = get_reddit_client_for_worker(reddit_clients, worker_id)
                future = executor.submit(process_subreddit_worker, reddit_client, ranking_entry, logger, worker_id)
                future_to_ranking[future] = ranking_entry

            # Collect results as they complete
            for future in as_completed(future_to_ranking):
                ranking_entry = future_to_ranking[future]

                with lock:
                    checked_count += 1

                try:
                    sfw_entry = future.result()

                    if sfw_entry:
                        with lock:
                            sfw_results.append((ranking_entry['rank'], sfw_entry))
                    else:
                        with lock:
                            skipped_count += 1

                except Exception as e:
                    with lock:
                        skipped_count += 1
                        logger.warning(f"⚠️  Error processing r/{ranking_entry['subreddit']}: {e}")

                if progress_bar:
                    progress_bar.update(1)

    finally:
        if progress_bar:
            progress_bar.close()

    # Sort results by original mod rank to maintain ordering
    sfw_results.sort(key=lambda x: x[0])

    # Extract just the entries and assign SFW ranks
    sfw_subreddits = []
    for _, sfw_entry in sfw_results:
        sfw_entry['subreddit']['sfw_rank'] = len(sfw_subreddits) + 1
        sfw_subreddits.append(sfw_entry)

    logger.info(f"📊 Collection complete:")
    logger.info(f"  Collected: {len(sfw_subreddits)} SFW subreddits with {MIN_RULES_FOR_MATCHING}+ rules and {MIN_MATCHED_COMMENTS}+ mod comments")
    logger.info(f"  Checked: {checked_count} subreddits total")
    logger.info(f"  Skipped: {skipped_count} subreddits (NSFW, insufficient rules, or errors)")

    return sfw_subreddits


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(2, "get_sfw_subreddits")
    log_stage_start(logger, 2, "Get SFW Subreddits with Minimum Mod Comments")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Load subreddit rankings from Stage 1
        rankings_file = os.path.join(PATHS['data'], 'stage1_subreddit_mod_comment_rankings.json')

        if not os.path.exists(rankings_file):
            logger.error(f"❌ Input file not found: {rankings_file}")
            logger.error("Make sure Stage 1 has completed successfully.")
            log_stage_end(logger, 2, success=False, elapsed_time=time.time() - start_time)
            return 1

        logger.info(f"🔍 Loading subreddit rankings from: {rankings_file}")
        rankings_data = read_json_file(rankings_file)

        total_subreddits = len(rankings_data['rankings'])
        logger.info(f"📊 Loaded {total_subreddits:,} subreddits from rankings")

        # Initialize Reddit API clients
        logger.info("🔌 Initializing Reddit API clients...")
        reddit_clients = initialize_reddit_clients(logger)

        if reddit_clients:
            logger.info(f"✅ Reddit API connected with {len(reddit_clients)} client(s)")
        else:
            logger.warning("⚠️  Reddit API unavailable - will skip all subreddits for safety")

        # Collect SFW subreddits
        sfw_subreddits = collect_sfw_subreddits(rankings_data, reddit_clients, logger)

        # Create output data
        output_data = {
            'summary': {
                'total_collected': len(sfw_subreddits),
                'min_mod_comments_threshold': MIN_MATCHED_COMMENTS,
                'min_rules_threshold': MIN_RULES_FOR_MATCHING,
                'source_file': rankings_file,
                'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'reddit_api_clients_used': len(reddit_clients),
                'reddit_api_used': len(reddit_clients) > 0,
                'filtering_method': 'Reddit API over18 flag' if reddit_clients else 'No filtering (API required)'
            },
            'subreddits': sfw_subreddits  # Each has 'subreddit' object with mod_comment_count and 'rules' list
        }

        # Save results
        output_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
        write_json_file(output_data, output_file, pretty=True)

        # Print summary
        elapsed = time.time() - start_time

        logger.info(f"Stage 2 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"Selected {len(sfw_subreddits)} SFW subreddits")
        logger.info(f"Results saved to: {output_file}")

        # Show top 10 selected subreddits
        logger.info(f"🏆 Top 10 SFW subreddits by mod comment count:")
        for i, entry in enumerate(sfw_subreddits[:10]):
            sub = entry['subreddit']
            rules_count = len(entry['rules'])
            subscribers = sub.get('subscribers', 'N/A')
            if isinstance(subscribers, int):
                subscribers = f"{subscribers:,}"
            logger.info(f"  {i+1:2d}. r/{sub['name']} - {sub['mod_comment_count']:,} mod comments, {subscribers} subscribers, {rules_count} rules")

        # Log minimum threshold information
        logger.info(f"📋 Collection criteria:")
        logger.info(f"  Minimum mod comments: {MIN_MATCHED_COMMENTS}")
        logger.info(f"  Minimum rules: {MIN_RULES_FOR_MATCHING}")
        logger.info(f"  SFW only: Yes")

        log_stage_end(logger, 2, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 2 execution")
        log_stage_end(logger, 2, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())