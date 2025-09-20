#!/usr/bin/env python3
"""
Stage 2: Get Top N SFW Subreddits

Takes the subreddit rankings from Stage 1 and selects the top N SFW subreddits
based on mod comment activity. Uses Reddit API to check NSFW status and collect
subreddit metadata and community rules.

Input:  subreddit_mod_comment_rankings.json
Output: top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS, create_directories
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



def initialize_reddit_client(logger) -> Optional[object]:
    """Initialize Reddit API client."""
    if not REDDIT_API_AVAILABLE:
        logger.warning("⚠️  Warning: praw or tqdm not available. Install with: pip install praw tqdm")
        return None

    try:
        # Try to use environment variables or praw.ini
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'reddit-mod-pipeline:v1.0')
        )

        # Test the connection
        reddit.user.me()
        return reddit

    except Exception as e:
        logger.error(f"⚠️  Reddit API initialization failed: {e}")
        logger.error("💡 Set environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET")
        return None


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

            # Remove PRAW internals
            rule_data.pop('_reddit', None)
            rule_data.pop('_fetched', None)

            rules_data.append(rule_data)

        return rules_data

    except Exception as e:
        logger.warning(f"⚠️  Error getting rules for r/{subreddit_name}: {e}")
        return []


def collect_sfw_subreddits(rankings_data: Dict[str, Any], reddit: object, logger) -> List[Dict[str, Any]]:
    """Collect top N SFW subreddits with full metadata and rules."""
    logger.info(f"🚀 Collecting top {TOP_N_SUBREDDITS_WITH_MOD_COMMENTS} SFW subreddits...")

    sfw_subreddits = []
    checked_count = 0
    nsfw_skipped = 0

    # Set up progress tracking
    if REDDIT_API_AVAILABLE:
        progress_bar = tqdm(total=TOP_N_SUBREDDITS_WITH_MOD_COMMENTS, desc="Collecting SFW subreddits")
    else:
        progress_bar = None

    try:
        for ranking_entry in rankings_data['rankings']:
            if len(sfw_subreddits) >= TOP_N_SUBREDDITS_WITH_MOD_COMMENTS:
                break

            subreddit_name = ranking_entry['subreddit']
            original_rank = ranking_entry['rank']
            mod_comment_count = ranking_entry['mod_comment_count']

            checked_count += 1

            # Check if NSFW
            is_nsfw = check_nsfw_status(reddit, subreddit_name, logger)

            if is_nsfw:
                nsfw_skipped += 1
                logger.info(f"🔒 Skipping r/{subreddit_name} (NSFW) - rank {original_rank}")
                continue

            # Get subreddit data
            subreddit_data = extract_subreddit_data(reddit, subreddit_name, original_rank, mod_comment_count, logger)
            if not subreddit_data:
                continue

            # Get community rules
            rules_data = extract_community_rules(reddit, subreddit_name, logger)

            # Add SFW rank
            subreddit_data['sfw_rank'] = len(sfw_subreddits) + 1

            # Create entry
            sfw_entry = {
                'subreddit': subreddit_data,
                'rules': rules_data
            }

            sfw_subreddits.append(sfw_entry)

            logger.info(f"✅ r/{subreddit_name} - SFW rank {len(sfw_subreddits)} (mod rank {original_rank}) - {len(rules_data)} rules")

            if progress_bar:
                progress_bar.update(1)

    finally:
        if progress_bar:
            progress_bar.close()

    logger.info(f"📊 Collection complete:")
    logger.info(f"  Collected: {len(sfw_subreddits)} SFW subreddits")
    logger.info(f"  Checked: {checked_count} subreddits total")
    logger.info(f"  Skipped: {nsfw_skipped} NSFW subreddits")

    return sfw_subreddits


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(2, "get_top_sfw_subreddits")
    log_stage_start(logger, 2, "Get Top N SFW Subreddits")

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

        # Initialize Reddit API
        logger.info("🔌 Initializing Reddit API...")
        reddit = initialize_reddit_client(logger)

        if reddit:
            logger.info("✅ Reddit API connected")
        else:
            logger.warning("⚠️  Reddit API unavailable - will skip all subreddits for safety")

        # Collect SFW subreddits
        sfw_subreddits = collect_sfw_subreddits(rankings_data, reddit, logger)

        if len(sfw_subreddits) < TOP_N_SUBREDDITS_WITH_MOD_COMMENTS:
            logger.warning(f"⚠️  Warning: Only collected {len(sfw_subreddits)} SFW subreddits, "
                  f"less than target {TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}")

        # Create output data
        output_data = {
            'summary': {
                'total_collected': len(sfw_subreddits),
                'target_count': TOP_N_SUBREDDITS_WITH_MOD_COMMENTS,
                'source_file': rankings_file,
                'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'reddit_api_used': reddit is not None,
                'filtering_method': 'Reddit API over18 flag' if reddit else 'No filtering (API required)'
            },
            'subreddits': sfw_subreddits
        }

        # Save results
        output_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')
        write_json_file(output_data, output_file)

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

        log_stage_end(logger, 2, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 2 execution")
        log_stage_end(logger, 2, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())