#!/usr/bin/env python3
"""
Bucket-level Rule Matcher

Processes a bucket of subreddits with a single vLLM instance for efficiency.
Called as subprocess with assigned CUDA device and subreddit list.

Usage: python 4_match_rules_bucket.py --cuda-device 0 --subreddits sub1,sub2,sub3
"""

import sys
import os
import time
import gc
import logging
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment for deterministic behavior and vLLM stability
os.environ['PYTHONHASHSEED'] = '0'
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ['TQDM_DISABLE'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from config import PATHS, EMBEDDING_MODEL, MIN_MATCHED_COMMENTS, create_directories
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads
from utils.reddit import normalize_subreddit_name, validate_comment_structure
from utils.logging import get_stage_logger
from utils.embedding_matcher import SimpleCommentRuleMatcher


def load_subreddit_data(subreddit_name, stage2_data):
    """Load comments and rules for a subreddit."""
    # Find rules from Stage 2
    rules = []
    for entry in stage2_data.get('subreddits', []):
        if entry['subreddit']['display_name'].lower() == subreddit_name.lower():
            rules = entry['rules']
            break

    # Load comments from Stage 3
    input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
    if not os.path.exists(input_file):
        return None, None, f"Input file not found: {input_file}"

    try:
        lines = read_zst_lines(input_file)
        comments = []
        for line in lines:
            if line.strip():
                try:
                    comment = json_loads(line)
                    if validate_comment_structure(comment):
                        comment['subreddit'] = normalize_subreddit_name(comment.get('subreddit', ''))
                        comments.append(comment)
                except Exception:
                    continue

        return comments, rules, None
    except Exception as e:
        return None, None, f"Error loading comments: {e}"


def main():
    parser = argparse.ArgumentParser(description="Process a bucket of subreddits with shared vLLM instance")
    parser.add_argument("--cuda-device", required=True, help="CUDA device ID (or 'None' for CPU)")
    parser.add_argument("--subreddits", required=True, help="Comma-separated list of subreddit names")
    args = parser.parse_args()

    cuda_device = args.cuda_device
    subreddit_list = [s.strip() for s in args.subreddits.split(',')]

    # Create directories
    create_directories()

    # Setup logging for this bucket
    bucket_id = f"bucket_cuda{cuda_device}"
    logger = get_stage_logger(4, "match_rules", worker_identifier=f"buckets/{bucket_id}")

    # Redirect vLLM logging
    for logger_name in ['vllm', 'vllm.engine', 'vllm.worker', 'ray']:
        vllm_logger = logging.getLogger(logger_name)
        vllm_logger.handlers = logger.handlers
        vllm_logger.setLevel(logging.WARNING)
        vllm_logger.propagate = False

    logger.info(f"🚀 Processing bucket with {len(subreddit_list)} subreddits on CUDA:{cuda_device}")
    start_time = time.time()

    # Set CUDA device
    if cuda_device != "None":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        logger.info(f"🎯 Using CUDA:{cuda_device}")
    else:
        logger.info(f"💻 Using CPU mode")

    # Load Stage 2 data
    rules_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
    stage2_data = read_json_file(rules_file)

    # Phase 1: Load and pretokenize ALL subreddits in bucket
    logger.info(f"📚 Phase 1: Loading and pretokenizing {len(subreddit_list)} subreddits...")
    tokenized_data = {}
    max_len = 0
    task_description = "Which rule is this moderator's comment referring to?"

    for subreddit in subreddit_list:
        comments, rules, error = load_subreddit_data(subreddit, stage2_data)

        if error:
            logger.error(f"❌ r/{subreddit}: {error}")
            stats = {"subreddit": subreddit, "error": error, "phase": 1}
            stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_stats.json")
            write_json_file(stats, stats_file, pretty=True)
            continue

        if len(rules) <= 1:
            logger.warning(f"⏭️  r/{subreddit}: Only {len(rules)} rule(s), skipping")
            stats = {"subreddit": subreddit, "skipped_reason": "Insufficient rules", "phase": 1}
            stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_stats.json")
            write_json_file(stats, stats_file, pretty=True)
            continue

        logger.info(f"🧩 r/{subreddit}: Pretokenizing {len(comments)} comments, {len(rules)} rules")
        tok_comments, tok_rules, sub_max = SimpleCommentRuleMatcher.pretokenize_inputs(
            comments, rules, EMBEDDING_MODEL, task_description
        )

        tokenized_data[subreddit] = {
            'comments': comments,
            'rules': rules,
            'tok_comments': tok_comments,
            'tok_rules': tok_rules,
            'max_len': sub_max
        }
        max_len = max(max_len, sub_max)
        logger.info(f"✅ r/{subreddit}: Max token length = {sub_max}")

    if not tokenized_data:
        logger.error("❌ No valid subreddits to process in bucket")
        return 1

    # Phase 2: Initialize vLLM once with bucket's max_model_len (with retry for rate limits)
    optimal_max_len = max(max_len + 50, 512)
    logger.info(f"🏗️  Phase 2: Initializing vLLM with max_model_len={optimal_max_len}")

    max_retries = 5
    retry_delay = 60  # Start with 60 seconds

    for attempt in range(max_retries):
        try:
            matcher = SimpleCommentRuleMatcher(max_model_len=optimal_max_len)
            logger.info(f"✅ vLLM initialized successfully")
            break
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(x in error_msg for x in ["429", "rate limit", "too many requests", "invalid repository id", "error retrieving"])

            if is_rate_limit:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"⚠️  HuggingFace rate limit or connection issue. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    logger.error(f"❌ vLLM initialization failed after {max_retries} attempts: {e}")
                    return 1
            else:
                # Non-rate-limit error, fail immediately
                logger.error(f"❌ vLLM initialization failed (non-rate-limit error): {e}")
                return 1
    else:
        logger.error("❌ vLLM initialization failed: Maximum retries exceeded")
        return 1

    # Phase 3: Process each subreddit with shared model
    logger.info(f"🔄 Phase 3: Processing {len(tokenized_data)} subreddits with shared vLLM instance...")

    for subreddit, data in tokenized_data.items():
        try:
            logger.info(f"🔄 r/{subreddit}: Embedding and computing similarities...")
            success = matcher.calculate_similarities_pretokenized(
                data['comments'], data['rules'],
                data['tok_comments'], data['tok_rules']
            )

            if success:
                logger.info(f"✅ r/{subreddit}: Similarity matrix saved")
                stats = {
                    "phase": 1,
                    "subreddit": subreddit,
                    "total_comments": len(data['comments']),
                    "total_rules": len(data['rules']),
                    "similarity_matrix_saved": True,
                    "phase_1_complete": True,
                    "max_token_length": data['max_len'],
                    "optimal_max_len": optimal_max_len
                }
            else:
                raise RuntimeError("Similarity calculation returned False")

        except Exception as e:
            logger.error(f"❌ r/{subreddit}: Failed - {e}")
            stats = {"subreddit": subreddit, "error": str(e), "phase": 1}

        # Save stats
        stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_stats.json")
        write_json_file(stats, stats_file, pretty=True)

    # Cleanup
    logger.info("🧹 Cleaning up vLLM resources...")
    try:
        del matcher
        del tokenized_data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️  Cleanup warning: {e}")

    elapsed = time.time() - start_time
    logger.info(f"✅ Bucket complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
