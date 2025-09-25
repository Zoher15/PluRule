#!/usr/bin/env python3
"""
Single Subreddit Rule Matcher

Processes a single subreddit for comment-rule matching using Qwen3-reranker.
Designed to be called as a subprocess to avoid multiprocessing issues with vLLM.

Usage: python 4_match_rules_single.py <subreddit_name> <cuda_device> <rules_file>
"""

import sys
import os
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set deterministic behavior for reproducible results
os.environ['PYTHONHASHSEED'] = '0'
# Disable vLLM's default logging configuration to avoid conflicts
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
# Disable tqdm progress bars to avoid log spam
os.environ['TQDM_DISABLE'] = '1'
# Enable debug logging for vLLM engine processes
os.environ['VLLM_ENGINE_LOG_LEVEL'] = 'DEBUG'

from config import (PATHS, SCORE_THRESHOLD, RERANKER_MODEL, MAX_MATCHED_COMMENTS)
from utils.files import (read_json_file, write_json_file, read_zst_lines, json_loads, write_zst_json_objects)
from utils.reddit import normalize_subreddit_name, validate_comment_structure, extract_submission_id

# Try to import embedding dependencies
try:
    import torch
    from vllm import LLM
    from vllm.inputs import TextPrompt
    from transformers import AutoTokenizer

    # Set deterministic behavior for PyTorch
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Import from the main script using importlib to handle numeric filename
import importlib.util
import sys as sys_module

def import_from_main_script():
    """Import functions from the main 4_match_rules script."""
    main_script_path = os.path.join(os.path.dirname(__file__), '4_match_rules.py')
    spec = importlib.util.spec_from_file_location("match_rules_main", main_script_path)
    module = importlib.util.module_from_spec(spec)
    sys_module.modules["match_rules_main"] = module
    spec.loader.exec_module(module)
    return module



# Import the needed classes and functions
main_module = import_from_main_script()
SimpleCommentRuleMatcher = main_module.SimpleCommentRuleMatcher


def main():
    if len(sys.argv) != 4:
        print("Usage: python 4_match_rules_single.py <subreddit_name> <cuda_device> <rules_file>")
        sys.exit(1)

    subreddit_name = sys.argv[1]
    cuda_device = sys.argv[2]
    rules_file = sys.argv[3]

    # Set up subreddit-specific logging in subdirectory
    from utils.logging import get_stage_logger
    from config import create_directories
    import logging

    # Ensure base directories exist
    create_directories()

    # Use a path-like identifier to create subdirectory
    logger = get_stage_logger(4, "match_rules", worker_identifier=f"subreddits/{subreddit_name}")

    # Redirect ALL vLLM related loggers to our subreddit logger with DEBUG level
    vllm_loggers = [
        'vllm', 'vllm.engine', 'vllm.engine.llm_engine', 'vllm.worker',
        'vllm.model_executor', 'vllm.core', 'vllm.core.scheduler',
        'vllm.distributed', 'vllm.config', 'ray'  # Ray is used by vLLM
    ]

    for logger_name in vllm_loggers:
        vllm_logger = logging.getLogger(logger_name)
        vllm_logger.handlers = logger.handlers
        vllm_logger.setLevel(logging.DEBUG)
        vllm_logger.propagate = False  # Prevent duplicate logs

    logger.info(f"🔍 Enabled comprehensive DEBUG logging for all vLLM components")

    logger.info(f"🚀 Starting rule matching for r/{subreddit_name} on CUDA:{cuda_device}")
    start_time = time.time()

    # Set CUDA device
    if cuda_device != "None":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        logger.info(f"🎯 Processing r/{subreddit_name} on CUDA:{cuda_device}")
    else:
        logger.info(f"💻 Processing r/{subreddit_name} (CPU mode)")

    # Load rules
    subreddit_rules = read_json_file(rules_file)
    rules = subreddit_rules.get(subreddit_name, [])

    input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
    output_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
    sample_output_file = os.path.join(PATHS['matched_comments_sample'], f"{subreddit_name}_match.jsonl.zst")
    stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_stats.json")

    # Check if input file exists
    if not os.path.exists(input_file):
        stats = {"subreddit": subreddit_name, "error": f"Input file not found: {input_file}"}
        write_json_file(stats, stats_file, pretty=True)
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Load and validate comments
    try:
        lines = read_zst_lines(input_file)
        all_comments = []
        for line in lines:
            if line.strip():
                try:
                    comment = json_loads(line)
                    if validate_comment_structure(comment):
                        comment['subreddit'] = normalize_subreddit_name(comment.get('subreddit', ''))
                        all_comments.append(comment)
                except Exception:
                    continue

        comments = all_comments
        logger.info(f"📚 Loaded {len(comments)} valid comments and {len(rules)} rules")

    except Exception as e:
        stats = {"subreddit": subreddit_name, "error": f"Error loading comments: {e}"}
        write_json_file(stats, stats_file, pretty=True)
        logger.error(f"❌ Error loading comments: {e}")
        sys.exit(1)

    # Process rules and comments
    if len(rules) <= 1:
        stats = {
            "total_comments": len(comments),
            "matched_comments": 0,
            "ambiguous_matches": 0,
            "match_percentage": 0.0,
            "ambiguous_percentage": 0.0,
            "rule_matches": {},
            "skipped_reason": "Only one rule or no rules - no reranking needed"
        }
        matched_comments = []
    else:
        # Match using reranker
        task_description = "Given a moderator's comment, retrieve the exact rule that it mentions."
        logger.info(f"🤖 Using reranker model: {RERANKER_MODEL}")

        logger.info(f"🏗️  Initializing SimpleCommentRuleMatcher...")
        try:
            matcher = SimpleCommentRuleMatcher(
                score_threshold=SCORE_THRESHOLD  # Using same threshold value but now it's score threshold
            )
            logger.info(f"✅ SimpleCommentRuleMatcher initialized successfully")
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            sys.exit(1)

        logger.info(f"🔄 Starting comment-rule matching for {len(comments)} comments and {len(rules)} rules...")
        try:
            matched_comments, stats = matcher.match_comments_to_rules(comments, rules)
            logger.info(f"✅ Comment-rule matching completed successfully")
        except Exception as e:
            logger.error(f"❌ Comment-rule matching failed: {e}")
            raise

        # Sample matched comments and extract submission IDs
        if matched_comments:
            import random
            random.seed(0)
            sample_size = min(len(matched_comments), MAX_MATCHED_COMMENTS)
            sampled_comments = random.sample(matched_comments, sample_size)
            submission_ids = [extract_submission_id(c.get('link_id', '')) for c in sampled_comments]
            submission_ids = list(set(filter(None, submission_ids)))

            stats['submission_ids'] = submission_ids
            stats['sampled_comments_count'] = len(sampled_comments)
            stats['unique_submission_ids'] = len(submission_ids)
            logger.info(f"📋 Sampled {len(sampled_comments)} comments, found {len(submission_ids)} unique submission IDs")
        else:
            stats['submission_ids'] = []
            stats['sampled_comments_count'] = 0
            stats['unique_submission_ids'] = 0

    # Add metadata
    stats['subreddit'] = subreddit_name
    stats['total_rules'] = len(rules)

    # Save results
    if matched_comments:
        try:
            write_zst_json_objects(output_file, matched_comments)
            logger.info(f"✅ Saved {len(matched_comments)} matched comments")
        except Exception as e:
            stats['save_error'] = str(e)
            logger.error(f"❌ Error saving matched comments: {e}")

        # Save sampled comments if sampling was performed
        if sampled_comments:
            try:
                os.makedirs(os.path.dirname(sample_output_file), exist_ok=True)
                write_zst_json_objects(sample_output_file, sampled_comments)
                logger.info(f"✅ Saved {len(sampled_comments)} sampled comments")
            except Exception as e:
                stats['sample_save_error'] = str(e)
                logger.error(f"❌ Error saving sampled comments: {e}")

    try:
        write_json_file(stats, stats_file, pretty=True)
    except Exception as e:
        logger.error(f"⚠️  Error saving stats: {e}")

    match_pct = stats.get('match_percentage', 0)
    ambiguous_count = stats.get('ambiguous_matches', 0)
    ambiguous_pct = stats.get('ambiguous_percentage', 0)
    logger.info(f"📊 {len(comments)} comments, {len(matched_comments)} matched ({match_pct:.1f}%), {ambiguous_count} ambiguous ({ambiguous_pct:.1f}%)")

    elapsed = time.time() - start_time
    logger.info(f"✅ Completed r/{subreddit_name} in {elapsed:.1f}s")

    # Explicit cleanup to avoid vLLM cleanup SIGABRT issues
    logger.info("🧹 Cleaning up vLLM resources...")
    try:
        if 'matcher' in locals() and hasattr(matcher, 'model') and matcher.model:
            del matcher.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")

    logger.info("🏁 Process exiting normally...")


if __name__ == "__main__":
    main()