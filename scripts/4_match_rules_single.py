#!/usr/bin/env python3
"""
Single Subreddit Rule Matcher

Processes a single subreddit for comment-rule matching using embeddings.
Designed to be called as a subprocess to avoid multiprocessing issues with vLLM.

Usage: python 4_match_rules_single.py <subreddit_name> <cuda_device> <rules_file>
"""

import sys
import os
import time
import gc
import logging
import argparse

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

from config import (PATHS, SIMILARITY_THRESHOLD, EMBEDDING_MODEL, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS)
from utils.files import (read_json_file, write_json_file, read_zst_lines, json_loads)
from utils.reddit import normalize_subreddit_name, validate_comment_structure

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM
from vllm.inputs import TokensPrompt
# Set up subreddit-specific logging in subdirectory
from utils.logging import get_stage_logger
from config import create_directories

# Set deterministic behavior for PyTorch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimpleCommentRuleMatcher:
    """Simplified comment-rule matcher using embeddings - Phase 1 only."""

    def __init__(self, model_name: str = None, max_model_len: int = 2048):
        self.model_name = model_name or EMBEDDING_MODEL

        try:
            # Import now at top level

            print(f"Loading embedding model: {self.model_name} with max_model_len={max_model_len + 50}")
            self.model = LLM(
                model=self.model_name,
                task="embed",
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                max_model_len=max_model_len + 50,
                seed=0
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.model = None
            raise

    @staticmethod
    def save_similarity_matrix(cosine_similarities, comments, rules, subreddit_name):
        """Save similarity matrix to disk in efficient PyTorch format."""
        try:
            comment_mapping = {comment['id']: row_idx for row_idx, comment in enumerate(comments)}
            rule_indices = [rule.get("rule_index", i) for i, rule in enumerate(rules)]

            similarity_data = {
                'cosine_similarity_matrix': cosine_similarities.float(),
                'comment_mapping': comment_mapping,
                'rule_indices': rule_indices,
                'subreddit': subreddit_name,
                'num_comments': len(comments),
                'num_rules': len(rules),
                'scoring_method': 'cosine_similarity'
            }

            output_dir = PATHS.get('matched_comments')
            if not output_dir:
                raise ValueError("Output directory path not configured")
            if not os.path.exists(output_dir):
                raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

            matrix_file = os.path.join(output_dir, f"{subreddit_name}_similarity_matrix.pt")
            torch.save(similarity_data, matrix_file)
            print(f"💾 Saved similarity matrix to {matrix_file}")

        except Exception as e:
            print(f"❌ Failed to save similarity matrix: {e}")
            raise

    def calculate_similarities_pretokenized(self, comments, rules, tokenized_comments, tokenized_rules):
        """Phase 1: Calculate cosine similarities and save to .pt files."""
        # Early validation
        if not comments or not rules:
            print("⚠️  No comments or rules to process")
            return False

        if len(rules) <= 1:
            print("⚠️  Only one rule or no rules - skipping similarity calculation")
            return False

        if not tokenized_comments or not tokenized_rules:
            print("⚠️  No tokenized data provided")
            return False

        if len(tokenized_comments) != len(comments) or len(tokenized_rules) != len(rules):
            print("❌ Mismatch between original and tokenized data lengths")
            return False

        if not self.model:
            print("❌ CRITICAL: Model is None - cannot perform similarity calculation")
            raise RuntimeError("Model initialization failed - cannot perform similarity calculation")

        try:
            # Import now at top level

            print(f"Creating TokensPrompt objects for {len(tokenized_comments)} comments and {len(tokenized_rules)} rules...")
            comment_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_comments]
            rule_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_rules]

            print(f"Embedding {len(comment_prompts)} comments...")
            comment_outputs = self.model.embed(comment_prompts)
            comment_embeddings = torch.tensor([o.outputs.embedding for o in comment_outputs])

            print(f"Embedding {len(rule_prompts)} rule documents...")
            rule_outputs = self.model.embed(rule_prompts)
            rule_embeddings = torch.tensor([o.outputs.embedding for o in rule_outputs])

            print("Computing similarities...")
            similarities = comment_embeddings @ rule_embeddings.T

            subreddit_name = comments[0].get('subreddit', 'unknown') if comments else 'unknown'
            self.save_similarity_matrix(similarities, comments, rules, subreddit_name)

            print(f"✅ Calculated similarities for {len(comments)} comments and {len(rules)} rules")
            return True

        except Exception as e:
            print(f"❌ Error during similarity calculation: {e}")
            return False

    @classmethod
    def pretokenize_inputs(cls, comments, rules, model_name, task_description):
        """
        Pretokenize comments and rules to find optimal max_model_len.

        Returns:
            - tokenized_comments: List of token IDs for each comment
            - tokenized_rules: List of token IDs for each rule
            - max_length: Maximum token length across all inputs
        """
        # Early validation
        if not comments or not rules:
            print("⚠️  No comments or rules to tokenize")
            return [], [], 0

        if len(rules) <= 1:
            print("⚠️  Only one rule or no rules - skipping tokenization")
            return [], [], 0

        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize comments with instruction
        print(f"Tokenizing {len(comments)} comments...")
        tokenized_comments = []
        comment_lengths = []

        for comment in tqdm(comments, desc="Tokenizing comments"):
            body = comment.get('body_clean', '') or comment.get('removal_reason_clean', '')
            formatted_comment = f'Instruct: {task_description}\nQuery: {body}'

            tokens = tokenizer.encode(formatted_comment, add_special_tokens=True)
            tokenized_comments.append(tokens)
            comment_lengths.append(len(tokens))

        # Tokenize rules (documents, no instruction)
        print(f"Tokenizing {len(rules)} rules...")
        tokenized_rules = []
        rule_lengths = []

        for rule in tqdm(rules, desc="Tokenizing rules"):
            rule_text = rule['rule_comprehensive']
            tokens = tokenizer.encode(rule_text, add_special_tokens=True)
            tokenized_rules.append(tokens)
            rule_lengths.append(len(tokens))

        # Calculate statistics
        max_comment_len = max(comment_lengths) if comment_lengths else 0
        max_rule_len = max(rule_lengths) if rule_lengths else 0
        max_length = max(max_comment_len, max_rule_len)

        avg_comment_len = sum(comment_lengths) / len(comment_lengths) if comment_lengths else 0
        avg_rule_len = sum(rule_lengths) / len(rule_lengths) if rule_lengths else 0

        print(f"Tokenization complete:")
        print(f"  Max comment length: {max_comment_len}")
        print(f"  Max rule length: {max_rule_len}")
        print(f"  Overall max length: {max_length}")
        print(f"  Avg comment length: {avg_comment_len:.1f}")
        print(f"  Avg rule length: {avg_rule_len:.1f}")

        return tokenized_comments, tokenized_rules, max_length


def main():
    parser = argparse.ArgumentParser(
        description="Process a single subreddit for comment-rule matching using embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--subreddit", required=True, help="Name of the subreddit to process")
    parser.add_argument("--cuda-device", required=True, help="CUDA device ID to use (or 'None' for CPU)")

    args = parser.parse_args()

    subreddit_name = args.subreddit
    cuda_device = args.cuda_device
    rules_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

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

    # Load rules from Stage 2 JSON structure
    stage2_data = read_json_file(rules_file)
    rules = []

    # Find the subreddit in the nested structure (case-insensitive match on display_name)
    for subreddit_entry in stage2_data.get('subreddits', []):
        if subreddit_entry['subreddit']['display_name'].lower() == subreddit_name.lower():
            rules = subreddit_entry['rules']
            break

    input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
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
            "phase": 1,
            "total_comments": len(comments),
            "similarity_matrix_saved": False,
            "phase_1_complete": True,
            "skipped_reason": "Only one rule or no rules - no similarity calculation needed"
        }
    else:
        # Pretokenize and match
        task_description = "Which rule is this moderator's comment referring to?"
        logger.info(f"🧩 Starting pretokenization for {len(comments)} comments and {len(rules)} rules...")
        logger.info(f"🤖 Using embedding model: {EMBEDDING_MODEL}")
        tokenized_comments, tokenized_rules, max_length = SimpleCommentRuleMatcher.pretokenize_inputs(
            comments, rules, EMBEDDING_MODEL, task_description
        )
        logger.info(f"✅ Pretokenization completed. Max token length: {max_length}")

        optimal_max_len = max(max_length + 50, 512)
        logger.info(f"📏 Using optimal max_model_len={optimal_max_len}")

        logger.info(f"🏗️  Initializing SimpleCommentRuleMatcher...")
        try:
            subreddit_matcher = SimpleCommentRuleMatcher(
                max_model_len=optimal_max_len
            )
            logger.info(f"✅ SimpleCommentRuleMatcher initialized successfully")
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            sys.exit(1)

        logger.info(f"🔄 Starting Phase 1: similarity calculation for {len(comments)} comments and {len(rules)} rules...")
        try:
            success = subreddit_matcher.calculate_similarities_pretokenized(
                comments, rules, tokenized_comments, tokenized_rules
            )
            if success:
                logger.info(f"✅ Phase 1: Similarity calculation completed successfully")
                # Create placeholder stats for Phase 1 completion
                stats = {
                    "phase": 1,
                    "total_comments": len(comments),
                    "similarity_matrix_saved": True,
                    "phase_1_complete": True,
                    "matching_pending": "Phase 2 required for matching with data-driven thresholds"
                }
            else:
                raise RuntimeError("Similarity calculation failed")
        except Exception as e:
            logger.error(f"❌ Phase 1: Similarity calculation failed: {e}")
            raise

        stats['max_token_length'] = max_length
        stats['optimal_max_len'] = optimal_max_len

        # Phase 1 complete - no sampling or matching yet
        stats['submission_ids'] = []
        stats['sampled_comments_count'] = 0
        stats['unique_submission_ids'] = 0
        logger.info(f"📋 Phase 1 complete - similarity matrix saved, matching deferred to Phase 2")

    # Add metadata
    stats['subreddit'] = subreddit_name
    stats['total_rules'] = len(rules)

    # Phase 1: No matched comments to save yet - only similarity matrix is saved
    logger.info(f"📁 Phase 1: Similarity matrix saved to matched_comments/{subreddit_name}_similarity_matrix.pt")
    logger.info(f"⏭️  Phase 2 will be handled by main script after all subreddits complete Phase 1")

    try:
        write_json_file(stats, stats_file, pretty=True)
    except Exception as e:
        logger.error(f"⚠️  Error saving stats: {e}")

    phase = stats.get('phase', 1)
    similarity_saved = stats.get('similarity_matrix_saved', False)
    logger.info(f"📊 Phase {phase}: {len(comments)} comments processed, similarity matrix saved: {similarity_saved}")

    elapsed = time.time() - start_time
    logger.info(f"✅ Completed r/{subreddit_name} in {elapsed:.1f}s")

    # Explicit cleanup to avoid vLLM cleanup SIGABRT issues
    logger.info("🧹 Cleaning up vLLM resources...")
    try:
        if 'subreddit_matcher' in locals() and hasattr(subreddit_matcher, 'model') and subreddit_matcher.model:
            del subreddit_matcher.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("✅ Cleanup completed successfully")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")

    logger.info("🏁 Process exiting normally...")


if __name__ == "__main__":
    main()