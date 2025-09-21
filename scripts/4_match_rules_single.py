#!/usr/bin/env python3
"""
Single Subreddit Rule Matcher

Processes a single subreddit for comment-rule matching using embeddings.
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

from config import (PATHS, SIMILARITY_THRESHOLD, EMBEDDING_MODEL, MAX_MATCHED_COMMENTS)
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
pretokenize_inputs = main_module.pretokenize_inputs


def main():
    if len(sys.argv) != 4:
        print("Usage: python 4_match_rules_single.py <subreddit_name> <cuda_device> <rules_file>")
        sys.exit(1)

    subreddit_name = sys.argv[1]
    cuda_device = sys.argv[2]
    rules_file = sys.argv[3]

    # Set CUDA device
    if cuda_device != "None":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        print(f"🔄 Processing r/{subreddit_name} on CUDA:{cuda_device}")
    else:
        print(f"🔄 Processing r/{subreddit_name} (CPU mode)")

    # Load rules
    subreddit_rules = read_json_file(rules_file)
    rules = subreddit_rules.get(subreddit_name, [])

    input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
    output_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
    stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_stats.json")

    # Check if input file exists
    if not os.path.exists(input_file):
        stats = {"subreddit": subreddit_name, "error": f"Input file not found: {input_file}"}
        write_json_file(stats, stats_file, pretty=True)
        print(f"❌ Input file not found: {input_file}")
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
        print(f"  Loaded {len(comments)} valid comments and {len(rules)} rules")

    except Exception as e:
        stats = {"subreddit": subreddit_name, "error": f"Error loading comments: {e}"}
        write_json_file(stats, stats_file, pretty=True)
        print(f"❌ Error loading comments: {e}")
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
            "skipped_reason": "Only one rule or no rules - no similarity matching needed"
        }
        matched_comments = []
    else:
        # Pretokenize and match
        task_description = "Which rule is this moderator's comment referring to?"
        print(f"  Pretokenizing inputs...")
        tokenized_comments, tokenized_rules, max_length = pretokenize_inputs(
            comments, rules, EMBEDDING_MODEL, task_description
        )

        optimal_max_len = max(max_length + 50, 512)
        print(f"  Using optimal max_model_len={optimal_max_len}")

        matcher = SimpleCommentRuleMatcher(
            similarity_threshold=SIMILARITY_THRESHOLD,
            max_model_len=optimal_max_len
        )

        matched_comments, stats = matcher.match_comments_to_rules_pretokenized(
            comments, rules, tokenized_comments, tokenized_rules
        )

        stats['max_token_length'] = max_length
        stats['optimal_max_len'] = optimal_max_len

        # Sample matched comments and extract submission IDs
        if matched_comments:
            import random
            random.seed(0)
            sample_size = min(len(matched_comments), MAX_MATCHED_COMMENTS)
            sampled_comments = random.sample(matched_comments, sample_size)
            submission_ids = [extract_submission_id(c.get('parent_id', '')) for c in sampled_comments]
            submission_ids = list(set(filter(None, submission_ids)))

            stats['submission_ids'] = submission_ids
            stats['sampled_comments_count'] = len(sampled_comments)
            stats['unique_submission_ids'] = len(submission_ids)
            print(f"  📋 Sampled {len(sampled_comments)} comments, found {len(submission_ids)} unique submission IDs")
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
            print(f"  ✅ Saved {len(matched_comments)} matched comments")
        except Exception as e:
            stats['save_error'] = str(e)
            print(f"  ❌ Error saving matched comments: {e}")

    try:
        write_json_file(stats, stats_file, pretty=True)
    except Exception as e:
        print(f"  ⚠️  Error saving stats: {e}")

    match_pct = stats.get('match_percentage', 0)
    ambiguous_count = stats.get('ambiguous_matches', 0)
    ambiguous_pct = stats.get('ambiguous_percentage', 0)
    print(f"  📊 {len(comments)} comments, {len(matched_comments)} matched ({match_pct:.1f}%), {ambiguous_count} ambiguous ({ambiguous_pct:.1f}%)")

    print(f"✅ Completed r/{subreddit_name}")


if __name__ == "__main__":
    main()