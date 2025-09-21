#!/usr/bin/env python3
"""
Stage 4: Match Comments to Rules

Uses semantic similarity to match moderator comments to subreddit rules.
Loads filtered comments from Stage 3 and subreddit rules from Stage 2,
then uses embedding models to find the best rule matches.

Input:  top_subreddits/{subreddit}_mod_comments.jsonl.zst files + top_N_sfw_subreddits.json
Output: matched_comments/{subreddit}_match.jsonl.zst + {subreddit}_stats.json files
"""

import sys
import os
import time
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, PROCESSES, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS,
                   SIMILARITY_THRESHOLD, EMBEDDING_MODEL, MIN_MATCHED_COMMENTS,
                   MAX_MATCHED_COMMENTS, create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import (read_json_file, write_json_file, process_files_parallel,
                        read_zst_lines, json_loads, write_zst_json_objects)
from utils.reddit import clean_rule_text, normalize_subreddit_name, validate_comment_structure, extract_submission_id
from utils.stats import calculate_jsd_from_uniform, rank_by_score, analyze_rule_distribution

# Try to import embedding dependencies
try:
    import torch
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from tqdm import tqdm
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    # Will log warning from main function when logger is available


def load_subreddit_rules(logger) -> Dict[str, List[Dict[str, Any]]]:
    """Load subreddit rules from Stage 2 output."""
    rules_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

    if not os.path.exists(rules_file):
        logger.error(f"❌ Rules file not found: {rules_file}")
        return {}

    logger.info(f"Loading subreddit rules from: {rules_file}")
    data = read_json_file(rules_file)

    subreddit_rules = {}
    for entry in data['subreddits']:
        subreddit_data = entry['subreddit']
        subreddit_name = subreddit_data.get('display_name', '').lower()

        if not subreddit_name:
            continue

        rules = []
        for rule in entry.get('rules', []):
            cleaned_rule = {
                'rule_index': rule.get('rule_index', 0),
                'short_name': rule.get('short_name_clean', ''),
                'description': rule.get('description_clean', ''),
                'kind': rule.get('kind', ''),
                'violation_reason': rule.get('violation_reason_clean', '')
            }

            # Combine relevant fields for embedding
            rule_text = f"Short Name: {cleaned_rule['short_name']}\nDescription: {cleaned_rule['description']}"
            if cleaned_rule['violation_reason']:
                rule_text += f"\nViolation Reason: {cleaned_rule['violation_reason']}"

            cleaned_rule['combined_text'] = rule_text.strip()
            rules.append(cleaned_rule)

        subreddit_rules[subreddit_name] = rules

    logger.info(f"Loaded rules for {len(subreddit_rules)} subreddits")
    return subreddit_rules


def pretokenize_inputs(comments: List[Dict[str, Any]], rules: List[Dict[str, Any]],
                      model_name: str, task_description: str) -> tuple:
    """
    Pretokenize comments and rules to find optimal max_model_len.

    Returns:
        - tokenized_comments: List of token IDs for each comment
        - tokenized_rules: List of token IDs for each rule
        - max_length: Maximum token length across all inputs
    """
    if not EMBEDDINGS_AVAILABLE:
        return [], [], 512

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
        rule_text = rule['combined_text']
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


def extract_submission_ids(comments: List[Dict[str, Any]]) -> List[str]:
    """Extract unique submission IDs from comments using utility function."""
    submission_ids = set()

    for comment in comments:
        # Check for submission_id field first
        if 'submission_id' in comment:
            submission_ids.add(comment['submission_id'])
        else:
            # Use utility function for link_id extraction
            submission_id = extract_submission_id(comment.get('link_id', ''))
            if submission_id:
                submission_ids.add(submission_id)

    return list(submission_ids)


def get_available_cuda_devices() -> List[int]:
    """Get list of available CUDA devices."""
    if not EMBEDDINGS_AVAILABLE:
        return []

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            return list(range(device_count))
        else:
            return []
    except Exception:
        return []


class SimpleCommentRuleMatcher:
    """Simplified comment-rule matcher using embeddings."""

    def __init__(self, model_name: str = None, similarity_threshold: float = 0.8, max_model_len: int = 2048):
        self.model_name = model_name or EMBEDDING_MODEL
        self.similarity_threshold = similarity_threshold
        self.task_description = "Which rule is this moderator's comment referring to?"

        if not EMBEDDINGS_AVAILABLE:
            print("⚠️  Embedding libraries not available - will skip matching")
            self.model = None
            return

        try:
            print(f"Loading embedding model: {self.model_name} with max_model_len={max_model_len}")
            self.model = LLM(
                model=self.model_name,
                task="embed",
                gpu_memory_utilization=0.95,
                enforce_eager=True,
                max_model_len=max_model_len + 50  # Add buffer for safety
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.model = None


    def match_comments_to_rules_pretokenized(self, comments: List[Dict[str, Any]], rules: List[Dict[str, Any]],
                                           tokenized_comments: List[List[int]], tokenized_rules: List[List[int]]) -> tuple:
        """Match comments to rules using cosine similarity with pretokenized inputs."""
        if not self.model or not comments or not rules:
            return [], {"total_comments": len(comments), "matched_comments": 0, "match_percentage": 0.0, "rule_matches": {}}

        # Skip if only one rule
        if len(rules) <= 1:
            return [], {
                "total_comments": len(comments),
                "matched_comments": 0,
                "match_percentage": 0.0,
                "rule_matches": {},
                "skipped_reason": "Only one rule or no rules - no similarity matching needed"
            }

        try:
            # Use pretokenized inputs directly with TokensPrompt
            print(f"Creating TokensPrompt objects for {len(tokenized_comments)} comments and {len(tokenized_rules)} rules...")
            comment_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_comments]
            rule_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_rules]

            print(f"Embedding {len(comment_prompts)} comments...")
            comment_outputs = self.model.embed(comment_prompts)
            comment_embeddings = torch.tensor([o.outputs.embedding for o in comment_outputs])

            print(f"Embedding {len(rule_prompts)} rule documents...")
            rule_outputs = self.model.embed(rule_prompts)
            rule_embeddings = torch.tensor([o.outputs.embedding for o in rule_outputs])

            # Compute cosine similarities
            print("Computing similarities...")
            similarities = comment_embeddings @ rule_embeddings.T

            # Find matches above threshold (with uniqueness check)
            matched_comments = []
            rule_match_counts = defaultdict(int)
            matched_count = 0
            ambiguous_count = 0

            for i, comment in enumerate(comments):
                comment_similarities = similarities[i]
                max_similarity = torch.max(comment_similarities)

                if max_similarity >= self.similarity_threshold:
                    # Check for ambiguous matches (multiple rules with same max score)
                    max_indices = torch.where(comment_similarities == max_similarity)[0]

                    if len(max_indices) > 1:
                        # Multiple rules have the same max score - ambiguous match, skip
                        ambiguous_count += 1
                        continue

                    # Unambiguous match
                    best_rule_idx = torch.argmax(comment_similarities).item()
                    best_rule = rules[best_rule_idx]

                    matched_comment = comment.copy()
                    matched_comment['matched_rule'] = {
                        'rule_index': best_rule['rule_index'],
                        'short_name': best_rule['short_name'],
                        'description': best_rule['description'],
                        'similarity_score': float(max_similarity)
                    }

                    matched_comments.append(matched_comment)
                    rule_match_counts[best_rule['rule_index']] += 1
                    matched_count += 1

            # Prepare statistics
            stats = {
                "total_comments": len(comments),
                "matched_comments": matched_count,
                "ambiguous_matches": ambiguous_count,
                "match_percentage": (matched_count / len(comments) * 100) if comments else 0.0,
                "ambiguous_percentage": (ambiguous_count / len(comments) * 100) if comments else 0.0,
                "rule_matches": dict(rule_match_counts)
            }

            return matched_comments, stats

        except Exception as e:
            print(f"❌ Error during pretokenized matching: {e}")
            return [], {
                "total_comments": len(comments),
                "matched_comments": 0,
                "ambiguous_matches": 0,
                "match_percentage": 0.0,
                "ambiguous_percentage": 0.0,
                "rule_matches": {},
                "error": str(e)
            }


def process_single_subreddit(args: tuple) -> Dict[str, Any]:
    """Process a single subreddit for comment-rule matching."""
    subreddit_name, subreddit_rules, cuda_device = args

    # Set CUDA device for this process
    if cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        print(f"🔄 Processing r/{subreddit_name} on CUDA:{cuda_device}")
    else:
        print(f"🔄 Processing r/{subreddit_name} (CPU mode)")

    input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
    output_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
    stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_stats.json")

    print(f"🔄 Processing r/{subreddit_name}")

    # Check if input file exists
    if not os.path.exists(input_file):
        return {
            "subreddit": subreddit_name,
            "error": f"Input file not found: {input_file}"
        }

    # Load comments with validation and normalization
    try:
        lines = read_zst_lines(input_file)

        # Parse and validate comments
        all_comments = []
        for line in lines:
            if line.strip():
                try:
                    comment = json_loads(line)

                    # Validate comment structure
                    if validate_comment_structure(comment):
                        # Normalize subreddit name for consistency
                        comment['subreddit'] = normalize_subreddit_name(comment.get('subreddit', ''))
                        all_comments.append(comment)

                except Exception:
                    continue  # Skip malformed comments

        comments = all_comments
        rules = subreddit_rules.get(subreddit_name, [])

        print(f"  Loaded {len(comments)} valid comments and {len(rules)} rules")

    except Exception as e:
        return {
            "subreddit": subreddit_name,
            "error": f"Error loading comments: {e}"
        }

    # Skip if only one rule (no point in similarity matching)
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
        # Pretokenize inputs to get optimal max_model_len
        task_description = "Which rule is this moderator's comment referring to?"
        print(f"  Pretokenizing inputs...")
        tokenized_comments, tokenized_rules, max_length = pretokenize_inputs(
            comments, rules, EMBEDDING_MODEL, task_description
        )

        # Calculate optimal max_model_len with buffer
        optimal_max_len = max(max_length + 50, 512)  # At least 512, plus 50 token buffer
        print(f"  Using optimal max_model_len={optimal_max_len}")

        # Initialize matcher with optimal settings
        matcher = SimpleCommentRuleMatcher(
            similarity_threshold=SIMILARITY_THRESHOLD,
            max_model_len=optimal_max_len
        )

        # Match using pretokenized inputs
        matched_comments, stats = matcher.match_comments_to_rules_pretokenized(
            comments, rules, tokenized_comments, tokenized_rules
        )

        # Add tokenization stats
        stats['max_token_length'] = max_length
        stats['optimal_max_len'] = optimal_max_len

        # Sample matched comments and extract submission IDs
        if matched_comments:
            import random
            random.seed(0)  # Consistent sampling
            sample_size = min(len(matched_comments), MAX_MATCHED_COMMENTS)
            sampled_comments = random.sample(matched_comments, sample_size)
            submission_ids = extract_submission_ids(sampled_comments)
            stats['submission_ids'] = submission_ids
            stats['sampled_comments_count'] = len(sampled_comments)
            stats['unique_submission_ids'] = len(submission_ids)
            print(f"  📋 Sampled {len(sampled_comments)} comments, found {len(submission_ids)} unique submission IDs")
        else:
            stats['submission_ids'] = []
            stats['sampled_comments_count'] = 0
            stats['unique_submission_ids'] = 0


    # Add subreddit info to stats
    stats['subreddit'] = subreddit_name
    stats['total_rules'] = len(rules)

    # Save matched comments if any
    if matched_comments:
        try:
            write_zst_json_objects(output_file, matched_comments)
            print(f"  ✅ Saved {len(matched_comments)} matched comments")
        except Exception as e:
            stats['save_error'] = str(e)
            print(f"  ❌ Error saving matched comments: {e}")

    # Save stats
    try:
        write_json_file(stats, stats_file)
    except Exception as e:
        print(f"  ⚠️  Error saving stats: {e}")

    match_pct = stats.get('match_percentage', 0)
    ambiguous_count = stats.get('ambiguous_matches', 0)
    ambiguous_pct = stats.get('ambiguous_percentage', 0)
    print(f"  📊 {len(comments)} comments, {len(matched_comments)} matched ({match_pct:.1f}%), {ambiguous_count} ambiguous ({ambiguous_pct:.1f}%)")

    return stats


def main():
    """Main execution function."""
    # Initialize logging
    logger = get_stage_logger(4, "match_rules")
    log_stage_start(logger, 4, "Match Comments to Rules")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Check if embeddings are available
        if not EMBEDDINGS_AVAILABLE:
            logger.error("❌ Embedding libraries not available!")
            logger.error("Install with: pip install vllm transformers torch")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Load subreddit rules
        logger.info("📚 Loading subreddit rules...")
        subreddit_rules = load_subreddit_rules(logger)

        if not subreddit_rules:
            logger.error("❌ No subreddit rules loaded!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Get subreddit files to process
        subreddit_files = []
        top_subreddits_dir = PATHS['top_subreddits']

        if not os.path.exists(top_subreddits_dir):
            logger.error(f"❌ Top subreddits directory not found: {top_subreddits_dir}")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        for filename in os.listdir(top_subreddits_dir):
            if filename.endswith('_mod_comments.jsonl.zst'):
                subreddit = filename.replace('_mod_comments.jsonl.zst', '')
                if subreddit in subreddit_rules:
                    subreddit_files.append(subreddit)

        if not subreddit_files:
            logger.error("❌ No subreddit files found to process!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Get available CUDA devices
        cuda_devices = get_available_cuda_devices()

        if cuda_devices:
            num_workers = len(cuda_devices)
            logger.info(f"Found {len(cuda_devices)} CUDA devices: {cuda_devices}")
            logger.info(f"Using {num_workers} parallel GPU processes")
        else:
            num_workers = min(PROCESSES, len(subreddit_files))  # Limit to available subreddits
            cuda_devices = [None] * num_workers  # CPU mode
            logger.info(f"No CUDA devices found - using {num_workers} CPU processes")

        logger.info(f"Found {len(subreddit_files)} subreddit files to process")

        # Assign CUDA devices to subreddits in round-robin fashion
        process_args = []
        for i, subreddit in enumerate(subreddit_files):
            cuda_device = cuda_devices[i % len(cuda_devices)]
            process_args.append((subreddit, subreddit_rules, cuda_device))

        logger.info("🚀 Processing subreddits...")
        results = process_files_parallel(process_args, process_single_subreddit, num_workers, logger)

        # Aggregate results
        successful_results = [r for r in results if not r.get("error")]
        failed_results = [r for r in results if r.get("error")]

        # Calculate JSD and rank subreddits using utilities
        logger.info("🔄 Calculating JSD and ranking subreddits...")

        # Add JSD scores to each subreddit
        for stats in successful_results:
            rule_matches = stats.get('rule_matches', {})
            stats['jsd_from_uniform'] = calculate_jsd_from_uniform(rule_matches)

        # Rank subreddits by JSD (using generic ranking utility)
        def has_enough_matches(item):
            return item.get('matched_comments', 0) >= MIN_MATCHED_COMMENTS

        ranked_results = rank_by_score(successful_results, 'jsd_from_uniform', ascending=True,
                                       filter_func=has_enough_matches)

        logger.info(f"Ranked {len([r for r in ranked_results if r.get('rank', 999999) != 999999])} subreddits with ≥{MIN_MATCHED_COMMENTS} matched comments")

        # Analyze rule distribution across all subreddits
        rule_analysis = analyze_rule_distribution(successful_results)

        total_comments = sum(r.get("total_comments", 0) for r in successful_results)
        total_matched = sum(r.get("matched_comments", 0) for r in successful_results)
        total_ambiguous = sum(r.get("ambiguous_matches", 0) for r in successful_results)
        overall_match_rate = (total_matched / total_comments * 100) if total_comments > 0 else 0
        overall_ambiguous_rate = (total_ambiguous / total_comments * 100) if total_comments > 0 else 0

        # Save consolidated statistics
        summary = {
            'total_subreddits_processed': len(subreddit_files),
            'successful_subreddits': len(successful_results),
            'failed_subreddits': len(failed_results),
            'total_comments': total_comments,
            'total_matched': total_matched,
            'total_ambiguous': total_ambiguous,
            'overall_match_rate': overall_match_rate,
            'overall_ambiguous_rate': overall_ambiguous_rate,
            'embedding_model': EMBEDDING_MODEL,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'cuda_devices_used': cuda_devices if cuda_devices[0] is not None else [],
            'parallel_workers': num_workers,
            'processing_time_seconds': time.time() - start_time,
            'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'rule_analysis': rule_analysis,
            'subreddit_stats': ranked_results  # Now includes ranks and JSD
        }

        summary_file = os.path.join(PATHS['data'], 'stage4_matching_summary.json')
        write_json_file(summary, summary_file)

        # Save ranked subreddits for Stage 5 (replaces the need for separate stage)
        rankings_file = os.path.join(PATHS['data'], 'subreddit_match_rankings.json')
        write_json_file(ranked_results, rankings_file)

        # Create submission IDs file for Stage 5 (replaces the need for separate stage)
        submission_ids_data = {}
        total_submission_ids = 0

        for stats in ranked_results:
            if stats.get('rank', 999999) != 999999:  # Only include ranked subreddits
                subreddit = stats['subreddit']
                submission_ids = stats.get('submission_ids', [])
                submission_ids_data[subreddit] = submission_ids
                total_submission_ids += len(submission_ids)

        submission_ids_output = {
            'metadata': {
                'total_subreddits': len(submission_ids_data),
                'sample_size_per_subreddit': MAX_MATCHED_COMMENTS,
                'min_matched_comments_threshold': MIN_MATCHED_COMMENTS,
                'random_seed': 0,
                'total_submission_ids': total_submission_ids
            },
            'subreddit_submission_ids': submission_ids_data
        }

        submission_ids_file = os.path.join(PATHS['data'], 'subreddit_submission_ids.json')
        write_json_file(submission_ids_output, submission_ids_file)

        elapsed = time.time() - start_time

        logger.info(f"🎉 Stage 4 Complete!")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info(f"📊 Processed {len(successful_results)} subreddits")
        logger.info(f"💬 Total comments: {total_comments:,}")
        logger.info(f"🎯 Total matched: {total_matched:,} ({overall_match_rate:.1f}%)")
        logger.info(f"❓ Total ambiguous: {total_ambiguous:,} ({overall_ambiguous_rate:.1f}%)")
        logger.info(f"📋 Total submission IDs: {total_submission_ids:,}")
        logger.info(f"🤖 Model: {EMBEDDING_MODEL}")
        logger.info(f"📏 Threshold: {SIMILARITY_THRESHOLD}")
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Rankings saved to: {rankings_file}")
        logger.info(f"Submission IDs saved to: {submission_ids_file}")

        if failed_results:
            logger.warning(f"⚠️  {len(failed_results)} subreddits failed:")
            for result in failed_results:
                logger.warning(f"  r/{result['subreddit']}: {result.get('error', 'Unknown error')}")

        # Show JSD ranking results
        ranked_only = [r for r in ranked_results if r.get('rank', 999999) != 999999]
        logger.info(f"🏆 Top 10 subreddits by JSD ranking (lower JSD = more uniform rule distribution):")
        logger.info(f"{'Rank':<5} {'Subreddit':<20} {'JSD':<8} {'Match%':<8} {'Matched':<8} {'Rules':<6}")
        logger.info("-" * 65)
        for stats in ranked_only[:10]:
            rank = stats.get('rank', 0)
            subreddit = stats['subreddit']
            jsd = stats.get('jsd_from_uniform', 0)
            match_pct = stats.get('match_percentage', 0)
            matched = stats.get('matched_comments', 0)
            num_rules = len(stats.get('rule_matches', {}))
            logger.info(f"{rank:<5} r/{subreddit:<19} {jsd:<8.4f} {match_pct:<8.1f} {matched:<8} {num_rules:<6}")

        # Show rule distribution summary
        logger.info(f"📋 Rule Distribution Summary:")
        logger.info(f"Total unique rules across all subreddits: {rule_analysis['total_rules']}")
        logger.info(f"Total rule matches: {rule_analysis['total_matches']:,}")
        top_rules = list(rule_analysis['top_rules'].items())[:5]
        logger.info("Top 5 most matched rules:")
        for rule_id, count in top_rules:
            logger.info(f"  Rule {rule_id}: {count:,} matches")

        log_stage_end(logger, 4, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        log_error_and_continue(logger, e, "Stage 4 execution")
        log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())