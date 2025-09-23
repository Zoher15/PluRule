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

# Set vLLM multiprocessing method BEFORE any imports to prevent SIGABRT cleanup issues
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, PROCESSES, TOP_N_SUBREDDITS_WITH_MOD_COMMENTS,
                   SIMILARITY_THRESHOLD, EMBEDDING_MODEL, MIN_MATCHED_COMMENTS,
                   MAX_MATCHED_COMMENTS, create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_progress, log_stats, log_error_and_continue
from utils.files import (read_json_file, write_json_file,
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


def load_subreddit_rules(logger) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Load subreddit rules and comment counts from Stage 2 output."""
    rules_file = os.path.join(PATHS['data'], f'stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json')

    if not os.path.exists(rules_file):
        logger.error(f"❌ Rules file not found: {rules_file}")
        return {}

    logger.info(f"Loading subreddit rules from: {rules_file}")
    data = read_json_file(rules_file)

    subreddit_rules = {}
    subreddit_comment_counts = {}
    for entry in data['subreddits']:
        subreddit_data = entry['subreddit']
        subreddit_name = subreddit_data.get('display_name', '').lower()

        if not subreddit_name:
            continue

        # Store comment count
        subreddit_comment_counts[subreddit_name] = subreddit_data.get('mod_comment_count', 0)

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
    return subreddit_rules, subreddit_comment_counts


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


def is_cuda_memory_available(device_id: int, threshold: float = 0.85) -> bool:
    """Check if CUDA device has enough free memory (less than threshold used)."""
    try:
        import torch
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            memory_usage = allocated_memory / total_memory
            return memory_usage < threshold
        return False
    except Exception:
        return False

def get_available_cuda_devices() -> List[int]:
    """Get list of available CUDA devices."""
    if not EMBEDDINGS_AVAILABLE:
        return []

    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            all_devices = list(range(device_count))
            # Temporarily skip CUDA 0 and 4 (they are busy)
            excluded_devices = {4}
            return [device for device in all_devices if device not in excluded_devices]
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
                gpu_memory_utilization=0.93,  # Lower to prevent memory fragmentation and cleanup issues
                enforce_eager=True,
                max_model_len=max_model_len + 50,  # Add buffer for safety
                seed=0  # Ensure deterministic results
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            self.model = None

    def match_comments_to_rules_pretokenized(self, comments: List[Dict[str, Any]], rules: List[Dict[str, Any]],
                                           tokenized_comments: List[List[int]], tokenized_rules: List[List[int]]) -> tuple:
        """Match comments to rules using cosine similarity with pretokenized inputs."""
        if not self.model:
            print("❌ CRITICAL: Model is None - cannot perform matching")
            raise RuntimeError("Model initialization failed - cannot perform matching")

        if not comments or not rules:
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
                    # Check for ambiguous matches (multiple rules above threshold)
                    above_threshold = comment_similarities >= self.similarity_threshold

                    if torch.sum(above_threshold) > 1:
                        # Multiple rules above threshold - ambiguous match, skip
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

            # Prepare statistics with all rules (including those with 0 matches)
            rule_matches = {}
            for rule in rules:
                rule_index = rule['rule_index']
                rule_matches[str(rule_index)] = rule_match_counts.get(rule_index, 0)

            # Sort by rule index (convert back to int for sorting, then to string)
            rule_matches = dict(sorted(rule_matches.items(), key=lambda x: int(x[0])))

            stats = {
                "total_comments": len(comments),
                "matched_comments": matched_count,
                "ambiguous_matches": ambiguous_count,
                "match_percentage": (matched_count / len(comments) * 100) if comments else 0.0,
                "ambiguous_percentage": (ambiguous_count / len(comments) * 100) if comments else 0.0,
                "rule_matches": rule_matches
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


def main():
    """Main execution function."""
    # No longer need multiprocessing start method since we use subprocess

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

        # Load subreddit rules and comment counts
        logger.info("📚 Loading subreddit rules...")
        subreddit_rules, subreddit_comment_counts = load_subreddit_rules(logger)

        if not subreddit_rules:
            logger.error("❌ No subreddit rules loaded!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Get subreddit files to process, sorted by mod comment count (highest first)
        available_subreddits = []
        top_subreddits_dir = PATHS['top_subreddits']

        if not os.path.exists(top_subreddits_dir):
            logger.error(f"❌ Top subreddits directory not found: {top_subreddits_dir}")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        for filename in os.listdir(top_subreddits_dir):
            if filename.endswith('_mod_comments.jsonl.zst'):
                subreddit = filename.replace('_mod_comments.jsonl.zst', '')
                if subreddit in subreddit_rules:
                    # Skip subreddits with only one rule (no ambiguity to resolve)
                    rule_count = len(subreddit_rules[subreddit])
                    if rule_count <= 1:
                        logger.info(f"⏭️  Skipping r/{subreddit}: only {rule_count} rule(s)")
                        continue

                    comment_count = subreddit_comment_counts.get(subreddit, 0)
                    available_subreddits.append((subreddit, comment_count))

        if not available_subreddits:
            logger.error("❌ No subreddit files found to process!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Sort by mod comment count (highest first)
        available_subreddits.sort(key=lambda x: x[1], reverse=True)
        subreddit_files = [subreddit for subreddit, _ in available_subreddits]

        logger.info(f"📊 Processing {len(subreddit_files)} subreddits (sorted by mod comment count):")
        for i, (subreddit, count) in enumerate(available_subreddits[:10]):  # Show top 10
            logger.info(f"  {i+1:2d}. r/{subreddit}: {count:,} mod comments")
        if len(available_subreddits) > 10:
            logger.info(f"  ... and {len(available_subreddits) - 10} more subreddits")

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

        logger.info("🚀 Processing subreddits using dynamic CUDA assignment...")

        # Save rules to temporary file for subprocess
        rules_temp_file = os.path.join(PATHS['data'], 'temp_rules.json')
        write_json_file(subreddit_rules, rules_temp_file, pretty=False)

        # Process subreddits using subprocess calls with dynamic CUDA assignment
        import subprocess
        import concurrent.futures
        import queue

        def run_single_subreddit(subreddit_name, cuda_device):
            cmd = [
                sys.executable,
                'scripts/4_match_rules_single.py',
                subreddit_name,
                str(cuda_device) if cuda_device is not None else "None",
                rules_temp_file
            ]

            try:
                logger.info(f"🚀 Starting r/{subreddit_name} on CUDA:{cuda_device}")
                # Redirect subprocess output to /dev/null to prevent cluttering main process stdout
                result = subprocess.run(cmd, timeout=3600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if result.returncode == 0:
                    logger.info(f"✅ Completed r/{subreddit_name}")
                    # Load the stats file to return results
                    stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_stats.json")
                    if os.path.exists(stats_file):
                        return read_json_file(stats_file)
                    else:
                        return {"subreddit": subreddit_name, "error": "Stats file not created"}
                else:
                    logger.error(f"❌ Failed r/{subreddit_name} (exit code: {result.returncode})")
                    logger.info(f"🔍 Check logs/stage4_match_rules/subreddits/{subreddit_name}.log for details")
                    return {"subreddit": subreddit_name, "error": f"Process failed with exit code {result.returncode}"}
            except subprocess.TimeoutExpired:
                return {"subreddit": subreddit_name, "error": "Process timeout (30 min)"}
            except Exception as e:
                return {"subreddit": subreddit_name, "error": str(e)}

        # Create work queue and assign each CUDA device as a worker
        subreddit_queue = queue.Queue()
        for subreddit in subreddit_files:
            subreddit_queue.put(subreddit)

        def worker_with_cuda(cuda_device):
            """Worker function that continuously processes subreddits on assigned CUDA device"""
            results = []
            while True:
                try:
                    subreddit_name = subreddit_queue.get(timeout=1)  # 1 second timeout
                except queue.Empty:
                    break  # No more subreddits to process

                # Check if CUDA device memory is available before processing
                if not is_cuda_memory_available(cuda_device):
                    logger.warning(f"⚠️ CUDA:{cuda_device} memory usage > 95%, skipping {subreddit_name}")
                    result = {"subreddit": subreddit_name, "error": "CUDA device memory not available"}
                else:
                    # Process this subreddit on the assigned CUDA device
                    result = run_single_subreddit(subreddit_name, cuda_device)
                results.append(result)
                subreddit_queue.task_done()
            return results

        # Use ThreadPoolExecutor with exactly num_workers (one per CUDA device)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit one worker per CUDA device
            future_to_cuda = {executor.submit(worker_with_cuda, cuda_device): cuda_device
                             for cuda_device in cuda_devices}

            # Collect all results
            all_results = []
            for future in concurrent.futures.as_completed(future_to_cuda):
                cuda_device = future_to_cuda[future]
                try:
                    worker_results = future.result()
                    all_results.extend(worker_results)
                    print(f"🏁 CUDA:{cuda_device} worker completed")
                except Exception as e:
                    print(f"❌ CUDA:{cuda_device} worker failed: {e}")

        results = all_results

        # Cleanup temp file
        if os.path.exists(rules_temp_file):
            os.remove(rules_temp_file)

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
        write_json_file(summary, summary_file, pretty=True)

        # Save ranked subreddits for Stage 5 (replaces the need for separate stage)
        rankings_file = os.path.join(PATHS['data'], 'subreddit_match_rankings.json')
        write_json_file(ranked_results, rankings_file, pretty=True)

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
        write_json_file(submission_ids_output, submission_ids_file, pretty=True)

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