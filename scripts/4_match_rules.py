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
from collections import defaultdict

# Set vLLM multiprocessing method BEFORE any imports to prevent SIGABRT cleanup issues
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, MIN_RULES_FOR_MATCHING,
                   EMBEDDING_MODEL, MIN_MATCHED_COMMENTS,
                   MAX_MATCHED_COMMENTS, GOLD_PERCENTILE, AMBIGUOUS_PERCENTILE,
                   create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end, log_error_and_continue
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads, write_zst_json_objects
from utils.reddit import extract_submission_id, validate_comment_structure
from utils.stats import calculate_jsd_from_uniform, rank_by_score, analyze_rule_distribution

import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess
import concurrent.futures
import queue
import glob
import random


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
    try:
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

def create_distribution_plot(output_dir: str, all_similarities: List[np.ndarray], gold_percentile: int, ambiguous_percentile: int, logger=None):
    """Create cosine similarity distribution plot with percentiles."""

    # Find all .pt files
    pt_files = glob.glob(os.path.join(output_dir, '*.pt'))
    if not pt_files:
        if logger:
            logger.warning("⚠️  No .pt files found for distribution plot")
        return None, None

    if logger:
        logger.info(f"Creating distribution plot from {len(pt_files)} similarity matrices...")

    # Calculate percentiles
    p25 = np.percentile(all_similarities, 25)
    p75 = np.percentile(all_similarities, 75)
    ambiguous_threshold = np.percentile(all_similarities, ambiguous_percentile)
    gold_threshold = np.percentile(all_similarities, gold_percentile)
    mean_sim = np.mean(all_similarities)
    median_sim = np.median(all_similarities)

    # Create plot
    plt.figure(figsize=(15, 8))
    plt.hist(all_similarities, bins=100, alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Normalized Density')
    plt.title(f'Distribution of Cosine Similarity Scores\nAcross All Subreddits (n={len(all_similarities):,})')
    plt.grid(True, alpha=0.3)

    # Add statistical lines
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'Median: {median_sim:.3f}')
    plt.axvline(p25, color='orange', linestyle=':', linewidth=1.5, label=f'25th percentile: {p25:.3f}')
    plt.axvline(p75, color='purple', linestyle=':', linewidth=1.5, label=f'75th percentile: {p75:.3f}')
    plt.axvline(ambiguous_threshold, color='cyan', linestyle=':', linewidth=1.5, label=f'{ambiguous_percentile}th percentile: {ambiguous_threshold:.3f}')
    plt.axvline(gold_threshold, color='darkred', linestyle=':', linewidth=1.5, label=f'{gold_percentile}th percentile: {gold_threshold:.3f}')

    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, 'cosine_similarity_distribution_all_percentiles.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"📈 Distribution plot saved to: {plot_file}")
        logger.info(f"Gold threshold ({GOLD_PERCENTILE}th percentile): {gold_threshold:.4f}")
        logger.info(f"Ambiguous threshold ({AMBIGUOUS_PERCENTILE}th percentile): {ambiguous_threshold:.4f}")
    return float(gold_threshold), float(ambiguous_threshold)  # Return thresholds for use in matching

def load_similarity_matrix(matrix_file):
    """Load a single similarity matrix and extract all scores."""
    try:
        subreddit_name = os.path.basename(matrix_file).replace('_similarity_matrix.pt', '')
        similarity_data = torch.load(matrix_file, map_location='cpu')
        similarities = similarity_data['cosine_similarity_matrix']
        all_scores = similarities.flatten().numpy()

        return {
            'subreddit': subreddit_name,
            'similarity_data': similarity_data,
            'all_scores': all_scores
        }
    except Exception as e:
        return {'subreddit': os.path.basename(matrix_file).replace('_similarity_matrix.pt', ''), 'error': str(e)}

def process_subreddit_matching(load_result, logger=None):
    """Apply thresholds and create matches for one subreddit."""
    try:
        subreddit_name = load_result['subreddit']
        similarity_data = load_result['similarity_data']
        similarities = similarity_data['cosine_similarity_matrix']

        # Get thresholds from the load_result (passed from Phase 2)
        gold_threshold = load_result.get('gold_threshold')
        ambiguous_threshold = load_result.get('ambiguous_threshold')

        # Load comments
        input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
        if not os.path.exists(input_file):
            return {"subreddit": subreddit_name, "error": f"Input file not found: {input_file}"}

        lines = read_zst_lines(input_file)
        comments = []
        for line in lines:
            if line.strip():
                try:
                    comment = json_loads(line)
                    if validate_comment_structure(comment):
                        comments.append(comment)
                except Exception:
                    continue

        # Get pre-loaded rules from load_result
        rules = load_result.get('rules', [])

        # Apply matching with global thresholds
        matched_comments = []
        rule_match_counts = defaultdict(int)
        matched_count = 0
        ambiguous_count = 0

        for i, comment in enumerate(comments):
            if i >= similarities.shape[0]:
                break  # Don't go beyond similarity matrix bounds

            comment_similarities = similarities[i]
            max_similarity = torch.max(comment_similarities)

            # Check for ambiguous matches
            above_ambiguous = comment_similarities > ambiguous_threshold
            num_above_ambiguous = torch.sum(above_ambiguous)

            if num_above_ambiguous > 1:
                ambiguous_count += 1
                continue

            # Check gold threshold
            if max_similarity >= gold_threshold:
                best_rule_idx = torch.argmax(comment_similarities).item()
                best_rule = rules[best_rule_idx]

                matched_comment = comment.copy()
                matched_comment['matched_rule'] = {
                    'rule_index': best_rule['rule_index'],
                    'short_name': best_rule['short_name_clean'],
                    'description': best_rule['description_clean'],
                    'similarity_score': float(max_similarity)  # Convert to Python float
                }

                matched_comments.append(matched_comment)
                rule_match_counts[best_rule['rule_index']] += 1
                matched_count += 1

        # Prepare statistics
        rule_matches = {}
        for rule in rules:
            rule_index = rule['rule_index']
            rule_matches[str(rule_index)] = rule_match_counts.get(rule_index, 0)

        submission_ids = extract_submission_ids(matched_comments)

        # Sample if too many
        if len(matched_comments) > MAX_MATCHED_COMMENTS:
            random.seed(0)
            matched_comments = random.sample(matched_comments, MAX_MATCHED_COMMENTS)

        # Save files
        if matched_comments:
            match_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
            write_zst_json_objects(match_file, matched_comments)

        stats = {
            "subreddit": subreddit_name,
            "total_comments": len(comments),
            "total_rules": len(rules),
            "matched_comments": matched_count,
            "ambiguous_matches": ambiguous_count,
            "match_percentage": (matched_count / len(comments) * 100) if comments else 0.0,
            "ambiguous_percentage": (ambiguous_count / len(comments) * 100) if comments else 0.0,
            "rule_matches": rule_matches,
            "gold_threshold": float(gold_threshold),
            "ambiguous_threshold": float(ambiguous_threshold),
            "submission_ids": submission_ids,
            "sampled_comments_count": len(matched_comments)
        }

        stats_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_stats.json")
        write_json_file(stats, stats_file, pretty=True)

        # Log progress for this subreddit
        if logger:
            logger.info(f"✅ r/{subreddit_name}: {matched_count}/{len(comments)} matched ({stats['match_percentage']:.1f}%), {len(rules)} rules")

        return stats

    except Exception as e:
        return {"subreddit": load_result['subreddit'], "error": str(e)}

def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Match Comments to Rules")
    parser.add_argument("--phase2-only", action="store_true",
                       help="Skip Phase 1 (similarity matrix creation) and run only Phase 2 (matching)")
    args = parser.parse_args()

    # Initialize logging
    logger = get_stage_logger(4, "match_rules")
    log_stage_start(logger, 4, "Match Comments to Rules")

    start_time = time.time()

    try:
        # Create directories
        create_directories()

        # Load Stage 2 data for comment counts and rule validation
        logger.info("📚 Loading subreddit data...")
        rules_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
        stage2_data = read_json_file(rules_file)

        # Load Stage 3 summary for actual filtered comment counts
        logger.info("📚 Loading Stage 3 summary for comment counts...")
        stage3_summary_file = os.path.join(PATHS['data'], 'stage3_filter_and_consolidate_summary.json')
        stage3_data = read_json_file(stage3_summary_file)
        stage3_comment_counts = stage3_data.get('subreddit_details', {})

        # Get subreddits with >1 rule and existing comment files, sorted by mod comment count (highest first)
        available_subreddits = []
        skipped_low_comments = []
        top_subreddits_dir = PATHS['top_subreddits']

        if not os.path.exists(top_subreddits_dir):
            logger.error(f"❌ Top subreddits directory not found: {top_subreddits_dir}")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        for entry in stage2_data['subreddits']:
            subreddit_name = entry['subreddit']['display_name'].lower()
            rule_count = len(entry.get('rules', []))

            # Skip subreddits with insufficient rules (should have been filtered in Stage 2, but double-check)
            if rule_count < MIN_RULES_FOR_MATCHING:
                logger.info(f"⏭️  Skipping r/{subreddit_name}: only {rule_count} rule(s) (need {MIN_RULES_FOR_MATCHING}+)")
                continue

            # Check if comment file exists
            comment_file = os.path.join(top_subreddits_dir, f"{subreddit_name}_mod_comments.jsonl.zst")
            if os.path.exists(comment_file):
                # Get actual comment count from Stage 3 (post-filtering)
                actual_comment_count = stage3_comment_counts.get(subreddit_name, {}).get('comments', 0)

                # Skip subreddits with fewer comments than MIN_MATCHED_COMMENTS
                # (they won't meet the threshold anyway, so no point processing them)
                if actual_comment_count < MIN_MATCHED_COMMENTS:
                    skipped_low_comments.append((subreddit_name, actual_comment_count))
                    continue

                available_subreddits.append((subreddit_name, actual_comment_count))

        if not available_subreddits:
            logger.error("❌ No subreddit files found to process!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Sort by mod comment count (highest first)
        available_subreddits.sort(key=lambda x: x[1], reverse=True)
        subreddit_files = [subreddit for subreddit, _ in available_subreddits]

        # Report filtering statistics
        if skipped_low_comments:
            logger.info(f"⏭️  Skipped {len(skipped_low_comments)} subreddits with <{MIN_MATCHED_COMMENTS} comments (won't meet threshold)")
            if len(skipped_low_comments) <= 20:  # Show all if 20 or fewer
                for subreddit, count in sorted(skipped_low_comments, key=lambda x: x[1], reverse=True):
                    logger.info(f"     r/{subreddit}: {count} comments")
            else:  # Show top 10 and bottom 10
                logger.info(f"     Top 10 skipped:")
                for subreddit, count in sorted(skipped_low_comments, key=lambda x: x[1], reverse=True)[:10]:
                    logger.info(f"       r/{subreddit}: {count} comments")
                logger.info(f"     ... and {len(skipped_low_comments) - 10} more")

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
            num_workers = min(1, len(subreddit_files))  # Limit to available subreddits
            cuda_devices = [None] * num_workers  # CPU mode
            logger.info(f"No CUDA devices found - using {num_workers} CPU processes")

        if args.phase2_only:
            logger.info("⏭️  Skipping Phase 1 (--phase2-only flag set)")
            all_results = []  # No Phase 1 results
        else:
            logger.info(f"Found {len(subreddit_files)} subreddit files to process")

            logger.info("🚀 Phase 1: Processing subreddits using dynamic CUDA assignment...")

            # Process subreddits using subprocess calls with dynamic CUDA assignment
            def run_single_subreddit(subreddit_name, cuda_device):
                cmd = [
                    sys.executable,
                    'scripts/4_match_rules_single.py',
                    '--subreddit', subreddit_name,
                    '--cuda-device', str(cuda_device) if cuda_device is not None else "None"
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
                        logger.info(f"🏁 CUDA:{cuda_device} worker completed")
                    except Exception as e:
                        logger.error(f"❌ CUDA:{cuda_device} worker failed: {e}")

        results = all_results

        # Phase 2: Load similarity matrices, compute thresholds, and perform matching
        logger.info("🚀 Phase 2: Loading similarity matrices and computing global thresholds...")


        # Find all similarity matrix files
        output_dir = PATHS.get('matched_comments')
        matrix_files = glob.glob(os.path.join(output_dir, '*_similarity_matrix.pt'))
        logger.info(f"Found {len(matrix_files)} similarity matrices for Phase 2")

        # Load all matrices sequentially to avoid memory issues
        logger.info(f"Loading {len(matrix_files)} similarity matrices sequentially...")
        phase2_loads = []
        for i, matrix_file in enumerate(matrix_files):
            if i % 200 == 0:  # Progress every 200 files
                logger.info(f"Loading matrix {i+1}/{len(matrix_files)}: {os.path.basename(matrix_file)}")
            try:
                result = load_similarity_matrix(matrix_file)
                phase2_loads.append(result)
            except Exception as load_error:
                logger.error(f"❌ Failed to load {matrix_file}: {load_error}")
                phase2_loads.append({'subreddit': os.path.basename(matrix_file).replace('_similarity_matrix.pt', ''), 'error': str(load_error)})

        successful_loads = [r for r in phase2_loads if 'error' not in r]
        failed_loads = [r for r in phase2_loads if 'error' in r]

        if failed_loads:
            logger.warning(f"⚠️  Failed to load {len(failed_loads)} matrices in Phase 2")

        # Compute global thresholds
        logger.info("🧮 Computing global thresholds from all similarity scores...")
        all_similarity_scores = np.concatenate([r['all_scores'] for r in successful_loads])

        gold_threshold, ambiguous_threshold = create_distribution_plot(output_dir, all_similarity_scores, GOLD_PERCENTILE, AMBIGUOUS_PERCENTILE, logger)


        logger.info(f"📊 Gold threshold ({GOLD_PERCENTILE}th percentile): {gold_threshold:.4f}")
        logger.info(f"📊 Ambiguous threshold ({AMBIGUOUS_PERCENTILE}th percentile): {ambiguous_threshold:.4f}")

        # Load Stage 2 rules once for all subreddits
        logger.info("📚 Loading rules from Stage 2 data...")
        rules_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
        stage2_data = read_json_file(rules_file)

        # Create a mapping of subreddit name to rules
        subreddit_rules = {}
        for entry in stage2_data['subreddits']:
            subreddit_name = entry['subreddit']['display_name'].lower()
            subreddit_rules[subreddit_name] = entry['rules']

        # Add thresholds and rules to each load result for the matching function
        for load_result in successful_loads:
            load_result['gold_threshold'] = gold_threshold
            load_result['ambiguous_threshold'] = ambiguous_threshold
            # Add the rules for this subreddit
            subreddit_name = load_result['subreddit']
            load_result['rules'] = subreddit_rules.get(subreddit_name, [])

        # Process matching sequentially to avoid segfaults
        logger.info(f"🚀 Phase 2: Processing {len(successful_loads)} subreddits sequentially...")

        phase2_results = []
        total = len(successful_loads)

        for i, load_result in enumerate(successful_loads):
            subreddit_name = load_result['subreddit']
            try:
                result = process_subreddit_matching(load_result, logger)
                phase2_results.append(result)

                if (i + 1) % 50 == 0 or (i + 1) == total:  # Progress every 50 subreddits
                    logger.info(f"📊 Completed {i + 1}/{total} subreddits ({(i + 1)/total*100:.1f}%) - Latest: r/{subreddit_name}")

            except Exception as e:
                logger.error(f"❌ Failed r/{subreddit_name}: {e}")
                phase2_results.append({"subreddit": subreddit_name, "error": str(e)})

        # Update results to use Phase 2 outputs
        results = phase2_results

        # Aggregate results
        successful_results = [r for r in results if not r.get("error")]
        failed_results = [r for r in results if r.get("error")]

        # Calculate JSD and rank subreddits using utilities
        logger.info("🔄 Calculating JSD and ranking subreddits...")

        # Add JSD scores to each subreddit
        for stats in successful_results:
            rule_matches = stats.get('rule_matches', {})
            if rule_matches:  # Only calculate JSD if we have rule matches
                stats['jsd_from_uniform'] = calculate_jsd_from_uniform(rule_matches)
            else:
                stats['jsd_from_uniform'] = float('inf')  # No matches = worst JSD

        # Rank subreddits by JSD (using generic ranking utility)
        def has_enough_matches(item):
            return item.get('matched_comments', 0) >= MIN_MATCHED_COMMENTS

        ranked_results = rank_by_score(successful_results, 'jsd_from_uniform', ascending=True,
                                       filter_func=has_enough_matches)

        logger.info(f"Ranked {len([r for r in ranked_results if r.get('rank', 999999) != 999999])} subreddits with ≥{MIN_MATCHED_COMMENTS} matched comments")

        # Create clean versions without submission_ids for summary/rankings files
        def remove_submission_ids(stats_list):
            """Remove submission_ids from stats for cleaner output files."""
            return [{k: v for k, v in stats.items() if k != 'submission_ids'} for stats in stats_list]

        clean_ranked_results = remove_submission_ids(ranked_results)

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
            'gold_threshold': gold_threshold,
            'ambiguous_threshold': ambiguous_threshold,
            'cuda_devices_used': cuda_devices if cuda_devices and cuda_devices[0] is not None else [],
            'parallel_workers': num_workers,
            'processing_time_seconds': time.time() - start_time,
            'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'rule_analysis': rule_analysis,
            'subreddit_stats': clean_ranked_results  # Now includes ranks and JSD
        }

        summary_file = os.path.join(PATHS['data'], 'stage4_matching_summary.json')
        write_json_file(summary, summary_file, pretty=True)

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

        submission_ids_file = os.path.join(PATHS['data'], 'stage4_subreddit_submission_ids.json')
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
        logger.info(f"📏 Gold threshold: {gold_threshold:.4f} ({GOLD_PERCENTILE}th percentile)")
        logger.info(f"📏 Ambiguous threshold: {ambiguous_threshold:.4f} ({AMBIGUOUS_PERCENTILE}th percentile)")
        logger.info(f"Summary saved to: {summary_file}")
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