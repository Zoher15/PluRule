#!/usr/bin/env python3
"""
Stage 4: Match Comments to Rules

Uses semantic similarity to match moderator comments to subreddit rules.
Phase 1: Create similarity matrices using bucket-based parallel processing
Phase 2: Apply global thresholds and match comments to rules

Input:  top_subreddits/{subreddit}_mod_comments.jsonl.zst files + stage2_*.json
Output: matched_comments/{subreddit}_match.jsonl.zst + stats files
"""

import sys
import os
from collections import defaultdict

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import time
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (PATHS, MIN_RULES_FOR_MATCHING, EMBEDDING_MODEL, MIN_MATCHED_COMMENTS,
                   GOLD_PERCENTILE, AMBIGUOUS_PERCENTILE, create_directories)
from utils.logging import get_stage_logger, log_stage_start, log_stage_end
from utils.files import read_json_file, write_json_file, read_zst_lines, json_loads, write_zst_json_objects
from utils.reddit import extract_submission_id, validate_comment_structure
from utils.stats import calculate_jsd_from_uniform, rank_by_score, analyze_rule_distribution

import matplotlib.pyplot as plt
import numpy as np
import torch
import subprocess
import glob


def extract_submission_ids(comments: List[Dict[str, Any]]) -> List[str]:
    """Extract unique submission IDs from comments."""
    submission_ids = set()
    for comment in comments:
        if 'submission_id' in comment:
            submission_ids.add(comment['submission_id'])
        else:
            submission_id = extract_submission_id(comment.get('link_id', ''))
            if submission_id:
                submission_ids.add(submission_id)
    return list(submission_ids)


def get_available_cuda_devices() -> List[int]:
    """Get list of available CUDA devices."""
    try:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    except Exception:
        return []


def assign_subreddits_to_buckets(subreddits: List[tuple], num_buckets: int) -> List[List[str]]:
    """
    Assign subreddits to buckets using greedy load balancing.

    Args:
        subreddits: List of (subreddit_name, comment_count) tuples, sorted descending
        num_buckets: Number of buckets (GPUs)

    Returns:
        List of buckets, each containing list of subreddit names
    """
    buckets = [{'subreddits': [], 'total_comments': 0} for _ in range(num_buckets)]

    for subreddit, count in subreddits:
        # Assign to least-loaded bucket
        min_bucket = min(buckets, key=lambda b: b['total_comments'])
        min_bucket['subreddits'].append(subreddit)
        min_bucket['total_comments'] += count

    return [b['subreddits'] for b in buckets]


def create_distribution_plot(output_dir: str, all_similarities: np.ndarray,
                            gold_percentile: int, ambiguous_percentile: int, logger=None):
    """Create cosine similarity distribution plot with percentiles."""
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
    plt.axvline(ambiguous_threshold, color='cyan', linestyle=':', linewidth=1.5,
                label=f'{ambiguous_percentile}th percentile: {ambiguous_threshold:.3f}')
    plt.axvline(gold_threshold, color='darkred', linestyle=':', linewidth=1.5,
                label=f'{gold_percentile}th percentile: {gold_threshold:.3f}')

    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, 'cosine_similarity_distribution_all_percentiles.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"📈 Distribution plot saved to: {plot_file}")
        logger.info(f"Gold threshold ({gold_percentile}th percentile): {gold_threshold:.4f}")
        logger.info(f"Ambiguous threshold ({ambiguous_percentile}th percentile): {ambiguous_threshold:.4f}")

    return float(gold_threshold), float(ambiguous_threshold)


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
        return {'subreddit': os.path.basename(matrix_file).replace('_similarity_matrix.pt', ''),
                'error': str(e)}


def process_subreddit_matching(load_result, logger=None):
    """Apply thresholds and create matches for one subreddit."""
    try:
        subreddit_name = load_result['subreddit']
        similarity_data = load_result['similarity_data']
        similarities = similarity_data['cosine_similarity_matrix']
        gold_threshold = load_result['gold_threshold']
        ambiguous_threshold = load_result['ambiguous_threshold']

        # Load comments
        input_file = os.path.join(PATHS['top_subreddits'], f"{subreddit_name}_mod_comments.jsonl.zst")
        if not os.path.exists(input_file):
            return {"subreddit": subreddit_name, "error": f"Input file not found"}

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

        rules = load_result.get('rules', [])

        # Apply matching with global thresholds
        matched_comments = []
        rule_match_counts = defaultdict(int)
        matched_count = 0
        ambiguous_count = 0

        for i, comment in enumerate(comments):
            if i >= similarities.shape[0]:
                break

            comment_similarities = similarities[i]
            max_similarity = torch.max(comment_similarities)

            # Check for ambiguous matches (multiple rules above ambiguous threshold)
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
                    'short_name_clean': best_rule['short_name_clean'],
                    'description_clean': best_rule['description_clean'],
                    'similarity_score': float(max_similarity)
                }

                matched_comments.append(matched_comment)
                rule_match_counts[best_rule['rule_index']] += 1
                matched_count += 1

        # Save matched comments
        if matched_comments:
            match_file = os.path.join(PATHS['matched_comments'], f"{subreddit_name}_match.jsonl.zst")
            write_zst_json_objects(match_file, matched_comments)

        # Prepare statistics
        rule_matches = {str(rule['rule_index']): rule_match_counts.get(rule['rule_index'], 0)
                       for rule in rules}
        submission_ids = extract_submission_ids(matched_comments)

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

        if logger:
            logger.info(f"✅ r/{subreddit_name}: {matched_count}/{len(comments)} matched ({stats['match_percentage']:.1f}%)")

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

    logger = get_stage_logger(4, "match_rules")
    log_stage_start(logger, 4, "Match Comments to Rules")
    start_time = time.time()

    try:
        create_directories()

        # Load Stage 2 data
        logger.info("📚 Loading Stage 2 subreddit data...")
        stage2_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')
        stage2_data = read_json_file(stage2_file)

        # Build comment count lookup
        stage3_comment_counts = {
            entry['subreddit']['display_name'].lower(): entry['subreddit']['mod_comment_count']
            for entry in stage2_data.get('subreddits', [])
        }

        # Get available subreddits with comment files
        available_subreddits = []
        top_subreddits_dir = PATHS['top_subreddits']

        if not os.path.exists(top_subreddits_dir):
            logger.error(f"❌ Top subreddits directory not found: {top_subreddits_dir}")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        for entry in stage2_data['subreddits']:
            subreddit_name = entry['subreddit']['display_name'].lower()
            rule_count = len(entry.get('rules', []))

            if rule_count < MIN_RULES_FOR_MATCHING:
                continue

            comment_file = os.path.join(top_subreddits_dir, f"{subreddit_name}_mod_comments.jsonl.zst")
            if os.path.exists(comment_file):
                actual_comment_count = stage3_comment_counts.get(subreddit_name, 0)
                if actual_comment_count >= MIN_MATCHED_COMMENTS:
                    available_subreddits.append((subreddit_name, actual_comment_count))

        if not available_subreddits:
            logger.error("❌ No subreddit files found to process!")
            log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
            return 1

        # Sort by comment count (descending)
        available_subreddits.sort(key=lambda x: x[1], reverse=True)
        subreddit_names = [s[0] for s in available_subreddits]

        logger.info(f"📊 Processing {len(subreddit_names)} subreddits")
        for i, (subreddit, count) in enumerate(available_subreddits[:10]):
            logger.info(f"  {i+1:2d}. r/{subreddit}: {count:,} mod comments")
        if len(available_subreddits) > 10:
            logger.info(f"  ... and {len(available_subreddits) - 10} more subreddits")

        # Phase 1: Create similarity matrices using bucket processing
        if args.phase2_only:
            logger.info("⏭️  Skipping Phase 1 (--phase2-only flag set)")
        else:
            cuda_devices = get_available_cuda_devices()

            if cuda_devices:
                num_workers = len(cuda_devices)
                logger.info(f"🎯 Found {len(cuda_devices)} CUDA devices: {cuda_devices}")
            else:
                num_workers = 1
                cuda_devices = [None]
                logger.info(f"💻 No CUDA devices found - using CPU mode")

            # Assign subreddits to buckets
            buckets = assign_subreddits_to_buckets(available_subreddits, num_workers)

            logger.info(f"📦 Created {len(buckets)} buckets with greedy load balancing:")
            for i, bucket in enumerate(buckets):
                bucket_comments = sum(stage3_comment_counts.get(s, 0) for s in bucket)
                logger.info(f"  Bucket {i} (CUDA:{cuda_devices[i]}): {len(bucket)} subreddits, {bucket_comments:,} comments")

            # Launch bucket processors
            logger.info(f"🚀 Phase 1: Launching {num_workers} bucket processors...")
            processes = []

            for cuda_id, bucket in zip(cuda_devices, buckets):
                if not bucket:
                    continue

                cmd = [
                    sys.executable,
                    'scripts/4_match_rules_bucket.py',
                    '--cuda-device', str(cuda_id) if cuda_id is not None else "None",
                    '--subreddits', ','.join(bucket)
                ]

                logger.info(f"🚀 Starting bucket on CUDA:{cuda_id} with {len(bucket)} subreddits")
                p = subprocess.Popen(cmd)
                processes.append((p, cuda_id, len(bucket)))

            # Wait for all buckets to complete
            for p, cuda_id, count in processes:
                p.wait()
                if p.returncode == 0:
                    logger.info(f"✅ Bucket CUDA:{cuda_id} completed ({count} subreddits)")
                else:
                    logger.error(f"❌ Bucket CUDA:{cuda_id} failed (exit code: {p.returncode})")

        # Phase 2: Load similarity matrices and compute global thresholds
        logger.info("🚀 Phase 2: Loading similarity matrices and computing global thresholds...")

        output_dir = PATHS.get('matched_comments')
        matrix_files = glob.glob(os.path.join(output_dir, '*_similarity_matrix.pt'))
        logger.info(f"Found {len(matrix_files)} similarity matrices")

        # Load all matrices
        logger.info(f"Loading {len(matrix_files)} similarity matrices...")
        phase2_loads = []
        for i, matrix_file in enumerate(matrix_files):
            if i % 200 == 0:
                logger.info(f"Loading matrix {i+1}/{len(matrix_files)}")
            result = load_similarity_matrix(matrix_file)
            phase2_loads.append(result)

        successful_loads = [r for r in phase2_loads if 'error' not in r]
        failed_loads = [r for r in phase2_loads if 'error' in r]

        if failed_loads:
            logger.warning(f"⚠️  Failed to load {len(failed_loads)} matrices")

        # Compute global thresholds
        logger.info("🧮 Computing global thresholds from all similarity scores...")
        all_similarity_scores = np.concatenate([r['all_scores'] for r in successful_loads])

        gold_threshold, ambiguous_threshold = create_distribution_plot(
            output_dir, all_similarity_scores, GOLD_PERCENTILE, AMBIGUOUS_PERCENTILE, logger
        )

        # Load rules for matching
        logger.info("📚 Loading rules from Stage 2 data...")
        subreddit_rules = {}
        for entry in stage2_data['subreddits']:
            subreddit_name = entry['subreddit']['display_name'].lower()
            subreddit_rules[subreddit_name] = entry['rules']

        # Add thresholds and rules to each load result
        for load_result in successful_loads:
            load_result['gold_threshold'] = gold_threshold
            load_result['ambiguous_threshold'] = ambiguous_threshold
            subreddit_name = load_result['subreddit']
            load_result['rules'] = subreddit_rules.get(subreddit_name, [])

        # Process matching sequentially
        logger.info(f"🚀 Phase 2: Processing {len(successful_loads)} subreddits...")
        phase2_results = []

        for i, load_result in enumerate(successful_loads):
            result = process_subreddit_matching(load_result, logger)
            phase2_results.append(result)

            if (i + 1) % 50 == 0 or (i + 1) == len(successful_loads):
                logger.info(f"📊 Completed {i + 1}/{len(successful_loads)} ({(i + 1)/len(successful_loads)*100:.1f}%)")

        # Aggregate results
        successful_results = [r for r in phase2_results if not r.get("error")]
        failed_results = [r for r in phase2_results if r.get("error")]

        # Calculate JSD and rank subreddits
        logger.info("🔄 Calculating JSD and ranking subreddits...")

        for stats in successful_results:
            rule_matches = stats.get('rule_matches', {})
            if rule_matches:
                stats['jsd_from_uniform'] = calculate_jsd_from_uniform(rule_matches)
            else:
                stats['jsd_from_uniform'] = float('inf')

        def has_enough_matches(item):
            return item.get('matched_comments', 0) >= MIN_MATCHED_COMMENTS

        ranked_results = rank_by_score(successful_results, 'jsd_from_uniform', ascending=True,
                                      filter_func=has_enough_matches)

        logger.info(f"Ranked {len([r for r in ranked_results if r.get('rank', 999999) != 999999])} subreddits")

        # Remove submission_ids from summary for cleaner output
        clean_ranked_results = [{k: v for k, v in stats.items() if k != 'submission_ids'}
                               for stats in ranked_results]

        # Analyze rule distribution
        rule_analysis = analyze_rule_distribution(successful_results)

        total_comments = sum(r.get("total_comments", 0) for r in successful_results)
        total_matched = sum(r.get("matched_comments", 0) for r in successful_results)
        total_ambiguous = sum(r.get("ambiguous_matches", 0) for r in successful_results)
        overall_match_rate = (total_matched / total_comments * 100) if total_comments > 0 else 0
        overall_ambiguous_rate = (total_ambiguous / total_comments * 100) if total_comments > 0 else 0

        # Save summary
        summary = {
            'total_subreddits_processed': len(subreddit_names),
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
            'processing_time_seconds': time.time() - start_time,
            'collection_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'rule_analysis': rule_analysis,
            'subreddit_stats': clean_ranked_results
        }

        summary_file = os.path.join(PATHS['data'], 'stage4_matching_summary.json')
        write_json_file(summary, summary_file, pretty=True)

        # Create submission IDs file for Stage 5
        submission_ids_data = {}
        total_submission_ids = 0

        for stats in ranked_results:
            if stats.get('rank', 999999) != 999999:
                subreddit = stats['subreddit']
                submission_ids = stats.get('submission_ids', [])
                submission_ids_data[subreddit] = submission_ids
                total_submission_ids += len(submission_ids)

        submission_ids_output = {
            'metadata': {
                'total_subreddits': len(submission_ids_data),
                'sampling': 'none',
                'min_matched_comments_threshold': MIN_MATCHED_COMMENTS,
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
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Submission IDs saved to: {submission_ids_file}")

        if failed_results:
            logger.warning(f"⚠️  {len(failed_results)} subreddits failed:")
            for result in failed_results[:10]:
                logger.warning(f"  r/{result['subreddit']}: {result.get('error', 'Unknown error')}")

        # Show top 10 by JSD ranking
        ranked_only = [r for r in ranked_results if r.get('rank', 999999) != 999999]
        logger.info(f"🏆 Top 10 subreddits by JSD ranking:")
        logger.info(f"{'Rank':<5} {'Subreddit':<20} {'JSD':<8} {'Match%':<8} {'Matched':<8}")
        logger.info("-" * 60)
        for stats in ranked_only[:10]:
            rank = stats.get('rank', 0)
            subreddit = stats['subreddit']
            jsd = stats.get('jsd_from_uniform', 0)
            match_pct = stats.get('match_percentage', 0)
            matched = stats.get('matched_comments', 0)
            logger.info(f"{rank:<5} r/{subreddit:<19} {jsd:<8.4f} {match_pct:<8.1f} {matched:<8}")

        log_stage_end(logger, 4, success=True, elapsed_time=elapsed)
        return 0

    except Exception as e:
        logger.error(f"❌ Stage 4 execution failed: {e}", exc_info=True)
        log_stage_end(logger, 4, success=False, elapsed_time=time.time() - start_time)
        return 1


if __name__ == "__main__":
    exit(main())
