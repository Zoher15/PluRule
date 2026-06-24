#!/usr/bin/env python3
"""
PluRule Evaluation Script

Evaluate vision-language and API models on PluRule with clustered datasets
and multimodal Reddit discussion context.

Features:
- Multiple model support: Qwen3-VL via vLLM and OpenAI API models
- Configurable context types: control what information is exposed in prompts
- Phrase variations: baseline, chain-of-thought, etc.
- Two-stage evaluation: reasoning generation + clean answer extraction
- Comprehensive metrics: overall, per-rule-cluster, per-subreddit-cluster accuracy

Usage:
    python evaluate.py --model qwen3-vl-8b-instruct --split test --context submission-discussion --phrase cot --mode prefill
    python evaluate.py --model qwen3-vl-30b-thinking --split val --context submission-media-discussion-user --phrase baseline --mode prefill
    python evaluate.py --model qwen3-vl-8b-instruct --split test --context none --phrase analyze --mode prompt --debug
"""

import argparse
import json
import sys
import os
from pathlib import Path

# =============================================================================
# EARLY ENVIRONMENT SETUP (before any imports that might use GPU)
# =============================================================================

# Parse CUDA argument early to set CUDA_VISIBLE_DEVICES before importing vLLM
def _parse_cuda_arg() -> str:
    """Parse --cuda argument early, before heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cuda', type=str, default='1')
    args, _ = parser.parse_known_args()
    return args.cuda

# Set CUDA devices BEFORE importing any GPU-related modules
os.environ['CUDA_VISIBLE_DEVICES'] = _parse_cuda_arg()

# Fix MKL threading and set vLLM environment variables
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add eval directory to path
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# Now safe to import config and helpers (which may import vLLM)
import config
import helpers


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PluRule Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        choices=config.get_supported_models(),
        help='Model to evaluate'
    )

    parser.add_argument(
        '--split', '-s',
        type=str,
        required=True,
        choices=list(config.DATASET_FILES.keys()),
        help='Dataset split to evaluate on'
    )

    parser.add_argument(
        '--context', '-c',
        type=str,
        required=True,
        help='Context flags (dash-separated): none, submission, media, discussion, user. Examples: "none", "submission-media", "submission-media-discussion-user"'
    )

    parser.add_argument(
        '--phrase', '-p',
        type=str,
        required=True,
        choices=config.get_supported_phrases(),
        help='Phrase to use (e.g., cot, baseline)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='prefill',
        choices=config.PHRASE_MODES,
        help='Phrase mode: prefill (append after template) or prompt (append to question)'
    )

    parser.add_argument(
        '--cuda',
        type=str,
        default='1',
        help='CUDA device IDs for vLLM (e.g., "0,1")'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with only 5 thread pairs'
    )

    parser.add_argument(
        '--override',
        action='store_true',
        help='Override existing results if they exist'
    )

    parser.add_argument(
        '--max-response-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for response generation (default: 2048)'
    )

    parser.add_argument(
        '--instruct',
        action='store_true',
        help='Render chat templates in instruct mode by disabling/stripping Qwen thinking tokens'
    )

    parser.add_argument(
        '--rag-k',
        type=int,
        default=0,
        help='Number of retrieved few-shot examples to prepend per target thread'
    )

    parser.add_argument(
        '--rag-retrieval-path',
        type=Path,
        default=None,
        help='Path to a precomputed RAG target-comment similarity artifact'
    )

    parser.add_argument(
        '--rag-filter',
        type=str,
        default='none',
        choices=['none', 'subreddit', 'subreddit-cluster', 'rule-cluster'],
        help='Filter retrieved examples before selecting top matches'
    )

    parser.add_argument(
        '--rag-balance',
        type=str,
        default='mixed',
        choices=['mixed', 'top', 'random'],
        help='Few-shot selection policy: mixed balances nearest neighbors, top uses nearest neighbors only, random samples deterministic source target comments without the similarity artifact'
    )

    parser.add_argument(
        '--rag-source-split',
        type=str,
        default='train',
        choices=list(config.DATASET_FILES.keys()),
        help='Candidate split used by RAG'
    )

    parser.add_argument(
        '--rag-trace-path',
        type=Path,
        default=Path(os.environ.get(
            'PLURULE_RAG_TRACE_PATH',
            '/home/exouser/discloze/data/traces/train/full.jsonl'
        )),
        help='Discloze trace JSONL whose rationales should be used for retrieved few-shot examples'
    )

    parser.add_argument(
        '--rag-trace-style',
        type=str,
        default=None,
        choices=['response-only', 'rationale-think', 'rationale-plain', 'template'],
        help='Few-shot trace assistant-turn format (required when --rag-k > 0)'
    )

    args = parser.parse_args()
    if args.rag_k > 0 and args.model != 'rag-vote' and not args.rag_trace_style:
        parser.error('--rag-trace-style is required when RAG is enabled (--rag-k > 0)')
    return args


def _log_section(logger, title: str) -> None:
    """Log a section header."""
    logger.info("\n" + "="*80)
    logger.info(title)
    logger.info("="*80)


def _append_run_suffix(run_suffix: str, child: str) -> str:
    return f"{run_suffix}/{child}" if run_suffix else child


def _display_top_clusters(logger, metrics: dict, cluster_type: str, n: int = 5) -> None:
    """
    Display top N clusters by count.

    Args:
        logger: Logger instance
        metrics: Metrics dictionary
        cluster_type: 'rule' or 'subreddit'
        n: Number of top clusters to display
    """
    cluster_key = f'per_{cluster_type}_cluster'
    logger.info(f"Top {n} {cluster_type.title()} Clusters by Count:")

    clusters = sorted(
        metrics[cluster_key].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:n]

    for cluster_label, stats in clusters:
        logger.info(f"  {cluster_label}:")
        logger.info(f"    Overall: {stats['overall_accuracy']:.4f} ({stats['count']} pairs)")
        logger.info(f"    Violating: {stats['violating_accuracy']:.4f}, Compliant: {stats['compliant_accuracy']:.4f}")


def main():
    """Main evaluation pipeline."""

    # Parse arguments
    args = parse_arguments()

    if args.rag_k < 0:
        print("--rag-k must be non-negative", file=sys.stderr)
        sys.exit(2)

    # Baseline models pin the RAG settings they were designed for (see BASELINE_MODELS).
    model_config = config.get_model_config(args.model)
    for arg_name, value in model_config.get('forced_args', {}).items():
        if arg_name == 'rag_k' and args.rag_k > 0:
            continue
        if arg_name == 'rag_balance' and args.rag_balance == 'random':
            continue
        setattr(args, arg_name, value)

    # Validate arguments
    try:
        config.validate_config_combination(
            args.model,
            args.split,
            args.context,
            args.phrase,
            args.mode
        )
    except ValueError as e:
        print(f"Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Normalize mode for baseline phrase (mode is ignored for baseline)
    baseline_mode_normalized = args.phrase == 'baseline' and args.mode != 'prefill'
    if args.phrase == 'baseline':
        args.mode = 'prefill'  # Normalize to prefill to avoid duplicate processing

    rag_retrieval_path = None
    rag_artifact_sha256 = None
    if args.rag_k > 0 and args.rag_balance != 'random':
        rag_retrieval_path = args.rag_retrieval_path or helpers.default_rag_retrieval_path(
            args.split,
            args.rag_source_split
        )
        try:
            rag_artifact_sha256 = helpers.rag_artifact_sha256(rag_retrieval_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Invalid RAG retrieval artifact: {e}", file=sys.stderr)
            sys.exit(1)

    run_suffix = helpers.build_rag_run_suffix(
        args.rag_k,
        args.rag_filter,
        args.rag_balance,
        source_split=args.rag_source_split,
        artifact_sha256=rag_artifact_sha256,
        trace_style=args.rag_trace_style
    )
    if args.instruct:
        run_suffix = _append_run_suffix(run_suffix, "instruct")

    # Create logger
    logger, log_path = helpers.create_logger(
        args.split,
        args.model,
        args.context,
        args.phrase,
        args.mode,
        run_suffix=run_suffix
    )

    # Log CUDA setup (already set at module import time)
    logger.info(f"🖥️  CUDA_VISIBLE_DEVICES set to '{args.cuda}'")

    if baseline_mode_normalized:
        logger.warning(
            f"⚠️  --mode is ignored for baseline phrase. "
            f"Normalizing to 'prefill' mode. Output will be in baseline/ directory."
        )

    logger.info("="*80)
    logger.info("🚀 REDDIT MODERATION EVALUATION")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Context: {args.context}")
    logger.info(f"Phrase: {args.phrase} (mode: {args.mode})")
    logger.info(f"Instruct mode: {args.instruct}")
    if args.rag_k > 0:
        rag_log = (
            f"RAG: k={args.rag_k}, filter={args.rag_filter}, "
            f"balance={args.rag_balance}, source_split={args.rag_source_split}"
        )
        if rag_artifact_sha256:
            rag_log += f", artifact={rag_artifact_sha256[:12]}"
        logger.info(rag_log)
    else:
        logger.info("RAG: disabled")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("="*80)

    try:
        # Check if results already exist
        output_dir = config.get_dir(
            config.OUTPUT_DIR,
            args.split,
            args.model,
            args.context,
            args.phrase,
            args.mode,
            run_suffix=run_suffix
        )

        # If checkpoint has failed entries, clear stale results so retry flows naturally
        checkpoint_path = output_dir / "flex_checkpoint.json"
        if checkpoint_path.exists() and not args.override:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            failed_count = sum(
                1 for v in checkpoint_data.values()
                if (isinstance(v, dict) and not v.get('content'))
                or (isinstance(v, str) and not v)
            )
            if failed_count > 0:
                for stale in list(output_dir.glob("performance_*.json")) + list(output_dir.glob("reasoning_*.json")):
                    stale.unlink()
                logger.info(f"♻️  Checkpoint has {failed_count} failed entries — cleared stale results for retry")

        existing_performance_files = list(output_dir.glob("performance_*.json"))

        if existing_performance_files and not args.override:
            # Check if existing results are from debug mode (5 pairs)
            latest_perf_file = sorted(existing_performance_files)[-1]
            with open(latest_perf_file, 'r') as f:
                perf_data = json.load(f)
                total_pairs = perf_data.get('metrics', {}).get('overall', {}).get('total_pairs', 0)

            if total_pairs == 5:
                logger.warning(f"⚠️  Found debug results (5 pairs) in {output_dir}")
                logger.warning(f"   Current run will {'use debug mode' if args.debug else 'use full dataset'}")
                if not args.debug:
                    logger.info(f"♻️  Automatically overriding debug results with full evaluation")
                else:
                    logger.warning(f"   Use --override flag to overwrite existing debug results")
                    logger.info("❌ Evaluation skipped")
                    return
            else:
                logger.warning(f"⚠️  Results already exist in {output_dir}")
                logger.warning(f"   Found {len(existing_performance_files)} existing performance file(s) with {total_pairs} pairs")
                logger.warning(f"   Use --override flag to overwrite existing results")
                logger.info("❌ Evaluation skipped")
                return

        elif existing_performance_files and args.override:
            logger.info(f"♻️  Override mode: Will overwrite existing results in {output_dir}")

        # 1. Load dataset
        _log_section(logger, "STEP 1: LOADING DATASET")
        thread_pairs = helpers.load_dataset(args.split, logger, debug=args.debug)

        rag_examples_by_target = None
        rag_config = None
        if args.rag_k > 0:
            _log_section(logger, "STEP 2A: PREPARING RAG EXAMPLES")
            rag_trace_path = None if args.rag_trace_style == 'template' else args.rag_trace_path
            rag_config = {
                'k': args.rag_k,
                'filter': args.rag_filter,
                'balance': args.rag_balance,
                'source_split': args.rag_source_split,
                'retrieval_path': str(rag_retrieval_path) if rag_retrieval_path else None,
                'retrieval_artifact_sha256': rag_artifact_sha256,
                'trace_path': str(rag_trace_path) if rag_trace_path else None,
                'trace_style': args.rag_trace_style,
                'run_suffix': run_suffix
            }
            if rag_retrieval_path:
                logger.info(f"RAG retrieval artifact: {rag_retrieval_path}")
            rag_examples_by_target = helpers.prepare_rag_examples(
                thread_pairs=thread_pairs,
                split=args.split,
                retrieval_path=rag_retrieval_path,
                k=args.rag_k,
                filter_mode=args.rag_filter,
                balance=args.rag_balance,
                source_split=args.rag_source_split,
                trace_path=rag_trace_path,
                logger=logger
            )

        results = []

        if model_config['type'] == 'baseline':
            _log_section(logger, "STEP 2B: RAG VOTE BASELINE")
            results = helpers.evaluate_rag_vote_baseline(
                thread_pairs,
                args.split,
                rag_examples_by_target,
                logger
            )
        else:
            # 2. Build prompts
            _log_section(logger, "STEP 2B: BUILDING PROMPTS" if args.rag_k > 0 else "STEP 2: BUILDING PROMPTS")
            thread_pairs = helpers.build_prompts_for_thread_pairs(
                thread_pairs,
                args.context,
                args.phrase,
                args.model,
                args.mode,
                logger,
                split=args.split,
                rag_examples_by_target=rag_examples_by_target,
                rag_trace_style=args.rag_trace_style,
                instruct=args.instruct
            )

        # 3. Process evaluation
        if model_config['type'] == 'vllm':
            # Let vLLM handle request scheduling via max_num_seqs instead of reloading per outer batch.
            logger.info(f"\nProcessing {len(thread_pairs)} pairs with one vLLM engine")
            logger.info("="*80)

            _log_section(logger, "STEP 3: APPLYING CHAT TEMPLATES")
            thread_pairs, resource_stats = helpers.apply_chat_template(
                thread_pairs,
                args.model,
                logger,
                instruct=args.instruct
            )

            _log_section(logger, "STEP 4: TWO-STAGE EVALUATION")
            num_gpus = len(args.cuda.split(','))
            results = helpers.evaluate_two_stage_vllm(
                thread_pairs, args.model, model_config, num_gpus,
                resource_stats, args.max_response_tokens, args.context, logger,
                instruct=args.instruct
            )
        elif model_config['type'] == 'api':
            batch_size = 200  # Stage 2 batch size for local vLLM answer extraction.
            # API models: Send all data at once for Stage 1 (OpenAI handles 190MB/50K limits)
            # Stage 2 uses batch_size for local vLLM processing
            logger.info(f"\nProcessing {len(thread_pairs)} pairs via API (Stage 2 batch_size={batch_size})")
            logger.info("="*80)

            _log_section(logger, "STEP 3: BUILDING PROMPTS FOR API")
            # Note: Chat templates for API models are applied inside evaluate_two_stage_api

            _log_section(logger, "STEP 4: TWO-STAGE EVALUATION (API)")
            results = helpers.evaluate_two_stage_api(
                thread_pairs, model_config, output_dir,
                args.context, args.max_response_tokens, logger,
                override=args.override,
                stage2_batch_size=batch_size,
                instruct=args.instruct
            )

        logger.info("\n" + "="*80)
        logger.info(f"✓ Evaluation complete! Total results: {len(results)}")
        logger.info("="*80)

        # 5. Calculate metrics
        _log_section(logger, "STEP 5: CALCULATING METRICS")
        metrics = helpers.calculate_metrics(results, logger)

        # 6. Save results
        _log_section(logger, "STEP 6: SAVING RESULTS")
        reasoning_path, performance_path = helpers.save_results(
            results,
            metrics,
            output_dir,
            args.model,
            args.split,
            args.context,
            args.phrase,
            args.mode,
            logger,
            rag_config=rag_config,
            instruct=args.instruct
        )

        # 7. Display final results
        _log_section(logger, "✅ EVALUATION COMPLETE")
        logger.info("Overall Metrics:")
        logger.info(f"  Total thread pairs: {metrics['overall']['total_pairs']}")
        logger.info(f"  Total threads: {metrics['overall']['total_threads']}")
        logger.info(f"  Overall accuracy: {metrics['overall']['overall_accuracy']:.4f}")
        logger.info(f"  Violating accuracy: {metrics['overall']['violating_accuracy']:.4f}")
        logger.info(f"  Compliant accuracy: {metrics['overall']['compliant_accuracy']:.4f}")
        logger.info("")

        # Display top clusters
        _display_top_clusters(logger, metrics, 'rule')
        logger.info("")
        _display_top_clusters(logger, metrics, 'subreddit')

        logger.info("")
        logger.info(f"📊 Reasoning traces: {reasoning_path}")
        logger.info(f"📊 Performance metrics: {performance_path}")
        logger.info(f"📝 Evaluation logs: {log_path}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
