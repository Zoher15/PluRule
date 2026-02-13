#!/usr/bin/env python3
"""
Generate job lines for the Reddit moderation evaluation pipeline.

This script creates job lines in the format required by the job scheduler:
GPU_ID, script_path, conda_env, interpreter, arguments

Usage:
    python generate_jobs.py > jobs.txt
"""

import argparse
from pathlib import Path
from itertools import product

# Configuration
SCRIPT_PATH = "/data3/zkachwal/reddit-mod-collection-pipeline/eval/evaluate.py"
CONDA_ENV = "reddit-mod-pipeline"
INTERPRETER = "python"

# Evaluation configurations
# Model configurations: name -> {ncudas, priority}
# Priority: 0 is highest, higher numbers are lower priority
MODELS = {
    'qwen3-vl-4b-instruct': {'ncudas': 1, 'priority': 0},
    'qwen3-vl-8b-instruct': {'ncudas': 1, 'priority': 1},
    'qwen3-vl-30b-instruct': {'ncudas': 1, 'priority': 2},
    'qwen3-vl-4b-thinking': {'ncudas': 1, 'priority': 0},
    'qwen3-vl-8b-thinking': {'ncudas': 1, 'priority': 1},
    'qwen3-vl-30b-thinking': {'ncudas': 1, 'priority': 2},
    # 'gpt5.2-low': {'ncudas': 1, 'priority': 0},
    # 'gpt5.2-high': {'ncudas': 1, 'priority': 1},
}

SPLITS = [
    'test',
]

CONTEXTS = [
    'submission-media-discussion-user',
    'submission-discussion-user',
    'submission-discussion',
    'discussion',
    'none'
]

PHRASES = [
    'baseline'
]

MODES = [
    'prefill'
]

# Optional flags
MAX_RESPONSE_TOKENS = 2048
DEBUG = False
OVERRIDE = True


def generate_job_lines(
    models=None,
    splits=None,
    contexts=None,
    phrases=None,
    modes=None,
    debug=False,
    override=True,
    max_response_tokens=2048
):
    """
    Generate job lines for all combinations of parameters.

    Args:
        models: List of models to evaluate (default: all)
        splits: List of dataset splits (default: all)
        contexts: List of context types (default: all)
        phrases: List of phrases (default: all)
        modes: List of modes (default: all)
        debug: Whether to use debug mode
        override: Whether to override existing results
        max_response_tokens: Maximum response tokens
    """
    # Use defaults if not specified
    models = models or MODELS
    splits = splits or SPLITS
    contexts = contexts or CONTEXTS
    phrases = phrases or PHRASES
    modes = modes or MODES

    job_lines = []

    # Generate all combinations
    for model_name, split, context, phrase, mode in product(models, splits, contexts, phrases, modes):
        # Skip duplicate baseline modes (baseline is the same for prefill and prompt)
        if phrase == 'baseline' and mode != 'prefill':
            continue

        # Get model configuration
        model_config = MODELS[model_name]
        ncudas = model_config['ncudas']
        priority = model_config['priority']

        # Build arguments
        args = [
            f"--model {model_name}",
            f"--split {split}",
            f"--context {context}",
            f"--phrase {phrase}",
            f"--mode {mode}",
            f"--max-response-tokens {max_response_tokens}",
        ]

        if debug:
            args.append("--debug")

        if override:
            args.append("--override")

        args_str = " ".join(args)

        # Create job line
        # Format: priority, script_path, conda_env, interpreter, arguments (with --ncudas)
        args_str_with_ncudas = f"--ncudas {ncudas} {args_str}"
        job_line = f"{priority},{SCRIPT_PATH},{CONDA_ENV},{INTERPRETER},{args_str_with_ncudas}"
        job_lines.append(job_line)

    return job_lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate job lines for Reddit moderation evaluation"
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(MODELS.keys()),
        help='Models to evaluate (default: all)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        choices=SPLITS,
        help='Dataset splits (default: all)'
    )

    parser.add_argument(
        '--contexts',
        nargs='+',
        help='Context types (default: all)'
    )

    parser.add_argument(
        '--phrases',
        nargs='+',
        choices=PHRASES,
        help='Phrases (default: all)'
    )

    parser.add_argument(
        '--modes',
        nargs='+',
        choices=MODES,
        help='Modes (default: all)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Use debug mode (5 samples only)'
    )

    parser.add_argument(
        '--no-override',
        action='store_true',
        help='Do not override existing results'
    )

    parser.add_argument(
        '--max-response-tokens',
        type=int,
        default=2048,
        help='Maximum response tokens (default: 2048)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='reddit_jobs.txt',
        help='Output file (default: reddit_jobs.txt)'
    )

    args = parser.parse_args()

    # Generate job lines
    job_lines = generate_job_lines(
        models=args.models,
        splits=args.splits,
        contexts=args.contexts,
        phrases=args.phrases,
        modes=args.modes,
        debug=args.debug,
        override=not args.no_override,
        max_response_tokens=args.max_response_tokens
    )

    # Print header
    header = "# Reddit Moderation Evaluation Jobs"

    # Write to file
    with open(args.output, 'w') as f:
        f.write(header + '\n')
        f.write(f"# Total jobs: {len(job_lines)}\n")
        f.write("#\n")
        for line in job_lines:
            f.write(line + '\n')

    print(f"Generated {len(job_lines)} job lines → {args.output}")


if __name__ == "__main__":
    main()
