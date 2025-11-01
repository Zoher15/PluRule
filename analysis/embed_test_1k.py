#!/usr/bin/env python3
"""
Embed Test_1k Communities and Rules

Creates embeddings for subreddits and rules in the test_1k dataset:
1. Subreddit embeddings: Based on title + public description from Stage 2
2. Rule embeddings: Based on rule_comprehensive for rules that appear in test_1k

Output:
- output/embeddings/test_1k_subreddit_embeddings.tsv - Embedding vectors (one per line, tab-separated)
- output/embeddings/test_1k_subreddit_metadata.tsv - Metadata (Subreddit, Language, Title, Description, FullText)
- output/embeddings/test_1k_rule_embeddings.tsv - Embedding vectors (one per line, tab-separated)
- output/embeddings/test_1k_rule_metadata.tsv - Metadata (Subreddit, ShortName, Description, FullText)
"""

import sys
import os
import json
import time
import torch
import pandas as pd
import zstandard
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

# Disable vLLM's default logging configuration
os.environ['VLLM_CONFIGURE_LOGGING'] = '0'
os.environ['TQDM_DISABLE'] = '1'

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PATHS, MIN_MATCHED_COMMENTS, EMBEDDING_MODEL
from utils.files import read_json_file
from transformers import AutoTokenizer
from vllm import LLM
from vllm.inputs import TokensPrompt


def load_test_1k_data(logger) -> Dict:
    """Load test_1k hydrated dataset."""
    test_1k_file = os.path.join(PATHS['data'], 'test_1k_hydrated.json.zst')

    if not os.path.exists(test_1k_file):
        raise FileNotFoundError(f"test_1k file not found: {test_1k_file}")

    logger.info(f"Loading test_1k dataset from {test_1k_file}...")

    with open(test_1k_file, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            data = json.loads(reader.read())

    logger.info(f"  Loaded {len(data.get('subreddits', []))} subreddits")
    return data


def load_stage2_data(logger) -> Dict[str, Dict]:
    """Load Stage 2 subreddit data and create mapping."""
    stage2_file = os.path.join(PATHS['data'], f'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json')

    if not os.path.exists(stage2_file):
        raise FileNotFoundError(f"Stage 2 file not found: {stage2_file}")

    logger.info(f"Loading Stage 2 data from {stage2_file}...")
    data = read_json_file(stage2_file)

    # Create mapping: subreddit_name -> subreddit_data
    subreddit_map = {}
    for entry in data.get('subreddits', []):
        subreddit_data = entry.get('subreddit', {})
        name = subreddit_data.get('display_name', '').lower()
        if name:
            subreddit_map[name] = subreddit_data

    logger.info(f"  Loaded {len(subreddit_map)} subreddit descriptions")
    return subreddit_map


def prepare_subreddit_texts(test_1k_data: Dict, stage2_map: Dict[str, Dict], logger) -> Tuple[List[str], List[Dict]]:
    """Prepare subreddit texts for embedding and metadata."""
    texts = []
    metadata = []

    logger.info("Preparing subreddit texts...")

    for subreddit_obj in test_1k_data.get('subreddits', []):
        subreddit_name = subreddit_obj.get('subreddit', '')
        language = subreddit_obj.get('language', 'unknown')

        # Get title and description from stage2
        stage2_data = stage2_map.get(subreddit_name, {})
        title = stage2_data.get('title', subreddit_name)
        public_description = stage2_data.get('public_description', '')

        # Format: "Title: {title}\n\nDescription: {public_description}"
        text = f"r/{subreddit_name}: {title}\n\{public_description}"

        texts.append(text)
        metadata.append({
            'subreddit': subreddit_name,
            'language': language,
            'title': title,
            'description': public_description,
            'full_text': text  # Store full embedded text for later use
        })

    logger.info(f"  Prepared {len(texts)} subreddit texts")
    return texts, metadata


def prepare_rule_texts(test_1k_data: Dict, stage2_map: Dict[str, Dict], logger) -> Tuple[List[str], List[Dict]]:
    """Prepare rule texts for embedding and metadata."""
    texts = []
    metadata = []

    logger.info("Preparing rule texts...")

    for subreddit_obj in test_1k_data.get('subreddits', []):
        subreddit_name = subreddit_obj.get('subreddit', '')
        rule_distribution = subreddit_obj.get('rule_distribution', {})
        rules = subreddit_obj.get('rules', [])

        # Get subreddit info from stage2
        stage2_data = stage2_map.get(subreddit_name, {})
        title = stage2_data.get('title', subreddit_name)
        public_description = stage2_data.get('public_description', '')

        # Create mapping of short_name_clean -> rule
        rule_map = {}
        for rule in rules:
            short_name = rule.get('short_name_clean', '')
            if short_name:
                rule_map[short_name] = rule

        # Only include rules with non-zero violation counts
        for rule_name, count in rule_distribution.items():
            if count > 0 and rule_name in rule_map:
                rule = rule_map[rule_name]

                # Get metadata
                short_name_clean = rule.get('short_name_clean', '')
                description_clean = rule.get('description_clean', '')
                violation_reason = rule.get('violation_reason', '')

                # Build rule text without subreddit context (to avoid clustering by subreddit)
                text = f"Rule Name: {short_name_clean}\nRule Description: {description_clean}\nViolation Reason: {violation_reason}"

                if text:  # Only add if we have text to embed
                    texts.append(text)
                    metadata.append({
                        'subreddit': subreddit_name,
                        'short_name': short_name_clean,
                        'description': description_clean,
                        'full_text': text  # Store full embedded text for later use
                    })

    logger.info(f"  Prepared {len(texts)} rule texts (non-zero violations only)")
    return texts, metadata


def pretokenize_texts(texts: List[str], tokenizer, logger) -> Tuple[List[List[int]], int]:
    """Pretokenize texts and find max length (like stage 4).

    Returns:
        Tuple of (tokenized_texts, max_length)
    """
    logger.info(f"Tokenizing {len(texts)} texts...")
    tokenized_texts = []
    lengths = []

    for text in tqdm(texts, desc="Tokenizing"):
        # No instruction formatting - just tokenize the text directly (like rules in stage 4)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokenized_texts.append(tokens)
        lengths.append(len(tokens))

    max_length = max(lengths) if lengths else 0
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Avg length: {avg_length:.1f}")

    return tokenized_texts, max_length


def embed_with_vllm(tokenized_texts: List[List[int]], model_name: str, max_length: int, logger) -> List[List[float]]:
    """Embed pretokenized texts using vLLM (like stage 4).

    Returns:
        List of embeddings (one per text)
    """
    # Calculate optimal max_model_len with buffer
    optimal_max_len = max(max_length + 50, 512)
    logger.info(f"Initializing vLLM with max_model_len={optimal_max_len}...")

    # Initialize vLLM model
    model = LLM(model=model_name, task="embed", gpu_memory_utilization=0.95, enforce_eager=True, max_model_len=optimal_max_len, seed=0)
    logger.info("✅ vLLM model loaded")

    # Create TokensPrompt objects
    logger.info(f"Creating TokensPrompt objects for {len(tokenized_texts)} texts...")
    prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in tokenized_texts]

    # Embed in one batch (vLLM handles batching internally)
    logger.info(f"Embedding {len(prompts)} texts with vLLM...")
    outputs = model.embed(prompts)

    # Extract embeddings (vLLM already normalizes them)
    embeddings = [output.outputs.embedding for output in outputs]
    logger.info(f"  Generated {len(embeddings)} embeddings")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return embeddings


def write_tsv_files(embeddings: List[List[float]], metadata: List[Dict], embedding_file: Path,
                   metadata_file: Path, metadata_columns: List[str], logger):
    """Write embeddings and metadata to TSV files."""
    # Write embeddings
    logger.info(f"Writing embeddings to {embedding_file}...")
    with open(embedding_file, 'w') as f:
        for embedding in embeddings:
            line = '\t'.join(str(x) for x in embedding)
            f.write(line + '\n')

    # Write metadata using pandas to handle escaping properly
    logger.info(f"Writing metadata to {metadata_file}...")
    metadata_df = pd.DataFrame(metadata, columns=metadata_columns)
    metadata_df.to_csv(metadata_file, sep='\t', index=False)

    logger.info(f"  ✅ Wrote {len(embeddings)} rows to {embedding_file}")
    logger.info(f"  ✅ Wrote {len(metadata)} rows to {metadata_file}")


def main():
    """Main execution function."""
    # Create directories
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / 'logs' / 'embeddings'
    output_dir = base_dir / 'output' / 'embeddings'

    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'embed_test_1k_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(str(log_file)), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()

    try:
        # Load data
        test_1k_data = load_test_1k_data(logger)
        stage2_map = load_stage2_data(logger)

        # Prepare texts
        subreddit_texts, subreddit_metadata = prepare_subreddit_texts(test_1k_data, stage2_map, logger)
        rule_texts, rule_metadata = prepare_rule_texts(test_1k_data, stage2_map, logger)

        # Set CUDA device 0 by default
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            logger.info(f"🎯 Using CUDA device 0")
        else:
            logger.info(f"💻 Using CPU mode")

        # Load tokenizer once
        logger.info(f"\nLoading tokenizer for {EMBEDDING_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

        # Pretokenize and embed subreddits
        logger.info("\n" + "="*80)
        logger.info("EMBEDDING SUBREDDITS")
        logger.info("="*80)
        tokenized_subreddits, max_subreddit_len = pretokenize_texts(subreddit_texts, tokenizer, logger)
        subreddit_embeddings = embed_with_vllm(tokenized_subreddits, EMBEDDING_MODEL, max_subreddit_len, logger)

        # Pretokenize and embed rules
        logger.info("\n" + "="*80)
        logger.info("EMBEDDING RULES")
        logger.info("="*80)
        tokenized_rules, max_rule_len = pretokenize_texts(rule_texts, tokenizer, logger)
        rule_embeddings = embed_with_vllm(tokenized_rules, EMBEDDING_MODEL, max_rule_len, logger)

        # Write outputs
        logger.info("\n" + "="*80)
        logger.info("WRITING OUTPUTS")
        logger.info("="*80)

        write_tsv_files(subreddit_embeddings, subreddit_metadata, output_dir / 'test_1k_subreddit_embeddings.tsv',
                       output_dir / 'test_1k_subreddit_metadata.tsv', ['subreddit', 'language', 'title', 'description', 'full_text'], logger)

        write_tsv_files(rule_embeddings, rule_metadata, output_dir / 'test_1k_rule_embeddings.tsv',
                       output_dir / 'test_1k_rule_metadata.tsv', ['subreddit', 'short_name', 'description', 'full_text'], logger)

        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info(f"✅ COMPLETE! Time: {elapsed:.1f}s")
        logger.info("="*80)
        logger.info(f"Subreddit embeddings: {len(subreddit_embeddings)} x {len(subreddit_embeddings[0])}")
        logger.info(f"Rule embeddings: {len(rule_embeddings)} x {len(rule_embeddings[0])}")

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
