#!/usr/bin/env python3
"""
Test script to verify prompt generation works correctly.
"""

import sys
from pathlib import Path

# Add eval directory to path FIRST
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import config
import helpers

def main():
    """Test prompt generation with real data."""

    print("="*80)
    print("TESTING PROMPT GENERATION")
    print("="*80)

    # Create a simple logger
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('test')

    # Load a small sample from test split
    print("\n1. Loading dataset...")
    thread_pairs = helpers.load_dataset('test', logger, debug=True)
    print(f"   Loaded {len(thread_pairs)} thread pairs")

    # Test different context combinations
    contexts_to_test = ['none', 'submission', 'submission-discussion-user', 'subreddit-submission-media-discussion-user']

    for context_type in contexts_to_test:
        print(f"\n{'='*80}")
        print(f"TESTING CONTEXT: {context_type}")
        print(f"{'='*80}")

        # Build prompts
        print("\n2. Building prompts...")
        processed_pairs = helpers.build_prompts_for_thread_pairs(
            thread_pairs[:1],  # Just test first pair
            context_type,
            'baseline',  # No phrase
            'qwen3-vl-8b',
            'prefill',
            logger
        )

        # Print the moderated prompt
        pair = processed_pairs[0]
        messages = pair['moderated_prompt']['messages']
        content = messages[0]['content']

        # Extract text from content
        text_content = None
        for item in content:
            if item['type'] == 'text':
                text_content = item['text']
                break

        print("\n3. Generated Prompt (Moderated Thread):")
        print("-"*80)
        if text_content:
            # Print first 2000 characters
            print(text_content)
            # if len(text_content) > 2000:
            #     print(f"\n... [truncated, total length: {len(text_content)} characters]")
        print("-"*80)

        # Show answer options
        print("\n4. Correct Answer:")
        print(f"   {pair['moderated_correct_answer']}")

    print(f"\n{'='*80}")
    print("✅ PROMPT GENERATION TEST COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
