"""
Helper Functions Module for Reddit Moderation Evaluation

This module contains all utility functions for the evaluation system, including:
- Data loading from clustered datasets
- Prompt building based on context types
- Two-stage model inference (reasoning + answer extraction)
- Answer extraction and validation
- Metrics calculation (accuracy, per-cluster stats)
- Result saving
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime
from PIL import Image

import config

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _format_timestamp(created_utc) -> str:
    """Format Unix timestamp to human-readable date string."""
    # Handle both int and string timestamps
    if isinstance(created_utc, str):
        created_utc = int(created_utc)
    dt = datetime.fromtimestamp(created_utc)
    return dt.strftime("%a, %b %d, %Y, %I:%M%p").replace(" 0", " ")

def _clean_markdown_urls(text: str) -> str:
    """
    Remove markdown-style URLs and images from text.
    Used for submission bodies when media is shown.

    Removes patterns like:
    - [text](url)
    - ![alt](url)
    - Bare URLs

    Keeps the link text but removes the URL.
    """
    import re

    # Remove markdown images: ![alt](url) -> ""
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

    # Remove markdown links: [text](url) -> "text"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove bare URLs (http/https)
    text = re.sub(r'https?://[^\s]+', '', text)

    return text

def _clean_user_mentions(text: str) -> str:
    """
    Remove user mentions from text.
    Used for comment bodies.

    Removes patterns like:
    - u/username
    """
    import re

    # Remove user mentions: u/username -> ""
    text = re.sub(r'u/\w+', '', text)

    return text

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_dataset(split: str, logger: logging.Logger, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Load Reddit moderation dataset from clustered JSON file.

    Args:
        split: Dataset split ('train', 'val', 'test')
        logger: Logger instance
        debug: If True, load only first 5 thread pairs

    Returns:
        List of thread pair dictionaries with all necessary data
    """
    dataset_path = config.get_dataset_path(split)
    logger.info(f"📂 Loading dataset from {dataset_path}")

    # Load JSON (handle compressed and uncompressed)
    if dataset_path.suffix == '.zst':
        import zstandard as zstd
        with open(dataset_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                data = json.loads(reader.read())
    else:
        with open(dataset_path, 'r') as f:
            data = json.load(f)

    # Extract thread pairs
    thread_pairs = []
    for subreddit_data in data['subreddits']:
        subreddit = subreddit_data['subreddit']
        subreddit_cluster_id = subreddit_data.get('subreddit_cluster_id', -1)
        subreddit_cluster_label = subreddit_data.get('subreddit_cluster_label', 'Other')
        subreddit_title = subreddit_data.get('title', '')
        subreddit_description = subreddit_data.get('description', '')

        for pair in subreddit_data['thread_pairs']:
            # Get submission data
            submission_id = pair['metadata']['submission_id']
            submission_data = subreddit_data['submissions'].get(submission_id, {})

            thread_pair = {
                'subreddit': subreddit,
                'subreddit_title': subreddit_title,
                'subreddit_description': subreddit_description,
                'subreddit_cluster_id': subreddit_cluster_id,
                'subreddit_cluster_label': subreddit_cluster_label,
                'mod_comment_id': pair['mod_comment_id'],
                'submission_id': submission_id,
                'submission': submission_data,
                'rules': subreddit_data['rules'],
                'moderated_thread': pair['moderated_thread'],
                'unmoderated_thread': pair['unmoderated_thread'],
                'moderated_answer_options': pair['moderated_answer_options'],
                'moderated_correct_answer': pair['moderated_correct_answer'],
                'unmoderated_answer_options': pair['unmoderated_answer_options'],
                'unmoderated_correct_answer': pair['unmoderated_correct_answer'],
                'metadata': pair['metadata']
            }
            thread_pairs.append(thread_pair)

    # Debug mode: limit to first 5 pairs
    if debug:
        thread_pairs = thread_pairs[:5]
        logger.info(f"🐛 Debug mode: Using only {len(thread_pairs)} thread pairs")
    else:
        logger.info(f"✅ Loaded {len(thread_pairs)} thread pairs from {split} split")

    return thread_pairs

# =============================================================================
# PROMPT BUILDING FUNCTIONS
# =============================================================================

def build_prompts_for_thread_pairs(thread_pairs: List[Dict[str, Any]],
                                   context_type: str,
                                   phrase_name: str,
                                   model_name: str,
                                   mode: str,
                                   logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Build prompts for all thread pairs (both moderated and unmoderated).

    Args:
        thread_pairs: List of thread pair dictionaries
        context_type: Context type (e.g., 'thread_with_rule')
        phrase_name: Phrase name (e.g., 'cot')
        model_name: Model name
        mode: Phrase mode ('prefill' or 'prompt')
        logger: Logger instance

    Returns:
        List of thread pairs with prompts added
    """
    context_config = config.parse_context_flags(context_type)
    phrase_text = config.PHRASES.get(phrase_name, '')
    model_config = config.get_model_config(model_name)

    processed_pairs = []
    for pair in thread_pairs:
        # Build prompts for both moderated and unmoderated threads
        moderated_prompt = _build_single_prompt(
            pair,
            thread_type='moderated',
            context_config=context_config,
            phrase_text=phrase_text,
            model_config=model_config,
            mode=mode
        )

        unmoderated_prompt = _build_single_prompt(
            pair,
            thread_type='unmoderated',
            context_config=context_config,
            phrase_text=phrase_text,
            model_config=model_config,
            mode=mode
        )

        processed_pair = {
            **pair,
            'moderated_prompt': moderated_prompt,
            'unmoderated_prompt': unmoderated_prompt,
            'phrase_text': phrase_text if mode == 'prefill' else None,
            'mode': mode
        }
        processed_pairs.append(processed_pair)

    logger.info(f"🔨 Built prompts for {len(processed_pairs)} thread pairs ({len(processed_pairs) * 2} total threads)")
    return processed_pairs

def _build_single_prompt(pair: Dict[str, Any],
                        thread_type: str,
                        context_config: Dict[str, Any],
                        phrase_text: str,
                        model_config: Dict[str, Any],
                        mode: str) -> Dict[str, Any]:
    """
    Build prompt for a single thread (moderated or unmoderated).

    Args:
        pair: Thread pair dictionary
        thread_type: 'moderated' or 'unmoderated'
        context_config: Context configuration
        phrase_text: Phrase text to apply
        model_config: Model configuration
        mode: Phrase mode ('prefill' or 'prompt')

    Returns:
        Dictionary with 'messages' key
    """
    messages = []

    # Build question text based on context (pass phrase if mode is 'prompt')
    # Convert "Let's X" to "Please X." for prompt mode
    if mode == 'prompt' and phrase_text:
        prompt_phrase = phrase_text.replace("Let's", "Please")
        if not prompt_phrase.endswith('.'):
            prompt_phrase += '.'
    else:
        prompt_phrase = None
    question_text = _build_question_text(pair, thread_type, context_config, prompt_phrase)

    # Build user content with multimodal data
    content = _build_multimodal_content(
        pair,
        question_text,
        context_config,
        model_config
    )

    messages.append({"role": "user", "content": content})

    return {'messages': messages}

def _build_question_text(pair: Dict[str, Any],
                        thread_type: str,
                        context_config: Dict[str, Any],
                        prompt_phrase: str = None) -> str:
    """
    Build question text based on context configuration flags.

    Args:
        pair: Thread pair dictionary
        thread_type: 'moderated' or 'unmoderated'
        context_config: Context configuration with boolean flags
        prompt_phrase: Optional phrase to integrate into the question (for 'prompt' mode)

    Returns:
        Formatted question text
    """
    # Get the appropriate thread and answer options
    thread = pair[f'{thread_type}_thread']
    answer_options = pair[f'{thread_type}_answer_options']
    submission = pair.get('submission', {}).get('submission_object', {})

    # Build context parts in order: subreddit, rules, submission, discussion, question
    context_parts = []

    # 1. Subreddit (only show if flag is set)
    if context_config.get('include_subreddit', False):
        subreddit_name = pair['subreddit']
        subreddit_parts = [f"Subreddit: r/{subreddit_name}"]

        # Add title and description
        subreddit_title = pair.get('subreddit_title', '')
        if subreddit_title:
            subreddit_parts.append(f"Title: {subreddit_title}")

        subreddit_description = pair.get('subreddit_description', '')
        if subreddit_description:
            subreddit_parts.append(f"Description: {subreddit_description}")

        context_parts.append("\n".join(subreddit_parts))

    # 2. Rules (always shown)
    rules_text = _format_rules(pair['rules'])
    context_parts.append(f"\nRules:\n{rules_text}")

    # 3. Submission (if submission flag is set)
    if context_config.get('include_submission', False):
        # Determine if we should show actual media or placeholder
        show_media = context_config.get('include_media', False)
        show_users = context_config.get('include_user', False)
        submission_text = _format_submission(submission, pair, show_media, show_users)
        context_parts.append(f"\n{submission_text}")

    # 4. Discussion thread (if discussion flag is set)
    # Otherwise, just show the leaf comment
    show_full_discussion = context_config.get('include_discussion', False)
    show_users = context_config.get('include_user', False)

    # Always build user anonymization map (users are always anonymized when shown)
    user_map = _build_user_map(submission, thread)

    if show_full_discussion:
        # Show all comments with the last one marked
        discussion_text = _format_discussion(thread, user_map, show_users, mark_last=True)
        context_parts.append(f"\n{discussion_text}")
    else:
        # Show only the leaf comment (last comment in thread)
        leaf_comment_text = _format_leaf_comment(thread[-1], user_map, show_users)
        context_parts.append(f"\nComment 1 [TARGET COMMENT]:\n{leaf_comment_text}")

    # 5. Question and answer options
    options_str = _format_answer_options(answer_options)

    # Build question with optional phrase integration
    if prompt_phrase:
        question_text = f"\nQuestion: Does the [TARGET COMMENT] violate a rule? {prompt_phrase}\n\n{options_str}"
    else:
        question_text = f"\nQuestion: Does the [TARGET COMMENT] violate a rule?\n\n{options_str}"

    context_parts.append(question_text)

    return "\n".join(context_parts)

def _format_rules(rules: List[Dict[str, Any]]) -> str:
    """Format rules list."""
    formatted = []
    for idx, rule in enumerate(rules, start=1):
        short_name = rule['short_name_clean']
        description = rule['description_clean']
        violation_reason = rule['violation_reason_clean']

        rule_text = (
            f"{idx}. {short_name}\n"
            f"Description: {description}\n"
            f"Violation: {violation_reason}"
        )
        formatted.append(rule_text)

    return "\n\n".join(formatted)

def _format_submission(submission: Dict[str, Any],
                       pair: Dict[str, Any],
                       show_media: bool,
                       show_user: bool = False) -> str:
    """Format submission content."""
    title = submission.get('title', '[No title]')
    selftext = submission.get('selftext', '').strip()
    link_flair = submission.get('link_flair_text')
    created_utc = submission.get('created_utc', 0)

    # Format datetime
    date_str = _format_timestamp(created_utc)

    # Build submission header
    parts = [f"Submission Title: {title}"]
    if link_flair:
        parts.append(f"Flair: {link_flair}")
    if show_user:
        parts.append(f"Author: USER1")  # Submission author is always USER1
    parts.append(f"Posted: {date_str}")

    header = "\n".join(parts)

    # Check if media exists
    media_files = pair.get('submission', {}).get('media_files', [])
    has_media = len(media_files) > 0

    # Build body content
    body_parts = []

    # Add media placeholder if media exists but not included
    if has_media and not show_media:
        body_parts.append("[Image present but not shown]")

    # Add text content (always clean user mentions; clean URLs when media is shown)
    if selftext:
        cleaned_text = selftext
        if show_media:
            # Clean markdown URLs when showing actual images
            cleaned_text = _clean_markdown_urls(cleaned_text)
        # Always clean user mentions
        cleaned_text = _clean_user_mentions(cleaned_text)
        body_parts.append(cleaned_text)
    elif not has_media:  # Only show [No text] if there's also no media
        body_parts.append("[No text]")

    body_text = "\n".join(body_parts)

    # Add Body: prefix (first occurrence allows splitting for media interleaving)
    return f"{header}\nBody: {body_text}"

def _format_leaf_comment(comment: Dict[str, Any],
                        user_map: Dict[str, str],
                        show_user: bool = False) -> str:
    """Format a single leaf comment (without section header)."""
    author = comment.get('author', '[deleted]')
    author_flair = comment.get('author_flair_text')
    body = comment.get('body', '[deleted]')
    created_utc = comment.get('created_utc', 0)

    # Format datetime
    date_str = _format_timestamp(created_utc)

    # Get anonymized username
    user_label = user_map.get(author, author)

    # Clean user mentions from body
    cleaned_body = _clean_user_mentions(body)

    # Build comment (no header - caller adds section header)
    parts = []
    if author_flair:
        parts.append(f"Flair: {author_flair}")
    if show_user:
        parts.append(f"Author: {user_label}")
    parts.append(f"Posted: {date_str}")
    parts.append(f"Body: {cleaned_body}")

    return "\n".join(parts)

def _format_discussion(thread: List[Dict[str, Any]],
                       user_map: Dict[str, str],
                       show_user: bool = False,
                       mark_last: bool = True) -> str:
    """Format discussion thread with comments."""
    formatted = []

    for idx, comment in enumerate(thread, start=1):
        author = comment.get('author', '[deleted]')
        author_flair = comment.get('author_flair_text')
        body = comment.get('body', '[deleted]')
        created_utc = comment.get('created_utc', 0)

        # Format datetime
        date_str = _format_timestamp(created_utc)

        # Get anonymized username
        user_label = user_map.get(author, author)

        # Clean user mentions from body
        cleaned_body = _clean_user_mentions(body)

        # Build comment header
        is_last = (idx == len(thread))
        comment_label = f"Comment {idx}" + (" [TARGET COMMENT]" if mark_last and is_last else "") + ":"

        parts = [comment_label]
        if author_flair:
            parts.append(f"Flair: {author_flair}")
        if show_user:
            parts.append(f"Author: {user_label}")
        parts.append(f"Posted: {date_str}")
        parts.append(f"Body: {cleaned_body}")

        formatted.append("\n".join(parts))

    return "\n\n".join(formatted)

def _build_user_map(submission: Dict[str, Any],
                    thread: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from real usernames to anonymized labels."""
    # Collect all unique authors
    submission_author = submission.get('author', '[deleted]')

    # Start with submission author as USER1
    user_map = {submission_author: 'USER1'}

    # Add other users in order of appearance
    user_counter = 2
    for comment in thread:
        author = comment.get('author', '[deleted]')
        if author not in user_map:
            user_map[author] = f'USER{user_counter}'
            user_counter += 1

    return user_map

def _format_answer_options(options: List[Dict[str, str]]) -> str:
    """
    Format answer options as multiple choice.

    Args:
        options: List of option dictionaries with 'label' and 'rule' keys

    Returns:
        Formatted options string
    """
    formatted = []
    for option in options:
        label = option['label']
        rule = option['rule']
        formatted.append(f"{label} {rule}")

    return "\n".join(formatted)

def _build_multimodal_content(pair: Dict[str, Any],
                              question_text: str,
                              context_config: Dict[str, Any],
                              model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build multimodal content with images and text.

    Images are interleaved in the submission by splitting on first "Body: " marker.
    This splits submission header from body, allowing images to appear between them.
    Comment bodies (which also have "Body: ") remain intact in the text after the split.

    Args:
        pair: Thread pair dictionary
        question_text: Question text (may contain "Body: " marker in submission)
        context_config: Context configuration
        model_config: Model configuration

    Returns:
        List of content items (text and images properly interleaved)
    """
    content = []

    # Check if we need to add media and interleave it
    if context_config.get('include_media', False) and context_config.get('include_submission', False):
        submission = pair.get('submission', {})
        media_files = submission.get('media_files', [])

        if media_files and "Body: " in question_text:
            # Split on first "Body: " only (submission body)
            # Everything after (including comment bodies) stays together
            parts = question_text.split("Body: ", 1)
            before_body = parts[0] + "Body: "  # Keep the "Body: " prefix
            after_body = parts[1] if len(parts) > 1 else ""

            # Add text before images (submission header)
            content.append({"type": "text", "text": before_body})

            # Add images
            for media_path in media_files:
                if model_config['type'] == 'vllm':
                    if model_config['hf_path'].startswith('Qwen'):
                        content.append({"type": "image", "image": media_path})
                    else:  # LLaVA, Llama
                        content.append({"type": "image"})
                else:  # API models
                    content.append({"type": "image"})

            # Add text after images (submission body + rest of prompt)
            content.append({"type": "text", "text": after_body})
        else:
            # No media or no Body: marker, just add text
            content.append({"type": "text", "text": question_text})
    else:
        # No media to interleave, just add text
        content.append({"type": "text", "text": question_text})

    return content

# =============================================================================
# CHAT TEMPLATE APPLICATION
# =============================================================================

def apply_chat_template(thread_pairs: List[Dict[str, Any]],
                       model_name: str,
                       logger: logging.Logger) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply model-specific chat template to all prompts and calculate resource requirements.

    Args:
        thread_pairs: List of thread pairs with prompts
        model_name: Model name
        logger: Logger instance

    Returns:
        Tuple of (thread_pairs with formatted prompts, resource_stats dict)
        resource_stats contains:
            - max_images_per_prompt: Maximum number of images in any single prompt
            - max_model_len: Maximum token length across all prompts (including image tokens)
    """
    model_config = config.get_model_config(model_name)

    # API models don't use AutoProcessor
    if model_config['type'] == 'api':
        logger.info("⏭️  Skipping chat template for API model (will be handled in API call)")
        return thread_pairs, {'max_images_per_prompt': 0, 'max_model_len': 0}

    # vLLM models use AutoProcessor
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_config['hf_path'],
        trust_remote_code=model_config.get('trust_remote_code', False)
    )

    max_images_per_prompt = 0
    max_token_length = 0

    for pair in thread_pairs:
        # Count images in this pair
        submission = pair.get('submission', {})
        media_files = submission.get('media_files', [])
        num_images = len(media_files)
        max_images_per_prompt = max(max_images_per_prompt, num_images)

        # Apply template to both moderated and unmoderated threads
        for thread_type in ['moderated', 'unmoderated']:
            messages = pair[f'{thread_type}_prompt']['messages']

            # Generate prompt text (for vLLM)
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Add phrase text for prefill mode
            if pair.get('phrase_text') and pair['mode'] == 'prefill':
                prompt_text += pair['phrase_text']

            pair[f'{thread_type}_prompt_text'] = prompt_text

            # Tokenize with images to get accurate token count
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors='pt'
                )
                token_length = inputs['input_ids'].shape[1]
                max_token_length = max(max_token_length, token_length)
            except Exception as e:
                logger.warning(f"⚠️  Could not tokenize prompt for length calculation: {e}")

    resource_stats = {
        'max_images_per_prompt': max_images_per_prompt,
        'max_model_len': max_token_length
    }

    logger.info(f"✅ Applied chat template to {len(thread_pairs)} thread pairs")
    logger.info(f"📊 Resource requirements - Max images: {max_images_per_prompt}, Max tokens: {max_token_length}")

    return thread_pairs, resource_stats

# =============================================================================
# TWO-STAGE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_two_stage_vllm(thread_pairs: List[Dict[str, Any]],
                             model_name: str,
                             model_config: Dict[str, Any],
                             num_gpus: int,
                             resource_stats: Dict[str, Any],
                             max_response_tokens: int,
                             logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Two-stage evaluation using vLLM.

    Args:
        thread_pairs: Thread pairs with prompts
        model_name: Model name
        model_config: Model configuration
        num_gpus: Number of GPUs to use for tensor parallelism
        resource_stats: Resource statistics from chat template application
        max_response_tokens: Maximum tokens for response generation
        logger: Logger instance

    Returns:
        Evaluation results
    """
    from vllm import LLM

    # Initialize LLM engine
    logger.info(f"🚀 Initializing vLLM engine for {model_name}...")
    logger.info(f"📊 Using {num_gpus} GPU(s) for tensor parallelism")

    # For Qwen3-VL models, set multimodal limits based on actual data
    limit_mm_per_prompt = None
    if 'Qwen3-VL' in model_config['hf_path'] or 'Qwen/Qwen3-VL' in model_config['hf_path']:
        max_images = resource_stats.get('max_images_per_prompt', 50)
        limit_mm_per_prompt = {'image': max_images, 'video': 0}
        logger.info(f"📊 Setting limit_mm_per_prompt for Qwen3-VL: {limit_mm_per_prompt}")

    # Use actual max token length if available, otherwise fall back to config
    max_model_len = resource_stats.get('max_model_len', 0)
    if max_model_len == 0:
        max_model_len = model_config.get('max_model_len', 8192)
        logger.info(f"📊 Using configured max_model_len: {max_model_len}")
    else:
        # Add buffer for generation (response tokens)
        max_model_len = max_model_len + max_response_tokens
        logger.info(f"📊 Using calculated max_model_len: {max_model_len} (including {max_response_tokens} token buffer)")

    llm_engine = LLM(
        model=model_config['hf_path'],
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=model_config.get('gpu_memory_utilization', 0.9),
        trust_remote_code=model_config.get('trust_remote_code', True),
        max_model_len=max_model_len,
        limit_mm_per_prompt=limit_mm_per_prompt,
        seed=0
    )
    logger.info(f"✅ LLM engine initialized with tensor_parallel_size={num_gpus}")

    # Stage 1: Generate reasoning for all threads (moderated + unmoderated)
    logger.info("📝 Stage 1: Generating reasoning responses...")
    stage1_responses = _generate_stage1_vllm(thread_pairs, llm_engine, max_response_tokens, logger)
    logger.info(f"✅ Generated {len(stage1_responses)} × 2 Stage 1 reasoning responses")

    # Stage 2: Extract clean answers
    logger.info("🎯 Stage 2: Extracting clean answers...")
    results = _generate_stage2_vllm(thread_pairs, stage1_responses, llm_engine, max_response_tokens, logger)

    logger.info(f"✅ Completed two-stage evaluation for {len(results)} thread pairs")
    return results

def _generate_stage1_vllm(thread_pairs: List[Dict[str, Any]],
                         llm_engine,
                         max_response_tokens: int,
                         logger: logging.Logger,
                         moderated_prompts: List[str] = None,
                         unmoderated_prompts: List[str] = None,
                         sampling_params = None,
                         uuid_suffix: str = '') -> List[Dict[str, str]]:
    """
    Generate responses for all threads using vLLM (base function for both Stage 1 and Stage 2).

    Args:
        thread_pairs: Thread pairs with prompts
        llm_engine: vLLM engine
        max_response_tokens: Maximum tokens for response generation
        logger: Logger instance
        moderated_prompts: Optional custom prompts for moderated threads (defaults to pair['moderated_prompt_text'])
        unmoderated_prompts: Optional custom prompts for unmoderated threads (defaults to pair['unmoderated_prompt_text'])
        sampling_params: Optional custom sampling params (defaults to temperature=0, max_tokens from max_response_tokens)
        uuid_suffix: Optional suffix for multimodal UUIDs (e.g., '_s2' for stage 2)

    Returns:
        List of dicts with 'moderated' and 'unmoderated' responses
    """
    from vllm import SamplingParams

    # Use default prompts if not provided
    if moderated_prompts is None:
        moderated_prompts = [pair['moderated_prompt_text'] for pair in thread_pairs]
    if unmoderated_prompts is None:
        unmoderated_prompts = [pair['unmoderated_prompt_text'] for pair in thread_pairs]

    # Use default sampling params if not provided
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_response_tokens, stop=None)

    # Prepare inputs (flatten moderated + unmoderated)
    inputs = []
    for pair in thread_pairs:
        # Get submission media
        submission = pair.get('submission', {})
        media_files = submission.get('media_files', [])

        # Build inputs for both moderated and unmoderated threads
        for thread_type, prompts in [('mod', moderated_prompts), ('unmod', unmoderated_prompts)]:
            idx = thread_pairs.index(pair)

            # Load images separately for each thread to avoid sharing image objects
            images = []
            for media_path in media_files:
                try:
                    img = Image.open(media_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load image {media_path}: {e}")

            input_dict = {
                "prompt": prompts[idx],
                "multi_modal_data": {"image": images} if images else {},
                "multi_modal_uuids": {"image": [f"uuid_{thread_type}{uuid_suffix}_{pair['mod_comment_id']}_{j}"
                                                for j in range(len(images))]} if images else {}
            }
            inputs.append(input_dict)

    # Generate responses
    outputs = llm_engine.generate(inputs, sampling_params=sampling_params)

    # Unflatten responses back to pairs
    responses = []
    for i in range(0, len(outputs), 2):
        mod_response = outputs[i].outputs[0].text
        unmod_response = outputs[i + 1].outputs[0].text

        responses.append({
            'moderated': mod_response,
            'unmoderated': unmod_response
        })

    return responses

def _generate_stage2_vllm(thread_pairs: List[Dict[str, Any]],
                         stage1_responses: List[Dict[str, str]],
                         llm_engine,
                         max_response_tokens: int,
                         logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Generate Stage 2 clean answer extraction by calling Stage 1 internally.

    Args:
        thread_pairs: Thread pairs
        stage1_responses: Stage 1 reasoning responses
        llm_engine: vLLM engine
        max_response_tokens: Maximum tokens for response generation (not used in Stage 2, uses fixed 10 tokens)
        logger: Logger instance

    Returns:
        Complete evaluation results
    """
    from vllm import SamplingParams

    # Prepare Stage 2 prompts (append reasoning + answer phrase)
    moderated_prompts = [
        pair['moderated_prompt_text'] + stage1['moderated'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_responses)
    ]
    unmoderated_prompts = [
        pair['unmoderated_prompt_text'] + stage1['unmoderated'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_responses)
    ]

    # Generate clean answers using Stage 1 function (temperature=0, stop at newline to get single choice)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10
    )

    stage2_responses = _generate_stage1_vllm(
        thread_pairs=thread_pairs,
        llm_engine=llm_engine,
        max_response_tokens=max_response_tokens,
        logger=logger,
        moderated_prompts=moderated_prompts,
        unmoderated_prompts=unmoderated_prompts,
        sampling_params=sampling_params,
        uuid_suffix='_s2'
    )

    # Build final results
    results = []
    for pair, stage1, stage2 in zip(thread_pairs, stage1_responses, stage2_responses):
        mod_clean = stage2['moderated']
        unmod_clean = stage2['unmoderated']

        # Extract predictions
        mod_prediction = _extract_answer_choice(mod_clean)
        unmod_prediction = _extract_answer_choice(unmod_clean)

        # Get ground truth
        mod_correct = pair['moderated_correct_answer']
        unmod_correct = pair['unmoderated_correct_answer']

        # Calculate scores
        mod_score = 1 if mod_prediction == mod_correct else 0
        unmod_score = 1 if unmod_prediction == unmod_correct else 0

        result = {
            'mod_comment_id': pair['mod_comment_id'],
            'subreddit': pair['subreddit'],
            'submission_id': pair['submission_id'],

            'moderated': {
                'input_prompt': pair['moderated_prompt_text'],
                'reasoning_response': stage1['moderated'],
                'clean_answer_response': mod_clean,
                'extracted_prediction': mod_prediction,
                'correct_answer': mod_correct,
                'score': mod_score,
                'answer_options': pair['moderated_answer_options']
            },

            'unmoderated': {
                'input_prompt': pair['unmoderated_prompt_text'],
                'reasoning_response': stage1['unmoderated'],
                'clean_answer_response': unmod_clean,
                'extracted_prediction': unmod_prediction,
                'correct_answer': unmod_correct,
                'score': unmod_score,
                'answer_options': pair['unmoderated_answer_options']
            },

            'metadata': {
                'rule': pair['metadata']['rule'],
                'rule_cluster_id': pair['metadata']['rule_cluster_id'],
                'rule_cluster_label': pair['metadata']['rule_cluster_label'],
                'subreddit_cluster_id': pair['subreddit_cluster_id'],
                'subreddit_cluster_label': pair['subreddit_cluster_label']
            }
        }
        results.append(result)

    logger.info(f"Generated {len(results)} × 2 Stage 2 clean answers")
    return results

def evaluate_two_stage_api(_thread_pairs: List[Dict[str, Any]],
                            _model_config: Dict[str, Any],
                            logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Two-stage evaluation using Batch API (Claude, GPT-4V, etc.).

    PLACEHOLDER: To be implemented when adding API model support.

    Args:
        _thread_pairs: Thread pairs with prompts (unused, for future implementation)
        _model_config: Model configuration (unused, for future implementation)
        logger: Logger instance

    Returns:
        Evaluation results
    """
    logger.warning("⚠️  API model evaluation not yet implemented")
    raise NotImplementedError("API model evaluation will be added in future update")

# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def _extract_answer_choice(text: str) -> str:
    """
    Extract answer choice (a-h) from model response.

    Args:
        text: Model response text

    Returns:
        Extracted choice like "(a)", "(b)", etc., or empty string if not found
    """
    import re

    text = text.strip().lower()

    # Look for patterns like "(a)", "(b)", "a)", "a.", "option a", etc.
    # Match single letter a-z
    match = re.search(r'\(?([a-z])\)?', text)
    if match:
        letter = match.group(1)
        return f"({letter})"

    return ""

# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(results: List[Dict[str, Any]], logger: logging.Logger) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.

    Args:
        results: Evaluation results
        logger: Logger instance

    Returns:
        Metrics dictionary with overall and per-cluster accuracy
    """
    total_pairs = len(results)

    # Overall accuracy
    mod_correct = sum(r['moderated']['score'] for r in results)
    unmod_correct = sum(r['unmoderated']['score'] for r in results)
    total_correct = mod_correct + unmod_correct
    total_threads = total_pairs * 2

    overall_accuracy = total_correct / total_threads if total_threads > 0 else 0
    mod_accuracy = mod_correct / total_pairs if total_pairs > 0 else 0
    unmod_accuracy = unmod_correct / total_pairs if total_pairs > 0 else 0

    # Per-rule-cluster accuracy
    rule_cluster_stats = _calculate_cluster_accuracy(
        results,
        cluster_key='rule_cluster_label'
    )

    # Per-subreddit-cluster accuracy
    subreddit_cluster_stats = _calculate_cluster_accuracy(
        results,
        cluster_key='subreddit_cluster_label'
    )

    metrics = {
        'overall': {
            'total_pairs': total_pairs,
            'total_threads': total_threads,
            'overall_accuracy': overall_accuracy,
            'moderated_accuracy': mod_accuracy,
            'unmoderated_accuracy': unmod_accuracy,
            'moderated_correct': mod_correct,
            'unmoderated_correct': unmod_correct,
            'total_correct': total_correct
        },
        'per_rule_cluster': rule_cluster_stats,
        'per_subreddit_cluster': subreddit_cluster_stats
    }

    logger.info(f"📊 Metrics calculated - Overall: {overall_accuracy:.4f}, Mod: {mod_accuracy:.4f}, Unmod: {unmod_accuracy:.4f}")
    return metrics

def _calculate_cluster_accuracy(results: List[Dict[str, Any]],
                                cluster_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Calculate per-cluster accuracy statistics.

    Args:
        results: Evaluation results
        cluster_key: Key to group by ('rule_cluster_label' or 'subreddit_cluster_label')

    Returns:
        Dictionary mapping cluster labels to accuracy stats
    """
    cluster_stats = defaultdict(lambda: {
        'mod_correct': 0,
        'unmod_correct': 0,
        'total_correct': 0,
        'count': 0
    })

    for result in results:
        cluster_label = result['metadata'][cluster_key]

        cluster_stats[cluster_label]['mod_correct'] += result['moderated']['score']
        cluster_stats[cluster_label]['unmod_correct'] += result['unmoderated']['score']
        cluster_stats[cluster_label]['total_correct'] += result['moderated']['score'] + result['unmoderated']['score']
        cluster_stats[cluster_label]['count'] += 1

    # Calculate accuracies
    final_stats = {}
    for cluster, stats in cluster_stats.items():
        count = stats['count']
        total_threads = count * 2

        final_stats[cluster] = {
            'overall_accuracy': stats['total_correct'] / total_threads if total_threads > 0 else 0,
            'moderated_accuracy': stats['mod_correct'] / count if count > 0 else 0,
            'unmoderated_accuracy': stats['unmod_correct'] / count if count > 0 else 0,
            'count': count,
            'total_threads': total_threads
        }

    return final_stats

# =============================================================================
# RESULT SAVING
# =============================================================================

def save_results(results: List[Dict[str, Any]],
                metrics: Dict[str, Any],
                output_dir: Path,
                model_name: str,
                split: str,
                context: str,
                phrase: str,
                mode: str,
                logger: logging.Logger) -> Tuple[Path, Path]:
    """
    Save evaluation results and metrics.

    Args:
        results: Evaluation results
        metrics: Metrics dictionary
        output_dir: Output directory
        model_name: Model name
        split: Dataset split
        context: Context type
        phrase: Phrase name
        mode: Phrase mode
        logger: Logger instance

    Returns:
        Tuple of (reasoning_path, performance_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save reasoning traces
    reasoning_path = output_dir / f"reasoning_{timestamp}.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save performance metrics
    performance_data = {
        'model': model_name,
        'split': split,
        'context': context,
        'phrase': phrase,
        'mode': mode,
        'metrics': metrics
    }

    performance_path = output_dir / f"performance_{timestamp}.json"
    with open(performance_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, indent=2)

    logger.info(f"💾 Reasoning traces saved to: {reasoning_path}")
    logger.info(f"💾 Performance metrics saved to: {performance_path}")

    return reasoning_path, performance_path

# =============================================================================
# LOGGING
# =============================================================================

def create_logger(split: str, model: str, context: str, phrase: str, mode: str) -> Tuple[logging.Logger, Path]:
    """
    Create logger with file and console handlers.

    Args:
        split: Dataset split
        model: Model name
        context: Context type
        phrase: Phrase name
        mode: Phrase mode

    Returns:
        Tuple of (logger, log_file_path)
    """
    import sys

    logs_dir = config.get_dir(config.LOGS_DIR, split, model, context, phrase, mode)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"evaluation_{timestamp}.log"

    logger = logging.getLogger('reddit_mod_eval')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"📝 Logging to: {log_path}")
    return logger, log_path
