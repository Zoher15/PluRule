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
import base64
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
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
# OPENAI BATCH API - IMAGE ENCODING
# =============================================================================

def _encode_image_base64(image_path: str) -> str:
    """
    Encode a local image file to base64 data URL for OpenAI API.

    Args:
        image_path: Path to local image file

    Returns:
        Base64 data URL string (e.g., "data:image/jpeg;base64,...")
    """
    path = Path(image_path)

    # Determine MIME type from extension
    extension = path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(extension, 'image/jpeg')

    # Read and encode
    with open(path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:{mime_type};base64,{image_data}"


def _build_openai_messages(pair: Dict[str, Any],
                           thread_type: str,
                           context_config: Dict[str, Any],
                           logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible messages with base64 images.

    Args:
        pair: Thread pair dictionary with prompts
        thread_type: 'moderated' or 'unmoderated'
        context_config: Context configuration
        logger: Logger instance

    Returns:
        List of message dicts for OpenAI API
    """
    prompt_data = pair[f'{thread_type}_prompt']
    messages_in = prompt_data['messages']

    # Get media files from pair (for API models, image paths aren't in content dict)
    media_files = pair.get('submission', {}).get('media_files', [])
    image_idx = 0  # Track which image we're on

    messages_out = []
    for msg in messages_in:
        role = msg['role']
        content_in = msg['content']

        if isinstance(content_in, str):
            messages_out.append({"role": role, "content": content_in})
        elif isinstance(content_in, list):
            content_out = []
            for item in content_in:
                if item.get('type') == 'text':
                    content_out.append({"type": "text", "text": item['text']})
                elif item.get('type') == 'image':
                    # Only encode images if media flag is set
                    if context_config.get('include_media', False):
                        # Try to get path from item (Qwen format) or from media_files list
                        image_path = item.get('image')
                        if not image_path and image_idx < len(media_files):
                            image_path = media_files[image_idx]
                            image_idx += 1

                        if image_path:
                            try:
                                base64_url = _encode_image_base64(image_path)
                                content_out.append({
                                    "type": "image_url",
                                    "image_url": {"url": base64_url}
                                })
                            except Exception as e:
                                logger.warning(f"Failed to encode image {image_path}: {e}")
            messages_out.append({"role": role, "content": content_out})

    return messages_out


def _create_batch_jsonl_files(thread_pairs: List[Dict[str, Any]],
                              model_id: str,
                              max_tokens: int,
                              context: str,
                              output_path: Path,
                              logger: logging.Logger) -> List[Path]:
    """
    Create JSONL file(s) for OpenAI Batch API, splitting if needed for size limits.

    OpenAI limits: 50,000 requests per batch, 200 MB max file size.
    This function splits into multiple batch files when approaching limits.

    Args:
        thread_pairs: Thread pairs with prompts
        model_id: OpenAI model ID (e.g., 'gpt-5.2-2025-12-11')
        max_tokens: Maximum response tokens
        context: Context string for parsing flags
        output_path: Directory to save JSONL
        logger: Logger instance

    Returns:
        List of paths to created JSONL files
    """
    context_config = config.parse_context_flags(context)
    max_requests = config.OPENAI_BATCH_CONFIG['max_requests_per_batch']
    max_file_size = config.OPENAI_BATCH_CONFIG['max_file_size_bytes']

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = []
    current_batch_idx = 0
    current_file = None
    current_size = 0
    current_requests = 0

    def start_new_batch():
        nonlocal current_batch_idx, current_file, current_size, current_requests
        if current_file:
            current_file.close()
        jsonl_path = output_path / f'batch_requests_{current_batch_idx}.jsonl'
        jsonl_files.append(jsonl_path)
        current_file = open(jsonl_path, 'w')
        current_size = 0
        current_requests = 0
        current_batch_idx += 1
        logger.info(f"Starting batch file {current_batch_idx}: {jsonl_path}")

    # Start first batch
    start_new_batch()

    for pair in thread_pairs:
        for thread_type in ['moderated', 'unmoderated']:
            # Build custom_id: mod_comment_id + thread_type
            custom_id = f"{pair['mod_comment_id']}_{thread_type}"

            # Build OpenAI messages
            messages = _build_openai_messages(pair, thread_type, context_config, logger)

            # Create batch request entry
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "temperature": 0.0
                }
            }

            request_line = json.dumps(request) + '\n'
            request_size = len(request_line.encode('utf-8'))

            # Check if we need to start a new batch
            if current_requests >= max_requests or (current_size + request_size) > max_file_size:
                logger.info(f"Batch {current_batch_idx} reached limits: {current_requests} requests, {current_size / (1024*1024):.1f} MB")
                start_new_batch()

            current_file.write(request_line)
            current_size += request_size
            current_requests += 1

    # Close the last file
    if current_file:
        current_file.close()

    total_requests = len(thread_pairs) * 2
    logger.info(f"Created {len(jsonl_files)} batch file(s) with {total_requests} total requests")
    for i, path in enumerate(jsonl_files):
        file_size = path.stat().st_size / (1024 * 1024)
        logger.info(f"  Batch {i+1}: {path.name} ({file_size:.1f} MB)")

    return jsonl_files


# =============================================================================
# OPENAI BATCH API - STATE MANAGEMENT
# =============================================================================

@dataclass
class BatchState:
    """State for a single batch job."""
    batch_id: str
    input_file_id: str
    status: str  # 'created', 'validating', 'in_progress', 'completed', 'failed', 'expired', 'cancelled'
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    created_at: str
    completed_at: Optional[str]
    request_counts: Dict[str, int]  # total, completed, failed
    batch_index: int = 0  # Index in multi-batch sequence


@dataclass
class MultiBatchState:
    """State for multiple batch jobs (handles OpenAI size limits)."""
    batches: List[BatchState]
    total_requests: int
    jsonl_files: List[str]  # Paths to JSONL files


def _save_multi_batch_state(state: MultiBatchState, output_dir: Path, logger: logging.Logger) -> Path:
    """Save multi-batch state to JSON for recovery/resumption."""
    state_path = output_dir / config.OPENAI_BATCH_CONFIG['state_filename']

    state_dict = {
        'batches': [asdict(b) for b in state.batches],
        'total_requests': state.total_requests,
        'jsonl_files': state.jsonl_files
    }

    with open(state_path, 'w') as f:
        json.dump(state_dict, f, indent=2)

    logger.info(f"Saved multi-batch state ({len(state.batches)} batches) to: {state_path}")
    return state_path


def _load_multi_batch_state(output_dir: Path, logger: logging.Logger) -> Optional[MultiBatchState]:
    """Load existing multi-batch state if available."""
    state_path = output_dir / config.OPENAI_BATCH_CONFIG['state_filename']

    if not state_path.exists():
        return None

    with open(state_path, 'r') as f:
        state_dict = json.load(f)

    # Handle both old single-batch format and new multi-batch format
    if 'batches' in state_dict:
        batches = [BatchState(**b) for b in state_dict['batches']]
        state = MultiBatchState(
            batches=batches,
            total_requests=state_dict['total_requests'],
            jsonl_files=state_dict['jsonl_files']
        )
    else:
        # Legacy single-batch format - convert to multi-batch
        batch = BatchState(**state_dict)
        state = MultiBatchState(
            batches=[batch],
            total_requests=batch.request_counts.get('total', 0),
            jsonl_files=[]
        )

    logger.info(f"Loaded existing multi-batch state ({len(state.batches)} batches) from: {state_path}")
    return state


# =============================================================================
# OPENAI BATCH API - OPERATIONS
# =============================================================================

def _submit_openai_batch(jsonl_path: Path,
                         logger: logging.Logger) -> Tuple[str, str]:
    """
    Upload JSONL and create OpenAI batch job.

    Args:
        jsonl_path: Path to batch requests JSONL
        logger: Logger instance

    Returns:
        Tuple of (batch_id, input_file_id)
    """
    from openai import OpenAI

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    # Upload input file
    logger.info(f"Uploading batch input file: {jsonl_path}")
    with open(jsonl_path, 'rb') as f:
        file_response = client.files.create(file=f, purpose='batch')

    input_file_id = file_response.id
    logger.info(f"Uploaded file with ID: {input_file_id}")

    # Create batch
    logger.info("Creating batch job...")
    batch_response = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=config.OPENAI_BATCH_CONFIG['completion_window']
    )

    batch_id = batch_response.id
    logger.info(f"Created batch job with ID: {batch_id}")

    return batch_id, input_file_id


def _poll_batch_status(batch_id: str,
                       batch_index: int,
                       logger: logging.Logger) -> BatchState:
    """
    Poll a single batch job until completion.

    Args:
        batch_id: OpenAI batch ID
        batch_index: Index of this batch in multi-batch sequence
        logger: Logger instance

    Returns:
        Final BatchState
    """
    from openai import OpenAI

    client = OpenAI()
    poll_interval = config.OPENAI_BATCH_CONFIG['poll_interval_seconds']
    max_attempts = config.OPENAI_BATCH_CONFIG['max_poll_attempts']

    for attempt in range(max_attempts):
        batch = client.batches.retrieve(batch_id)

        state = BatchState(
            batch_id=batch.id,
            input_file_id=batch.input_file_id,
            status=batch.status,
            output_file_id=batch.output_file_id,
            error_file_id=batch.error_file_id,
            created_at=str(batch.created_at) if batch.created_at else None,
            completed_at=str(batch.completed_at) if batch.completed_at else None,
            request_counts={
                'total': batch.request_counts.total if batch.request_counts else 0,
                'completed': batch.request_counts.completed if batch.request_counts else 0,
                'failed': batch.request_counts.failed if batch.request_counts else 0
            },
            batch_index=batch_index
        )

        logger.info(
            f"[Batch {batch_index + 1}] Status: {state.status} "
            f"({state.request_counts['completed']}/{state.request_counts['total']} completed, "
            f"{state.request_counts['failed']} failed) - poll {attempt + 1}/{max_attempts}"
        )

        if state.status in ['completed', 'failed', 'expired', 'cancelled']:
            return state

        time.sleep(poll_interval)

    raise TimeoutError(f"Batch {batch_id} did not complete within {max_attempts * poll_interval} seconds")


def _download_batch_results(state: BatchState,
                            output_dir: Path,
                            logger: logging.Logger) -> Dict[str, str]:
    """
    Download batch results and parse into response dict.

    Args:
        state: Completed BatchState
        output_dir: Directory to save results
        logger: Logger instance

    Returns:
        Dict mapping custom_id to response text
    """
    from openai import OpenAI

    client = OpenAI()

    if state.status != 'completed':
        raise ValueError(f"Cannot download results for batch with status: {state.status}")

    # Download output file
    batch_idx = state.batch_index
    logger.info(f"[Batch {batch_idx + 1}] Downloading results from file: {state.output_file_id}")
    file_content = client.files.content(state.output_file_id)

    # Save raw output for debugging (include batch index in filename)
    output_path = output_dir / f'batch_output_{batch_idx}.jsonl'
    with open(output_path, 'wb') as f:
        f.write(file_content.content)
    logger.info(f"[Batch {batch_idx + 1}] Saved raw output to: {output_path}")

    # Parse results
    responses = {}
    failed_requests = []
    with open(output_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']

            if result.get('error'):
                logger.warning(f"[Batch {batch_idx + 1}] Request {custom_id} failed: {result['error']}")
                failed_requests.append({
                    'custom_id': custom_id,
                    'error': result['error']
                })
                responses[custom_id] = None
            else:
                # Extract response text
                response_body = result['response']['body']
                message_content = response_body['choices'][0]['message']['content']
                responses[custom_id] = message_content

    # Download and log error file if present
    if state.error_file_id:
        error_content = client.files.content(state.error_file_id)
        error_path = output_dir / f'batch_errors_{batch_idx}.jsonl'
        with open(error_path, 'wb') as f:
            f.write(error_content.content)
        logger.warning(f"[Batch {batch_idx + 1}] Had errors, saved to: {error_path}")

    # Save failed requests for debugging
    if failed_requests:
        logger.warning(f"[Batch {batch_idx + 1}] {len(failed_requests)} requests failed")
        with open(output_dir / f'failed_requests_{batch_idx}.json', 'w') as f:
            json.dump(failed_requests, f, indent=2)

    logger.info(f"[Batch {batch_idx + 1}] Downloaded {len(responses)} responses ({len(failed_requests)} failed)")
    return responses


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

    # 1. Subreddit (always shown)
    subreddit_name = pair['subreddit']
    subreddit_parts = [f"Subreddit: r/{subreddit_name}"]

    # Add title and description if available
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
    url = submission.get('url', '').strip()
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

    # Add URL if present
    if url:
        body_parts.append(url)

    # Add media placeholder if media exists but not included
    if has_media and not show_media:
        body_parts.append("[Image present but not shown]")

    # Add text content (always clean user mentions)
    if selftext:
        cleaned_text = _clean_user_mentions(selftext)
        body_parts.append(cleaned_text)
    elif not has_media and not url:  # Only show [No text] if there's no media and no URL
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
        # Apply template to both moderated and unmoderated threads
        for thread_type in ['moderated', 'unmoderated']:
            messages = pair[f'{thread_type}_prompt']['messages']

            # Count images actually in the content (not just in media_files)
            content = messages[0]['content']
            num_images = sum(1 for item in content if isinstance(item, dict) and item.get('type') == 'image')
            max_images_per_prompt = max(max_images_per_prompt, num_images)

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
                             context: str,
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
        context: Context string (e.g., "submission-media")
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
        # Add buffer for generation (response tokens) plus safety margin
        max_model_len = max_model_len + max_response_tokens + 50
        logger.info(f"📊 Using calculated max_model_len: {max_model_len} (including {max_response_tokens} token buffer + 50 token safety margin)")

    # Set max_num_seqs conservatively if media is included (images cause memory pressure)
    llm_kwargs = {
        'model': model_config['hf_path'],
        'tensor_parallel_size': num_gpus,
        'gpu_memory_utilization': model_config.get('gpu_memory_utilization', 0.9),
        'trust_remote_code': model_config.get('trust_remote_code', True),
        'max_model_len': max_model_len,
        'limit_mm_per_prompt': limit_mm_per_prompt,
        'seed': 0
    }

    if 'media' in context.split('-'):
        llm_kwargs['max_num_seqs'] = 32
        logger.info(f"📊 Setting max_num_seqs=32 due to media in context")
    else:
        logger.info(f"📊 Using vLLM auto max_num_seqs (no media in context)")

    llm_engine = LLM(**llm_kwargs)
    logger.info(f"✅ LLM engine initialized with tensor_parallel_size={num_gpus}")

    # Stage 1: Generate reasoning for all threads (moderated + unmoderated)
    logger.info("📝 Stage 1: Generating reasoning responses...")
    stage1_responses = _generate_stage1_vllm(thread_pairs, llm_engine, max_response_tokens, context, logger)
    logger.info(f"✅ Generated {len(stage1_responses)} × 2 Stage 1 reasoning responses")

    # Stage 2: Extract clean answers
    logger.info("🎯 Stage 2: Extracting clean answers...")
    results = _generate_stage2_vllm(thread_pairs, stage1_responses, llm_engine, max_response_tokens, context, logger)

    logger.info(f"✅ Completed two-stage evaluation for {len(results)} thread pairs")
    return results

def _generate_stage1_vllm(thread_pairs: List[Dict[str, Any]],
                         llm_engine,
                         max_response_tokens: int,
                         context: str,
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
        context: Context string (e.g., "submission-media")
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

    # Check if media should be included based on context flags
    include_media = 'media' in context.split('-')

    # Prepare inputs (flatten moderated + unmoderated)
    inputs = []
    for pair in thread_pairs:
        # Build inputs for both moderated and unmoderated threads
        for thread_type, prompts in [('mod', moderated_prompts), ('unmod', unmoderated_prompts)]:
            idx = thread_pairs.index(pair)

            # Only load and pass images if media flag is set in context
            images = []
            if include_media:
                submission = pair.get('submission', {})
                media_files = submission.get('media_files', [])

                # Load images separately for each thread to avoid sharing image objects
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
                         context: str,
                         logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Generate Stage 2 clean answer extraction by calling Stage 1 internally.

    Args:
        thread_pairs: Thread pairs
        stage1_responses: Stage 1 reasoning responses
        llm_engine: vLLM engine
        max_response_tokens: Maximum tokens for response generation (not used in Stage 2, uses fixed 10 tokens)
        context: Context string (e.g., "submission-media")
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
        context=context,
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

def evaluate_two_stage_api(thread_pairs: List[Dict[str, Any]],
                           model_config: Dict[str, Any],
                           output_dir: Path,
                           context: str,
                           max_response_tokens: int,
                           logger: logging.Logger,
                           override: bool = False) -> List[Dict[str, Any]]:
    """
    Two-stage evaluation using OpenAI Batch API (Stage 1) + local vLLM (Stage 2).

    Stage 1: OpenAI Batch API generates reasoning responses (async, cost-effective)
             Supports multiple batches if data exceeds OpenAI limits (50K requests, 200MB)
    Stage 2: Local Qwen3-VL-30B via vLLM extracts clean answers (fast, free)

    Args:
        thread_pairs: Thread pairs with prompts
        model_config: Model configuration from config.py
        output_dir: Output directory for state files and results
        context: Context string (e.g., "submission-media")
        max_response_tokens: Max tokens for Stage 1 response
        logger: Logger instance
        override: If True, re-run Stage 2 even if results exist

    Returns:
        Evaluation results list
    """
    # =========================================================================
    # CHECK FOR EXISTING RESULTS (skip if reasoning_*.json exists and not override)
    # =========================================================================
    existing_reasoning_files = list(output_dir.glob("reasoning_*.json")) if output_dir.exists() else []

    if existing_reasoning_files and not override:
        latest_reasoning_file = sorted(existing_reasoning_files)[-1]
        logger.info(f"✅ Found existing results: {latest_reasoning_file}")
        logger.info(f"   Skipping Stage 1 and Stage 2 (use --override to re-run)")

        with open(latest_reasoning_file, 'r') as f:
            results = json.load(f)

        logger.info(f"✅ Loaded {len(results)} existing results")
        return results

    if existing_reasoning_files and override:
        logger.info(f"♻️  Override mode: Will re-run Stage 2 (found {len(existing_reasoning_files)} existing result file(s))")

    # =========================================================================
    # STAGE 1: OpenAI Batch API for Reasoning (supports multiple batches)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 1: OpenAI Batch API - Generating Reasoning")
    logger.info("=" * 60)

    # Check for existing multi-batch state (resumption support)
    existing_state = _load_multi_batch_state(output_dir, logger)
    stage1_responses = {}

    if existing_state:
        # Resume from existing state
        logger.info(f"Found existing state with {len(existing_state.batches)} batch(es)")

        for batch_state in existing_state.batches:
            batch_idx = batch_state.batch_index

            if batch_state.status == 'completed':
                logger.info(f"[Batch {batch_idx + 1}] Already completed, downloading results...")
                batch_responses = _download_batch_results(batch_state, output_dir, logger)
                stage1_responses.update(batch_responses)

            elif batch_state.status in ['validating', 'in_progress', 'finalizing']:
                logger.info(f"[Batch {batch_idx + 1}] Resuming polling...")
                final_state = _poll_batch_status(batch_state.batch_id, batch_idx, logger)

                # Update state
                existing_state.batches[batch_idx] = final_state
                _save_multi_batch_state(existing_state, output_dir, logger)

                if final_state.status == 'completed':
                    batch_responses = _download_batch_results(final_state, output_dir, logger)
                    stage1_responses.update(batch_responses)
                else:
                    raise RuntimeError(f"Batch {batch_idx + 1} failed with status: {final_state.status}")

            elif batch_state.status in ['failed', 'expired', 'cancelled']:
                raise RuntimeError(f"Batch {batch_idx + 1} previously failed with status: {batch_state.status}")

            else:
                # Status is 'created' or unknown - need to submit
                logger.warning(f"[Batch {batch_idx + 1}] Status '{batch_state.status}' - may need manual intervention")

    else:
        # Create and submit new batches
        jsonl_files = _create_batch_jsonl_files(
            thread_pairs=thread_pairs,
            model_id=model_config['model_id'],
            max_tokens=max_response_tokens,
            context=context,
            output_path=output_dir,
            logger=logger
        )

        # Initialize multi-batch state
        batch_states = []
        total_requests = len(thread_pairs) * 2

        # Submit all batches
        for batch_idx, jsonl_path in enumerate(jsonl_files):
            logger.info(f"[Batch {batch_idx + 1}/{len(jsonl_files)}] Submitting...")
            batch_id, input_file_id = _submit_openai_batch(jsonl_path, logger)

            batch_state = BatchState(
                batch_id=batch_id,
                input_file_id=input_file_id,
                status='created',
                output_file_id=None,
                error_file_id=None,
                created_at=datetime.now().isoformat(),
                completed_at=None,
                request_counts={'total': 0, 'completed': 0, 'failed': 0},
                batch_index=batch_idx
            )
            batch_states.append(batch_state)

        # Save initial multi-batch state
        multi_state = MultiBatchState(
            batches=batch_states,
            total_requests=total_requests,
            jsonl_files=[str(p) for p in jsonl_files]
        )
        _save_multi_batch_state(multi_state, output_dir, logger)

        # Poll all batches until completion
        for batch_idx, batch_state in enumerate(batch_states):
            logger.info(f"[Batch {batch_idx + 1}/{len(batch_states)}] Polling for completion...")
            final_state = _poll_batch_status(batch_state.batch_id, batch_idx, logger)

            # Update state
            multi_state.batches[batch_idx] = final_state
            _save_multi_batch_state(multi_state, output_dir, logger)

            if final_state.status == 'completed':
                batch_responses = _download_batch_results(final_state, output_dir, logger)
                stage1_responses.update(batch_responses)
            else:
                raise RuntimeError(f"Batch {batch_idx + 1} failed with status: {final_state.status}")

    logger.info(f"Stage 1 complete: {len(stage1_responses)} total responses collected")

    # Organize Stage 1 responses by pair
    stage1_by_pair = []
    for pair in thread_pairs:
        mod_id = f"{pair['mod_comment_id']}_moderated"
        unmod_id = f"{pair['mod_comment_id']}_unmoderated"

        stage1_by_pair.append({
            'moderated': stage1_responses.get(mod_id, '') or '',
            'unmoderated': stage1_responses.get(unmod_id, '') or ''
        })

    # =========================================================================
    # STAGE 2: Local Qwen3-VL-30B via vLLM for Answer Extraction
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STAGE 2: Local vLLM - Extracting Clean Answers")
    logger.info("=" * 60)

    stage2_model = model_config.get('stage2_model', 'qwen3-vl-30b-instruct')
    stage2_config = config.get_model_config(stage2_model)

    # Apply chat template for Stage 2 model (gets base prompt with assistant turn prefix)
    thread_pairs, resource_stats = apply_chat_template(thread_pairs, stage2_model, logger)

    # Build Stage 2 prompts FIRST (Stage 1 reasoning + answer phrase as prefill)
    # These are appended after the assistant turn prefix from chat template
    moderated_prompts = [
        pair['moderated_prompt_text'] + stage1['moderated'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_by_pair)
    ]
    unmoderated_prompts = [
        pair['unmoderated_prompt_text'] + stage1['unmoderated'] + config.ANSWER_PHRASE
        for pair, stage1 in zip(thread_pairs, stage1_by_pair)
    ]

    # Calculate actual max token length for Stage 2 prompts (including reasoning)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        stage2_config['hf_path'],
        trust_remote_code=True
    )

    max_stage2_tokens = 0
    for mod_prompt, unmod_prompt in zip(moderated_prompts, unmoderated_prompts):
        mod_tokens = len(tokenizer.encode(mod_prompt))
        unmod_tokens = len(tokenizer.encode(unmod_prompt))
        max_stage2_tokens = max(max_stage2_tokens, mod_tokens, unmod_tokens)

    # Add buffer for generation (10 tokens) + safety margin
    stage2_max_model_len = max_stage2_tokens + 50
    logger.info(f"📊 Stage 2 max tokens (with reasoning): {max_stage2_tokens}, using max_model_len: {stage2_max_model_len}")

    # Initialize vLLM engine for Stage 2
    from vllm import LLM, SamplingParams

    logger.info(f"Initializing vLLM engine for Stage 2 ({stage2_model})...")

    # Determine GPU count from CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    num_gpus = len(cuda_devices.split(','))

    llm_engine = LLM(
        model=stage2_config['hf_path'],
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=stage2_config.get('gpu_memory_utilization', 0.95),
        trust_remote_code=True,
        max_model_len=stage2_max_model_len,
        seed=0
    )
    logger.info(f"✅ Stage 2 LLM engine initialized with {num_gpus} GPU(s)")

    # Generate Stage 2 answers (no images needed - just text extraction)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Use text-only context for Stage 2 (no media needed)
    stage2_context = context.replace('-media', '') if 'media' in context else context

    stage2_responses = _generate_stage1_vllm(
        thread_pairs=thread_pairs,
        llm_engine=llm_engine,
        max_response_tokens=10,
        context=stage2_context,  # No media for answer extraction
        logger=logger,
        moderated_prompts=moderated_prompts,
        unmoderated_prompts=unmoderated_prompts,
        sampling_params=sampling_params,
        uuid_suffix='_s2_api'
    )

    # =========================================================================
    # BUILD FINAL RESULTS
    # =========================================================================
    results = []
    for idx, (pair, stage1, stage2) in enumerate(zip(thread_pairs, stage1_by_pair, stage2_responses)):
        mod_clean = stage2['moderated']
        unmod_clean = stage2['unmoderated']

        mod_prediction = _extract_answer_choice(mod_clean)
        unmod_prediction = _extract_answer_choice(unmod_clean)

        mod_correct = pair['moderated_correct_answer']
        unmod_correct = pair['unmoderated_correct_answer']

        mod_score = 1 if mod_prediction == mod_correct else 0
        unmod_score = 1 if unmod_prediction == unmod_correct else 0

        result = {
            'mod_comment_id': pair['mod_comment_id'],
            'subreddit': pair['subreddit'],
            'submission_id': pair['submission_id'],

            'moderated': {
                'stage2_prompt': moderated_prompts[idx],
                'reasoning_response': stage1['moderated'],
                'clean_answer_response': mod_clean,
                'extracted_prediction': mod_prediction,
                'correct_answer': mod_correct,
                'score': mod_score,
                'answer_options': pair['moderated_answer_options']
            },

            'unmoderated': {
                'stage2_prompt': unmoderated_prompts[idx],
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

    # Save results to batch output directory for caching
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reasoning_path = output_dir / f"reasoning_{timestamp}.json"
    with open(reasoning_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Batch reasoning saved to: {reasoning_path}")

    logger.info(f"✅ Completed hybrid two-stage evaluation for {len(results)} thread pairs")
    return results

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
