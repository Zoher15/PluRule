"""
Reddit-specific utility functions.

Provides functions for detecting bots, validating comments,
processing Reddit data structures, and cleaning text.
"""

import re
import html
from typing import Dict, Any, Optional


def is_bot_or_automoderator(author: str) -> bool:
    """
    Check if author is a bot or AutoModerator.

    Args:
        author: Reddit username

    Returns:
        True if author appears to be a bot
    """
    if not author or author.lower() in ['[deleted]', '[removed]']:
        return True

    author_lower = author.lower()

    # Common bot indicators
    bot_indicators = [
        'bot', 'automod', 'automoderator', 'modteam', '-modteam',
        'removal', 'helper', 'reminder', 'converter', 'translator'
    ]

    # Check if author contains bot indicators
    for indicator in bot_indicators:
        if indicator in author_lower:
            return True

    # Check for common bot naming patterns
    if (author_lower.endswith('bot') or
        author_lower.endswith('-bot') or
        author_lower.endswith('_bot') or
        author_lower.startswith('bot-') or
        author_lower.startswith('bot_')):
        return True

    return False


def is_moderator_comment(comment: Dict[str, Any]) -> bool:
    """
    Check if a comment is a distinguished moderator comment.

    Args:
        comment: Comment data dictionary

    Returns:
        True if comment is distinguished as a moderator
    """
    return comment.get('distinguished') == 'moderator'


def is_moderator_reply_to_comment(comment: Dict[str, Any]) -> bool:
    """
    Check if a comment is a distinguished moderator comment replying to another comment.

    Args:
        comment: Comment data dictionary

    Returns:
        True if comment is from a moderator replying to another comment (not submission)
    """
    return (comment.get('distinguished') == 'moderator' and
            comment.get('parent_id', '').startswith('t1_'))


def extract_submission_id(link_id: str) -> Optional[str]:
    """
    Extract submission ID from Reddit link_id.

    Args:
        link_id: Reddit link_id (e.g., "t3_abc123")

    Returns:
        Submission ID without prefix, or None if invalid
    """
    if not link_id:
        return None

    if link_id.startswith('t3_'):
        return link_id[3:]

    return link_id if link_id else None


def extract_comment_id(parent_id: str) -> Optional[str]:
    """
    Extract comment ID from Reddit parent_id.

    Args:
        parent_id: Reddit parent_id (e.g., "t1_abc123")

    Returns:
        Comment ID without prefix, or None if invalid
    """
    if not parent_id:
        return None

    if parent_id.startswith('t1_'):
        return parent_id[3:]

    return parent_id if parent_id else None


def normalize_subreddit_name(subreddit: str) -> str:
    """
    Normalize subreddit name to lowercase and remove prefixes.

    Args:
        subreddit: Raw subreddit name

    Returns:
        Normalized subreddit name
    """
    if not subreddit:
        return ""

    # Remove r/ prefix if present
    if subreddit.startswith('r/'):
        subreddit = subreddit[2:]

    return subreddit.lower().strip()


def clean_rule_text(text: str) -> str:
    """
    Clean rule text by removing markdown, URLs, and prefixes.

    Args:
        text: Raw rule text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove markdown links but keep text: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove standalone URLs
    text = re.sub(r'https?://[^\s]+', '', text)

    # Remove markdown emphasis: **text** or *text* -> text
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)

    # Remove markdown headers: ## Header -> Header
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown code blocks: ```code``` -> code
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove common prefixes from rule names/descriptions
    # Patterns like "Rule 1: No spam", "Rule 1 - Be nice", "General: Be respectful"
    # Only remove prefix if followed by punctuation (: - .) AND content after it
    # This preserves standalone "Rule 1" or "General" as valid rule names
    original_text = text
    text = re.sub(r'^Rule\s*\d+[\.\w]*\s*[:\-\.]\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^General\s*[:\-\.]\s+', '', text, flags=re.IGNORECASE)

    # If cleaning resulted in empty string, restore original
    if not text.strip():
        text = original_text

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def has_media(submission: Dict[str, Any]) -> bool:
    """
    Check if submission contains media content.

    Args:
        submission: Submission data dictionary

    Returns:
        True if submission has media content
    """
    # Check various media indicators
    media_indicators = [
        submission.get('is_video', False),
        submission.get('is_gallery', False),
        submission.get('is_reddit_media_domain', False),
        bool(submission.get('media')),
        bool(submission.get('media_embed')),
        bool(submission.get('media_metadata')),
        bool(submission.get('secure_media')),
        bool(submission.get('secure_media_embed')),
        submission.get('post_hint') in ['image', 'rich:video', 'hosted:video', 'link'],
        submission.get('url', '').endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.webm')),
        'i.redd.it' in submission.get('url', ''),
        'v.redd.it' in submission.get('url', ''),
        'imgur.com' in submission.get('url', ''),
        'youtube.com' in submission.get('url', ''),
        'youtu.be' in submission.get('url', '')
    ]

    return any(media_indicators)


def validate_comment_structure(comment: Dict[str, Any]) -> bool:
    """
    Validate that a comment has required fields.

    Args:
        comment: Comment data dictionary

    Returns:
        True if comment has valid structure
    """
    required_fields = ['id', 'author', 'body', 'subreddit', 'created_utc']
    return all(field in comment for field in required_fields)


def validate_submission_structure(submission: Dict[str, Any]) -> bool:
    """
    Validate that a submission has required fields.

    Args:
        submission: Submission data dictionary

    Returns:
        True if submission has valid structure
    """
    required_fields = ['id', 'author', 'title', 'subreddit', 'created_utc']
    return all(field in submission for field in required_fields)


def filter_reddit_line(line: str, check_func) -> bool:
    """
    Fast filtering of Reddit JSON lines with pre-check optimization.

    Args:
        line: JSON line from Reddit file
        check_func: Function to check parsed JSON object

    Returns:
        True if line passes the check
    """
    try:
        from utils.files import json_loads
        obj = json_loads(line)
        return check_func(obj)
    except:
        return False


def build_subreddit_stats(comments: list) -> Dict[str, int]:
    """
    Build statistics by subreddit from a list of comments.

    Args:
        comments: List of comment dictionaries

    Returns:
        Dictionary mapping subreddit -> count
    """
    stats = {}
    for comment in comments:
        subreddit = normalize_subreddit_name(comment.get('subreddit', 'unknown'))
        stats[subreddit] = stats.get(subreddit, 0) + 1
    return stats