# Data Format Specification

This document describes the structure and contents of the hydrated dataset produced by the Reddit Mod Collection Pipeline.

## Overview

The dataset is a JSON file containing Reddit moderation data, including subreddit rules, submissions, comment trees, and matched thread pairs showing moderator actions. The data is designed for research on content moderation and rule enforcement on Reddit.

## Top-Level Structure

```json
{
 "metadata": { ... },
 "subreddits": [ ... ]
}
```

## Metadata

**Location:** `data["metadata"]`

```json
{
 "metadata": { ... },  // ← You are here
 "subreddits": [ ... ]
}
```

The `metadata` object contains dataset-level information:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Dataset version (e.g., "1.0") |
| `split` | string | Dataset split (e.g., "train", "val", "test", "test_1k") |
| `creation_date` | string | Date when dataset was created (YYYY-MM-DD) |
| `total_subreddits` | integer | Total number of subreddits in dataset |
| `total_thread_pairs` | integer | Total number of thread pairs across all subreddits |
| `total_submissions` | integer | Total number of submissions |
| `total_submissions_with_media` | integer | Number of submissions containing media files |
| `total_media_files` | integer | Total number of media files |
| `total_comments` | integer | Total number of comments across all thread pairs (moderated + unmoderated threads) |
| `embedding_model` | string | Name of the embedding model used for similarity matching |
| `gold_threshold` | float | Threshold for high-confidence rule matches |
| `ambiguous_threshold` | float | Threshold for ambiguous rule matches |
| `date_range` | array[string] | Start and end dates for submissions in dataset |
| `pipeline_statistics` | object | See Pipeline Statistics below |

### Pipeline Statistics

**Location:** `data["metadata"]["pipeline_statistics"]`

```json
{
 "metadata": {
  ...
  "pipeline_statistics": { ... }  // ← You are here
 },
 "subreddits": [ ... ]
}
```

Statistics from each stage of the data collection pipeline:

| Field | Type | Description |
|-------|------|-------------|
| `stage1_total_mod_comments` | integer | Total moderator comments collected in Stage 1 |
| `stage4_matched_comments` | integer | Comments matched to rules in Stage 4 |
| `stage6_successful_thread_pairs` | integer | Successfully built thread pairs in Stage 6 |
| `stage7_submissions_collected` | integer | Submissions collected in Stage 7 |

## Subreddit Objects

**Location:** `data["subreddits"][i]` (where `i` is the subreddit index)

```json
{
 "metadata": { ... },
 "subreddits": [  // ← Array of subreddit objects
  {
   "subreddit": "...",
   "rules": [ ... ],
   "submissions": { ... },
   "submission_trees": { ... },
   "thread_pairs": [ ... ],
   ...
  }
 ]
}
```

The `subreddits` array contains objects with the following structure:

### Subreddit-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `subreddit` | string | Name of the subreddit (without "r/" prefix) |
| `rules` | array[object] | List of subreddit rules (see Rules below) |
| `total_thread_pairs` | integer | Number of thread pairs for this subreddit |
| `rule_distribution` | object | Mapping of rule names to their frequency counts |
| `jsd_from_uniform` | float | Jensen-Shannon divergence from uniform distribution (measures rule balance) |
| `submissions` | object | Mapping of submission IDs to submission objects |
| `submission_trees` | object | Mapping of submission IDs to comment tree structures |
| `thread_pairs` | array[object] | Thread pairs showing moderation examples |
| `data_version` | string | Version of data for this subreddit |
| `last_updated` | string | Last update timestamp |
| `language` | string | Primary language of subreddit |
| `rank` | integer | Subreddit popularity rank |

### Rules

**Location:** `data["subreddits"][i]["rules"][j]` (where `j` is the rule index)

```json
{
 "subreddits": [
  {
   "subreddit": "...",
   "rules": [  // ← Array of rule objects
    {
     "short_name": "...",
     "description": "...",
     ...
    }
   ],
   ...
  }
 ]
}
```

Each rule object contains:

| Field | Type | Description |
|-------|------|-------------|
| `subreddit` | string | Subreddit name |
| `kind` | string | Rule applies to ("all", "link", "comment") |
| `description` | string | Original rule description with formatting |
| `short_name` | string | Short name/title of the rule |
| `violation_reason` | string | Reason text shown when rule is violated |
| `created_utc` | integer | Unix timestamp when rule was created |
| `priority` | integer | Rule priority/ordering |
| `description_html` | string | HTML-formatted description |
| `rule_index` | integer | Index of rule in subreddit's rule list |
| `short_name_clean` | string | Cleaned version of short name |
| `description_clean` | string | Plain text description without formatting |
| `violation_reason_clean` | string | Cleaned violation reason |
| `rule_comprehensive` | string | Combined rule text for embedding/matching |

### Submissions

**Location:** `data["subreddits"][i]["submissions"][submission_id]`

```json
{
 "subreddits": [
  {
   "subreddit": "...",
   "submissions": {  // ← Object keyed by submission ID
    "abc123": {
     "id": "abc123",
     "title": "...",
     "selftext": "...",
     ...
    },
    "def456": { ... }
   },
   ...
  }
 ]
}
```

Submissions are stored as objects keyed by submission ID. Each submission contains extensive Reddit metadata:

#### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Reddit submission ID (base36) |
| `title` | string | Submission title |
| `selftext` | string | Self-post text content (empty for link posts) |
| `author` | string | Username of author |
| `author_created_utc` | integer | Unix timestamp of author account creation |
| `created_utc` | integer | Unix timestamp of submission creation |
| `subreddit` | string | Subreddit name |
| `subreddit_id` | string | Reddit subreddit ID |
| `permalink` | string | Relative URL to submission |
| `url` | string | Full URL (external for link posts, reddit URL for self posts) |

#### Engagement Metrics

| Field | Type | Description |
|-------|------|-------------|
| `score` | integer | Net upvotes (upvotes - downvotes) |
| `upvote_ratio` | float | Ratio of upvotes to total votes (0.0-1.0) |
| `num_comments` | integer | Total number of comments |
| `num_crossposts` | integer | Number of times crossposted |
| `gilded` | integer | Number of Reddit Gold awards |
| `gildings` | object | Breakdown of award types |
| `all_awardings` | array | List of all awards received |
| `total_awards_received` | integer | Total count of awards |

#### Metadata Flags

| Field | Type | Description |
|-------|------|-------------|
| `is_self` | boolean | Whether this is a self/text post |
| `is_video` | boolean | Whether submission is a video |
| `is_original_content` | boolean | Whether marked as OC |
| `is_meta` | boolean | Whether post is meta-discussion |
| `over_18` | boolean | Whether NSFW |
| `spoiler` | boolean | Whether marked as spoiler |
| `archived` | boolean | Whether archived (no new comments/votes) |
| `locked` | boolean | Whether locked by moderators |
| `stickied` | boolean | Whether pinned to top of subreddit |
| `pinned` | boolean | Whether pinned |
| `hidden` | boolean | Whether hidden |
| `quarantine` | boolean | Whether in quarantined subreddit |

#### Flair

| Field | Type | Description |
|-------|------|-------------|
| `link_flair_text` | string | Link flair text |
| `link_flair_css_class` | string | CSS class for link flair |
| `link_flair_background_color` | string | Background color for flair |
| `link_flair_text_color` | string | Text color for flair |
| `link_flair_richtext` | array | Structured flair content |
| `author_flair_text` | string | Author's flair text |
| `author_flair_css_class` | string | CSS class for author flair |
| `author_flair_richtext` | array | Structured author flair |

#### Media

| Field | Type | Description |
|-------|------|-------------|
| `num_media` | integer | Number of media files in submission |
| `media_files` | array[string] | List of media file paths/URLs |
| `media` | object | Reddit media metadata (videos, embeds) |
| `media_embed` | object | Embedded media HTML/metadata |
| `secure_media` | object | HTTPS version of media |
| `secure_media_embed` | object | HTTPS version of embed |
| `thumbnail` | string | Thumbnail URL or type ("self", "default") |
| `thumbnail_width` | integer | Thumbnail width in pixels |
| `thumbnail_height` | integer | Thumbnail height in pixels |

#### Moderation

| Field | Type | Description |
|-------|------|-------------|
| `removed_by_category` | string | Reason for removal (null if not removed) |
| `distinguished` | string | Whether distinguished ("moderator", "admin", null) |
| `removal_reason` | string | Removal reason if applicable |

#### Other Fields

| Field | Type | Description |
|-------|------|-------------|
| `retrieved_utc` | integer | When this data was retrieved from Reddit |
| `edited` | boolean/float | False if not edited, timestamp if edited |
| `domain` | string | Domain of link (e.g., "self.subreddit", "youtube.com") |
| `contest_mode` | boolean | Whether in contest mode |
| `send_replies` | boolean | Whether author receives inbox replies |
| `subreddit_subscribers` | integer | Number of subreddit subscribers at retrieval time |
| `subreddit_type` | string | Subreddit type ("public", "private", "restricted") |

### Submission Trees

**Location:** `data["subreddits"][i]["submission_trees"][submission_id]`

```json
{
 "subreddits": [
  {
   "subreddit": "...",
   "submission_trees": {  // ← Object keyed by submission ID
    "abc123": {
     "root_comments": [ ... ],
     "parent_map": { ... },
     "children_map": { ... },
     "depth_levels": { ... },
     "total_comments": 123
    }
   },
   ...
  }
 ]
}
```

For each submission ID, the `submission_trees` object contains the comment tree structure:

| Field | Type | Description |
|-------|------|-------------|
| `root_comments` | array[string] | List of top-level comment IDs |
| `parent_map` | object | Mapping of comment IDs to their parent comment ID |
| `children_map` | object | Mapping of comment IDs to arrays of child comment IDs |
| `depth_levels` | object | Mapping of comment IDs to their depth in the tree (0 = root) |
| `total_comments` | integer | Total number of comments in the tree |

This structure enables efficient traversal of comment threads without storing redundant tree information in each comment object.

### Thread Pairs

**Location:** `data["subreddits"][i]["thread_pairs"][k]` (where `k` is the thread pair index)

```json
{
 "subreddits": [
  {
   "subreddit": "...",
   "thread_pairs": [  // ← Array of thread pair objects
    {
     "submission_id": "...",
     "mod_comment": { ... },
     "matched_rule": { ... },
     "moderated_thread": [ ... ],
     "unmoderated_thread": [ ... ],
     "unmod_thread_metadata": { ... },
     "rule_options": [ ... ]
    }
   ],
   ...
  }
 ]
}
```

Thread pairs represent moderation examples, showing both the moderated version and an unmoderated comparison thread. Each thread pair contains:

| Field | Type | Description |
|-------|------|-------------|
| `submission_id` | string | ID of the parent submission |
| `submission_date` | integer | Unix timestamp of submission |
| `mod_comment_id` | string | ID of the moderator removal comment |
| `mod_comment_date` | integer | Unix timestamp of mod comment |
| `mod_comment` | object | Full moderator comment object (see Comment Structure) |
| `matched_rule` | object | Rule that was matched (subset of rule fields) |
| `moderated_thread` | array[object] | Comment chain leading to moderation (see Comment Structure) |
| `unmoderated_thread` | array[object] | Comparison thread that was not moderated |
| `unmod_thread_metadata` | object | Metadata about the comparison (see below) |
| `rule_options` | array[object] | Multiple choice options for rule classification |

#### Matched Rule (in Thread Pairs)

**Location:** `data["subreddits"][i]["thread_pairs"][k]["matched_rule"]`

| Field | Type | Description |
|-------|------|-------------|
| `short_name_clean` | string | Cleaned rule name |
| `description_clean` | string | Cleaned rule description |
| `rule_index` | integer | Index of rule |
| `similarity_score` | float | Embedding similarity score (0.0-1.0) |

#### Unmoderated Thread Metadata

**Location:** `data["subreddits"][i]["thread_pairs"][k]["unmod_thread_metadata"]`

| Field | Type | Description |
|-------|------|-------------|
| `target_length` | integer | Length of target (moderated) thread |
| `common_ancestors` | integer | Number of shared ancestors between threads |
| `moderated_comment_depth` | integer | Depth of moderated comment in tree |
| `moderated_score` | integer | Score of the moderated comment |
| `unmoderated_score` | integer | Score of the unmoderated comparison comment |

#### Rule Options

**Location:** `data["subreddits"][i]["thread_pairs"][k]["rule_options"][m]`

Each rule option provides a multiple choice option:

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Option label (e.g., "(a)", "(b)") |
| `rule` | string | Rule name for this option |

### Comment Structure

**Locations:**
- `data["subreddits"][i]["thread_pairs"][k]["mod_comment"]`
- `data["subreddits"][i]["thread_pairs"][k]["moderated_thread"][n]`
- `data["subreddits"][i]["thread_pairs"][k]["unmoderated_thread"][n]`

```json
{
 "subreddits": [
  {
   "thread_pairs": [
    {
     "mod_comment": { ... },  // ← Single comment object
     "moderated_thread": [  // ← Array of comment objects
      { "id": "...", "body": "...", ... },
      { "id": "...", "body": "...", ... }
     ],
     "unmoderated_thread": [ ... ]  // ← Array of comment objects
    }
   ]
  }
 ]
}
```

Comments appear in `moderated_thread`, `unmoderated_thread`, and `mod_comment` fields. Each comment contains:

#### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Reddit comment ID |
| `body` | string | Original comment text with markdown |
| `body_clean` | string | Cleaned comment text |
| `author` | string | Username of comment author |
| `author_created_utc` | integer | Unix timestamp of author account creation |
| `created_utc` | integer | Unix timestamp of comment creation |
| `subreddit` | string | Subreddit name |
| `subreddit_id` | string | Reddit subreddit ID |
| `permalink` | string | Relative URL to comment |

#### Thread Position

| Field | Type | Description |
|-------|------|-------------|
| `link_id` | string | ID of parent submission (with "t3_" prefix) |
| `parent_id` | string | ID of parent comment or submission |
| `level` | integer | Depth in comment tree (0 = top-level) |

#### Engagement

| Field | Type | Description |
|-------|------|-------------|
| `score` | integer | Net upvotes |
| `controversiality` | integer | Controversy indicator (0 or 1) |
| `gilded` | integer | Number of gildings |
| `gildings` | object | Breakdown by gild type |

#### Flags

| Field | Type | Description |
|-------|------|-------------|
| `distinguished` | string | "moderator", "admin", or null |
| `stickied` | boolean | Whether pinned |
| `is_submitter` | boolean | Whether author is submission author |
| `collapsed` | boolean | Whether collapsed |
| `archived` | boolean | Whether archived |
| `send_replies` | boolean | Whether author receives inbox replies |
| `edited` | boolean/float | False if not edited, timestamp if edited |

#### Author Flair

| Field | Type | Description |
|-------|------|-------------|
| `author_flair_text` | string | Author's flair text |
| `author_flair_css_class` | string | CSS class for flair |
| `author_flair_richtext` | array | Structured flair content |
| `author_flair_background_color` | string | Flair background color |
| `author_flair_text_color` | string | Flair text color |

#### Moderation

| Field | Type | Description |
|-------|------|-------------|
| `removal_reason` | string | Removal reason if removed |
| `removal_reason_clean` | string | Cleaned removal reason |
| `matched_rule` | object | Matched rule (only in mod comments) |

#### Other

| Field | Type | Description |
|-------|------|-------------|
| `retrieved_on` | integer | When retrieved from Reddit |
| `subreddit_name_prefixed` | string | Subreddit with "r/" prefix |
| `subreddit_type` | string | Subreddit type |
| `no_follow` | boolean | Whether has nofollow attribute |
| `can_gild` | boolean | Whether can be gilded |
| `can_mod_post` | boolean | Whether user can moderate |
| `collapsed_reason` | string | Reason for collapse if applicable |

## Data Quality Notes

1. **Deleted/Removed Content**: Comments and submissions may show `[deleted]` or `[removed]` in text fields if the content was deleted by the user or removed by moderators after initial collection.

2. **Author Information**: If an author's account is deleted, the `author` field will show `[deleted]` and `author_created_utc` may be null.

3. **Media Files**: Media files are referenced by path/URL in the `media_files` array. The actual media content may or may not be included in the dataset distribution.

4. **Timestamps**: All timestamps are Unix epoch timestamps (seconds since January 1, 1970 UTC).

5. **Score Accuracy**: Scores (upvotes/downvotes) represent the values at retrieval time and may differ from current values.

6. **Flair**: Flair formatting has changed over Reddit's history. Older content may have different flair structures than newer content.

## Usage Examples

### Accessing a specific submission:
```python
import json

with open('eval_hydrated.json') as f:
    data = json.load(f)

# Get first subreddit
subreddit = data['subreddits'][0]
print(f"Subreddit: {subreddit['subreddit']}")

# Get a specific submission
submission_id = list(subreddit['submissions'].keys())[0]
submission = subreddit['submissions'][submission_id]
print(f"Title: {submission['title']}")
print(f"Score: {submission['score']}")
```

### Traversing comment trees:
```python
# Get submission tree
tree = subreddit['submission_trees'][submission_id]

# Get root comments
for root_id in tree['root_comments']:
    print(f"Root comment: {root_id}")

    # Get children
    if root_id in tree['children_map']:
        for child_id in tree['children_map'][root_id]:
            depth = tree['depth_levels'][child_id]
            print(f"  {'  ' * depth}Child: {child_id}")
```

### Analyzing thread pairs:
```python
for pair in subreddit['thread_pairs']:
    print(f"Matched rule: {pair['matched_rule']['short_name_clean']}")
    print(f"Similarity: {pair['matched_rule']['similarity_score']:.3f}")
    print(f"Moderated thread length: {len(pair['moderated_thread'])}")
    print(f"Unmoderated thread length: {len(pair['unmoderated_thread'])}")
```

## Version History

- **v1.0** (2025-10-26): Initial release with eval split
