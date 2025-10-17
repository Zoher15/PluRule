# Phase 3: Thread Construction

**Stages**: 5-6
**Purpose**: Build comment trees and create moderated/unmoderated discussion thread pairs

---

## Stage 5: Collect Submission Comments

**Script**: `scripts/5_collect_submission_comments.py`

**Purpose**: Collect all comments for target submissions identified in Stage 4.

**Process Flow** (3-phase):
1. **Phase 1 - Filter**: Stream RC files → filter for target submissions → write to temp subdirectories
2. **Phase 2 - Consolidate**: For each subreddit → organize comments by submission → save as pickle
3. **Phase 3 - Cleanup**: Remove temp directories

### Functions

#### `load_submission_ids(logger: Logger) -> Tuple[Set[str], Dict[str, Set[str]], Set[str]]`
Load submission IDs from Stage 4 output.

**Parameters**:
- `logger`: Logger instance

**Returns**: Tuple of (all_submission_ids, subreddit_to_ids, target_subreddits)
- `all_submission_ids`: Set of all unique submission IDs
- `subreddit_to_ids`: Dict mapping subreddit -> set of submission IDs
- `target_subreddits`: Set of normalized subreddit names

**Description**: Reads `stage4_subreddit_submission_ids.json`, creates multiple lookup structures for efficient filtering during RC file processing. Normalizes all subreddit names.

---

#### `get_rc_files(logger: Logger = None) -> List[str]`
Get list of RC files in date range.

**Parameters**:
- `logger`: Optional logger instance

**Returns**: List of RC file paths

**Description**: Uses `get_files_in_date_range` utility to find RC files within configured date range from PATHS['reddit_comments'].

---

#### `process_rc_file(args: Tuple) -> Dict[str, Any]`
Process single RC file to extract comments for target submissions (Phase 1 - Filter).

**Parameters**:
- `args`: Tuple of (rc_file_path, subreddit_to_ids, target_subreddits, temp_dir) for parallel processing

**Returns**: Dictionary with keys:
- `rc_file`: str (filename)
- `total_lines`: int
- `matched_comments`: int
- `subreddits_with_comments`: int
- `processing_time`: float
- `success`: bool

**Description**: Streams RC file, checks if comment belongs to target submission via link_id, writes to temp subdirectory using multi-output pattern. Each subreddit gets its own temp subdirectory. Uses worker logger with RC file identifier in `rc_files/` subdirectory.

---

#### `organize_subreddit_comments(args: Tuple) -> Dict[str, Any]`
Organize comments for a single subreddit into nested structure (Phase 2 - Consolidate).

**Parameters**:
- `args`: Tuple of (subreddit, target_submission_ids, temp_dir, output_dir) for parallel processing

**Returns**: Dictionary with keys:
- `subreddit`: str
- `comments_organized`: int
- `submissions_with_comments`: int
- `files_processed`: int
- `processing_time`: float
- `success`: bool

**Description**: Reads all temp RC files for a subreddit in chronological order, organizes into nested structure `{submission_id: {comment_id: comment}}`, saves as pickle with HIGHEST_PROTOCOL. Uses worker logger with subreddit identifier in `subreddits/` subdirectory.

---

#### `main() -> int`
Main execution function for Stage 5.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates 3-phase processing:
- Phase 1: Loads submission IDs, processes RC files in parallel
- Phase 2: Consolidates per subreddit in parallel
- Phase 3: Cleans up temp files with shutil.rmtree
Writes summary JSON with statistics.

---

### Input/Output

**Inputs**:
- `reddit_comments/RC_*.zst` from Stage 0
- `data/stage4_subreddit_submission_ids.json` from Stage 4

**Outputs**:
```
output/
├── organized_comments/
│   ├── {subreddit}_submission_comments.pkl
│   └── temp/ (removed after consolidation)
│       └── {subreddit}/
│           └── RC_YYYY-MM.zst
└── data/
    └── stage5_submission_comment_organization_stats.json
```

**Key Output**: Organized submission comments in `organized_comments/`

---

### Example Data Structures

**Organized Comments (pickle)**:
```python
{
    'submission_id_1': {
        'comment_id_a': {
            'id': 'comment_id_a',
            'author': 'user1',
            'body': 'This is a comment',
            'created_utc': 1640000000,
            'parent_id': 't3_submission_id_1',  # Top-level
            'score': 10
        },
        'comment_id_b': {
            'id': 'comment_id_b',
            'author': 'user2',
            'body': 'This is a reply',
            'created_utc': 1640000100,
            'parent_id': 't1_comment_id_a',  # Reply to comment_id_a
            'score': 5
        }
    },
    'submission_id_2': {
        # ... comments for this submission
    }
}
```

**Summary Output**:
```json
{
  "summary": {
    "total_subreddits": 1800,
    "successful_subreddits": 1750,
    "total_comments_organized": 85000000,
    "total_submissions_with_comments": 105000,
    "avg_comments_per_submission": 809.52,
    "phase1_time": 7200.5,
    "phase2_time": 1200.3,
    "phase3_time": 5.2,
    "total_time": 8406.0,
    "collection_date": "2025-01-09 14:30:22"
  },
  "subreddit_stats": [
    {
      "subreddit": "askreddit",
      "comments_organized": 125000,
      "submissions_with_comments": 845,
      "files_processed": 208,
      "processing_time": 15.67,
      "avg_comments_per_submission": 147.93
    }
  ]
}
```

---

## Stage 6: Build Trees and Threads

**Script**: `scripts/6_build_trees_and_threads.py`

**Purpose**: Build comment trees and create moderated/unmoderated discussion thread pairs.

**Process Flow**:
1. Load subreddit languages and complete rule sets from Stage 2
2. Find subreddits with required input files (organized comments + matched comments)
3. For each subreddit (parallel):
   - Load submission comments and matched mod comments
   - Build comment trees using BFS
   - For each mod comment, build thread pairs
   - Calculate JSD from rule distribution
4. Filter for qualified subreddits (≥500 pairs)
5. Separate English/non-English and rank by JSD

### Functions

#### `load_target_subreddits_and_rules(logger: Logger) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]`
Load subreddits with their languages and complete rule sets.

**Parameters**:
- `logger`: Logger instance

**Returns**: Tuple of (subreddit_languages, subreddit_rules)
- `subreddit_languages`: Dict mapping subreddit -> language code
- `subreddit_rules`: Dict mapping subreddit -> {rule_name: 0}

**Description**: Reads `stage2_top_N_sfw_subreddits.json`, extracts language and initializes complete rule set with 0 counts for each subreddit. Uses `short_name_clean` to match Stage 4's format.

---

#### `calculate_depth_levels(root_comments: List[str], children: Dict[str, List[str]]) -> Dict[int, List[str]]`
Calculate depth levels for comments using BFS.

**Parameters**:
- `root_comments`: List of root comment IDs
- `children`: Dictionary mapping parent_id -> [child_ids]

**Returns**: Dictionary mapping depth -> [comment_ids at this depth]

**Description**: Uses BFS to assign depth levels. Depth 0 = root comments (direct replies to submission). Tracks visited comments to avoid cycles.

---

#### `build_submission_tree(submission_id: str, comments: Dict[str, Dict]) -> Dict`
Build tree structure for a single submission.

**Parameters**:
- `submission_id`: Submission ID
- `comments`: Dictionary of comment_id -> comment object

**Returns**: Dictionary with keys:
- `children`: Dict[str, List[str]] (parent_id -> [child_ids])
- `parent_map`: Dict[str, str] (child_id -> parent_id)
- `root_comments`: List[str] (top-level comment IDs)
- `depth_levels`: Dict[int, List[str]]
- `total_comments`: int

**Description**: Builds parent-child relationships by parsing parent_id field. Cleans IDs using `extract_submission_id` and `extract_comment_id` utilities. Calls `calculate_depth_levels` for BFS depth assignment.

---

#### `build_subreddit_trees(submission_comments: Dict[str, Dict]) -> Dict[str, Any]`
Build trees for all submissions in a subreddit's data.

**Parameters**:
- `submission_comments`: Dictionary of submission_id -> {comment_id: comment}

**Returns**: Dictionary with keys:
- `trees`: Dict[str, Dict] (submission_id -> tree structure)
- `metadata`: Dict with total_submissions, total_comments

**Description**: Calls `build_submission_tree` for each submission, aggregates metadata. Skips empty submissions.

---

#### `load_mod_comments(match_file_path: str, logger: Logger, sample_size: int = 2000) -> List[Dict]`
Load and sample moderator comments from match file.

**Parameters**:
- `match_file_path`: Path to matched comments .jsonl.zst file
- `logger`: Logger instance
- `sample_size`: Maximum number of comments to sample (default 2000)

**Returns**: List of moderator comment dictionaries

**Description**: Reads match file using `read_zst_lines`, parses JSON. If more than sample_size comments, uses `random.sample` with seed=0 for reproducibility.

---

#### `build_thread_to_root(comment_id: str, comments: Dict[str, Dict]) -> List[Dict]`
Build thread from comment up to root (excluding submission).

**Parameters**:
- `comment_id`: Starting comment ID
- `comments`: Dictionary of comment_id -> comment object

**Returns**: List of comment objects from root to leaf (chronological order)

**Description**: Traces parent_id chain from comment to root. Stops when reaching submission (parent_id starts with t3_). Reverses path to get root-to-leaf order. Adds 'level' field to each comment (0-based).

---

#### `find_common_ancestors(comment1_id: str, comment2_id: str, comments: Dict[str, Dict]) -> int`
Count common ancestors between two comments.

**Parameters**:
- `comment1_id`: First comment ID
- `comment2_id`: Second comment ID
- `comments`: Dictionary of comment_id -> comment object

**Returns**: Number of common ancestors

**Description**: Builds path to root for each comment, reverses to get root-to-leaf order, counts common prefix length. Stops at submission boundary.

---

#### `find_best_alternative(moderated_comment_id: str, moderated_thread_length: int, submission_id: str, comments: Dict[str, Dict], trees: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]`
Find best alternative comment for unmoderated thread.

**Parameters**:
- `moderated_comment_id`: ID of moderated comment
- `moderated_thread_length`: Required thread length
- `submission_id`: Submission ID
- `comments`: Dictionary of comment_id -> comment object
- `trees`: Trees data structure

**Returns**: Tuple of (best_alternative_id, moderated_depth)
- `best_alternative_id`: ID of best alternative comment (None if not found)
- `moderated_depth`: Depth of moderated comment (for failure tracking)

**Description**:
1. Finds depth of moderated comment from tree
2. Gets all comments at same depth (sorted for determinism)
3. Excludes moderated comment itself
4. For each alternative, checks if thread length ≥ required length
5. Counts common ancestors with moderated comment
6. Selects alternative with:
   - Maximum common ancestors (prefer similar context)
   - Lowest score (break ties - prefer less prominent alternatives)
7. Returns best alternative or None

---

#### `build_thread_pair(mod_comment: Dict, comments: Dict[str, Dict], trees: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[int]]`
Build moderated and unmoderated thread pair for a moderator comment.

**Parameters**:
- `mod_comment`: Moderator comment dictionary
- `comments`: Dictionary of comment_id -> comment object
- `trees`: Trees data structure

**Returns**: Tuple of (thread_pair, failed_depth)
- `thread_pair`: Complete pair dictionary (None if failed)
- `failed_depth`: Depth where matching failed (None if succeeded)

**Description**:
1. Extracts moderated_comment_id from parent_id and submission_id from link_id
2. Validates data exists
3. Builds moderated thread using `build_thread_to_root`
4. Finds best alternative at exact same depth (no fallback)
5. Builds unmoderated thread
6. Calculates common ancestors
7. Returns complete pair with metadata or (None, failed_depth)

---

#### `process_subreddit(args: Tuple) -> Dict[str, Any]`
Process single subreddit: build trees and create discussion threads.

**Parameters**:
- `args`: Tuple of (subreddit_name, complete_rule_set) for parallel processing

**Returns**: Dictionary with keys:
- `subreddit`: str
- `status`: str ('completed' or 'failed')
- `trees_built`: int
- `total_comments`: int
- `trees_file_size_gb`: float
- `successful_pairs`: int
- `total_mod_comments`: int
- `success_rate`: float
- `threads_file_size_gb`: float
- `rule_distribution`: Dict[str, int]
- `jsd_from_uniform`: float
- `debug_counts`: Dict[str, int]
- `failed_depth_distribution`: Dict[str, int]
- `successful_depth_distribution`: Dict[str, int]
- `processing_time`: float
- `error`: str (if failed)

**Description**:
1. Creates worker logger with subreddit identifier
2. Loads submission comments (pickle) and matched comments (zst)
3. Builds trees in memory using `build_subreddit_trees`
4. Saves trees to disk
5. Loads and samples mod comments (max 2000)
6. For each mod comment, builds thread pair
7. Tracks success/failure with detailed debug counts
8. Calculates JSD from rule distribution (initialized with complete rule set)
9. Saves discussion threads to disk
10. Returns comprehensive statistics

---

#### `main() -> int`
Main execution function for Stage 6.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**:
1. Loads subreddit languages and complete rule sets from Stage 2
2. Finds subreddits with both required input files
3. Processes subreddits in parallel
4. Collects statistics (trees, pairs, sizes)
5. Filters for qualified subreddits (≥500 successful pairs)
6. Adds language information to qualified results
7. Separates English (language='en') from non-English
8. Ranks English subreddits by JSD (ascending=True, lower=better)
9. Non-English subreddits get rank=None
10. Writes comprehensive summary JSON

---

### Input/Output

**Inputs**:
- `output/organized_comments/{subreddit}_submission_comments.pkl` from Stage 5
- `output/matched_comments/{subreddit}_match.jsonl.zst` from Stage 4
- `data/stage2_top_N_sfw_subreddits.json` from Stage 2 (for rules and languages)

**Outputs**:
```
output/
├── comment_trees/
│   ├── {subreddit}_comment_trees.pkl
│   └── {subreddit}_discussion_threads.pkl
├── discussion_threads/ (symlinks for compatibility)
│   └── {subreddit}_discussion_threads.pkl
└── data/
    └── stage6_trees_and_threads_summary.json
```

---

### Example Data Structures

**Tree Structure**:
```python
{
    'children': {
        'comment_1': ['comment_2', 'comment_3'],
        'comment_2': ['comment_4'],
        # parent_id -> [child_ids]
    },
    'parent_map': {
        'comment_2': 'comment_1',
        'comment_3': 'comment_1',
        'comment_4': 'comment_2',
        # child_id -> parent_id
    },
    'root_comments': ['comment_1', 'comment_5'],
    'depth_levels': {
        0: ['comment_1', 'comment_5'],
        1: ['comment_2', 'comment_3'],
        2: ['comment_4']
    },
    'total_comments': 5
}
```

**Thread Pair**:
```python
{
    'mod_comment_id': 'mod_abc123',
    'mod_comment': {
        'id': 'mod_abc123',
        'author': 'ModeratorName',
        'body': 'Your post violates Rule 1',
        'created_utc': 1640000500,
        'score': 5,
        'matched_rule': {
            'short_name': 'Post must be an open-ended question',
            'similarity_score': 0.8765
        }
    },
    'moderated_thread': [
        {'id': 'root_comment', 'level': 0, 'body': 'Original comment', ...},
        {'id': 'reply_1', 'level': 1, 'body': 'First reply', ...},
        {'id': 'moderated_comment', 'level': 2, 'body': 'User comment', ...}
    ],
    'unmoderated_thread': [
        {'id': 'root_comment_2', 'level': 0, 'body': 'Different root', ...},
        {'id': 'reply_2', 'level': 1, 'body': 'Different reply', ...},
        {'id': 'alternative', 'level': 2, 'body': 'Same depth alternative', ...}
    ],
    'metadata': {
        'common_ancestors': 1,
        'rule': 'Post must be an open-ended question',
        'rule_similarity_score': 0.8765,
        'moderated_comment_id': 'moderated_comment',
        'unmoderated_comment_id': 'alternative',
        'submission_id': 'sub123',
        'moderated_score': 10,
        'unmoderated_score': 5,
        'target_length': 3
    }
}
```

**Trees Output (pickle)**:
```python
{
    'subreddit': 'askreddit',
    'source_file': '/path/to/askreddit_submission_comments.pkl',
    'trees': {
        'submission_id_1': {
            'children': {...},
            'parent_map': {...},
            'root_comments': [...],
            'depth_levels': {...},
            'total_comments': 25
        },
        'submission_id_2': { ... }
    },
    'metadata': {
        'total_submissions': 850,
        'total_comments': 125000
    }
}
```

**Threads Output (pickle)**:
```python
{
    'subreddit': 'askreddit',
    'thread_pairs': [
        { /* thread pair 1 */ },
        { /* thread pair 2 */ }
    ],
    'metadata': {
        'total_mod_comments': 2000,
        'successful_pairs': 850,
        'success_rate': 0.425
    }
}
```

**Stage 6 Summary**:
```json
{
  "summary": {
    "total_subreddits_processed": 1750,
    "completed_subreddits": 1740,
    "failed_subreddits": 10,
    "subreddits_with_500_plus_pairs": 125,
    "english_subreddits": 98,
    "other_language_subreddits": 27,
    "total_trees_built": 147500,
    "total_comments_processed": 85000000,
    "total_successful_pairs": 156000,
    "total_mod_comments": 3500000,
    "overall_success_rate": 0.446,
    "total_trees_size_gb": 45.2,
    "total_threads_size_gb": 12.8,
    "processing_time_seconds": 5400,
    "collection_date": "2025-01-09 14:30:22"
  },
  "subreddit_stats": [
    {
      "rank": 1,
      "subreddit": "askreddit",
      "language": "en",
      "status": "completed",
      "trees_built": 845,
      "total_comments": 125000,
      "successful_pairs": 850,
      "total_mod_comments": 2000,
      "success_rate": 0.425,
      "jsd_from_uniform": 0.0234,
      "rule_distribution": {
        "Post must be an open-ended question": 350,
        "No personal information": 250,
        "Be respectful": 250
      },
      "debug_counts": {
        "missing_submission": 50,
        "missing_parent_id": 20,
        "missing_moderated_comment": 30,
        "no_alternative_found": 1050
      }
    }
  ],
  "language_breakdown": {
    "english": {
      "subreddits": 98,
      "total_pairs": 95000
    },
    "non_english": {
      "subreddits": 27,
      "total_pairs": 18000
    }
  }
}
```

---

## Phase 3 Summary

**Total Stages**: 2
**Key Outputs**:
- Organized submission comments (~1750 subreddits)
- Comment trees with BFS-based depths (~147,500 trees)
- Discussion thread pairs (moderated + unmoderated, ~156,000 pairs)
- Qualified subreddits (≥500 pairs, ~125 subreddits)
- JSD rankings (English ranked, non-English unranked)

**Key Algorithms**:
- **BFS Tree Building**: Breadth-first search for depth calculation
- **Depth Matching**: Find alternatives at exact same depth (no fallback)
- **Ancestor Matching**: Prefer alternatives with maximum common ancestors
- **Score-based Tie Breaking**: Prefer lower score for less prominent alternatives
- **JSD Ranking**: Rank subreddits by rule distribution uniformity

**Qualification Criteria**:
- Minimum 500 successful thread pairs per subreddit
- Separated by language (English vs non-English)
- Only English subreddits ranked by JSD (ascending, lower = better)

**Next Phase**: Phase 4 enriches thread pairs with submission objects, downloads media, and creates final datasets.
