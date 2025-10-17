# Phase 1: Data Acquisition & Ranking

**Stages**: 0-2
**Purpose**: Download Reddit archives and identify top SFW subreddits with moderation activity

---

## Stage 0: Download Pushshift Archives

**Script**: `scripts/0_download_data.py`

**Purpose**: Download Reddit comment (RC) and submission (RS) archives from Internet Archive for the configured date range.

**Process Flow**:
1. Generate download URLs for RC and RS files
2. Download files in parallel with skip logic for existing files
3. Organize downloads by year subdirectories

### Functions

#### `generate_download_urls(date_range: Tuple[str, str], logger: Logger = None) -> List[Tuple[str, str, str]]`
Generate download URLs for RC and RS files within date range.

**Parameters**:
- `date_range`: Tuple of (start_date, end_date) in YYYY-MM format
- `logger`: Optional logger instance

**Returns**: List of tuples (url, filename, file_type)

**Description**: Creates URLs for both RC (comments) and RS (submissions) files from Internet Archive. Organizes files by year subdirectories.

---

#### `download_file(args: Tuple[str, str, str]) -> Dict[str, Any]`
Download a single file from URL.

**Parameters**:
- `args`: Tuple of (url, filename, file_type) for parallel processing

**Returns**: Dictionary with keys:
- `filename`: str
- `url`: str
- `status`: str ('downloaded', 'skipped', 'failed')
- `size_bytes`: int
- `download_time`: float
- `output_path`: str

**Description**: Downloads file with progress tracking. Skips if already exists. Uses requests with streaming and retry logic.

---

#### `main() -> int`
Main execution function for Stage 0.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates download process: generates URLs, creates directories, downloads in parallel, reports statistics.

---

### Input/Output

**Inputs**:
- `DATE_RANGE` from config (e.g., "2005-12" to "2023-02")

**Outputs**:
```
reddit_data/
├── comments/
│   ├── 2005/RC_2005-12.zst
│   ├── ...
│   └── 2023/RC_2023-02.zst
└── submissions/
    ├── 2005/RS_2005-12.zst
    ├── ...
    └── 2023/RS_2023-02.zst
```

**Output Files**: ~208 RC files, ~208 RS files (depending on date range)

---

### Example Data Structures

**Download URL Entry**:
```python
(
    "https://files.pushshift.io/reddit/comments/RC_2023-02.zst",
    "RC_2023-02.zst",
    "comments"
)
```

**Download Result**:
```python
{
    'filename': 'RC_2023-02.zst',
    'url': 'https://files.pushshift.io/reddit/comments/RC_2023-02.zst',
    'status': 'downloaded',
    'size_bytes': 15234567890,
    'download_time': 3245.6,
    'output_path': '/gpfs/slate-cnets/datasets/reddit/Pushshift/comments/2023/RC_2023-02.zst'
}
```

---

## Stage 1: Collect Moderator Comments

**Script**: `scripts/1_collect_mod_comments.py`

**Purpose**: Extract moderator comments from RC files, filter bots/AutoMod, and rank subreddits by moderation activity.

**Process Flow**:
1. Process RC files in parallel
2. Fast pre-filter (string check before JSON parsing)
3. Extract distinguished moderator comments replying to other comments
4. Collect subreddit statistics during processing
5. Generate rankings

### Functions

#### `process_comment_line(line: str, subreddit_counts: Dict = None) -> bool`
Fast pre-filter for moderator comment lines.

**Parameters**:
- `line`: JSON line from RC file
- `subreddit_counts`: Optional dict to track counts during filtering

**Returns**: True if line contains moderator comment

**Description**: Fast string check for `"distinguished":"moderator"` and `"parent_id":"t1_"` before JSON parsing. Provides 10x speedup.

---

#### `process_single_file(file_path: str) -> Dict[str, Any]`
Process a single RC file to extract moderator comments.

**Parameters**:
- `file_path`: Path to RC_*.zst file

**Returns**: Dictionary with keys:
- `file`: str (input filename)
- `output`: str (output filename)
- `subreddit_counts`: Dict[str, int]
- `lines_processed`: int
- `lines_matched`: int
- `processing_time`: float
- `success`: bool

**Description**: Streams RC file, applies fast pre-filter, parses JSON for matched lines, filters bots/AutoMod, writes to output file. Collects subreddit statistics in-memory during processing.

---

#### `collect_subreddit_stats(results: List[Dict], logger: Logger) -> Dict[str, int]`
Aggregate subreddit statistics from parallel processing results.

**Parameters**:
- `results`: List of processing results from parallel workers
- `logger`: Logger instance

**Returns**: Dictionary mapping subreddit -> total mod comment count

**Description**: Combines subreddit counts from all RC files, no re-reading required.

---

#### `generate_rankings(subreddit_counts: Dict[str, int]) -> Dict[str, Any]`
Generate ranked list of subreddits by mod comment count.

**Parameters**:
- `subreddit_counts`: Dictionary mapping subreddit -> count

**Returns**: Dictionary with keys:
- `summary`: Dict with total_subreddits, total_mod_comments, collection_date
- `rankings`: List[Dict] with rank, subreddit, mod_comment_count

**Description**: Sorts subreddits by count descending, assigns ranks, creates summary statistics.

---

#### `main() -> int`
Main execution function for Stage 1.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates mod comment extraction: gets RC files in date range, processes in parallel, collects statistics, generates rankings, writes summary JSON.

---

### Input/Output

**Inputs**:
- `reddit_comments/RC_*.zst` files from Stage 0

**Outputs**:
```
data/
├── mod_comments/
│   └── RC_YYYY-MM_mod_comments.zst (one per RC file)
└── stage1_subreddit_mod_comment_rankings.json
```

**Key Output**: `stage1_subreddit_mod_comment_rankings.json`

---

### Example Data Structures

**Moderator Comment (extracted)**:
```python
{
    'id': 'abc123',
    'author': 'ModeratorName',
    'body': 'Your post has been removed for violating Rule 1.',
    'subreddit': 'AskReddit',
    'created_utc': 1640000000,
    'parent_id': 't1_xyz789',
    'link_id': 't3_sub123',
    'distinguished': 'moderator',
    'score': 5
}
```

**Subreddit Rankings Output**:
```json
{
  "summary": {
    "total_subreddits": 45000,
    "total_mod_comments": 12500000,
    "collection_date": "2025-01-09 14:30:22"
  },
  "rankings": [
    {"rank": 1, "subreddit": "askreddit", "mod_comment_count": 150000},
    {"rank": 2, "subreddit": "pics", "mod_comment_count": 120000},
    {"rank": 3, "subreddit": "gaming", "mod_comment_count": 95000}
  ]
}
```

---

## Stage 2: Get Top SFW Subreddits

**Script**: `scripts/2_get_top_sfw_subreddits.py`

**Purpose**: Fetch top N SFW subreddits with community rules via Reddit API using PRAW.

**Process Flow**:
1. Load rankings from Stage 1
2. Initialize Reddit API client
3. Process subreddits sequentially with retry logic
4. Check NSFW status
5. Extract subreddit metadata and rules
6. Clean rule text (remove markdown, HTML, URLs)

### Functions

#### `initialize_reddit_client(logger: Logger) -> Optional[praw.Reddit]`
Initialize PRAW Reddit client with credentials from environment.

**Parameters**:
- `logger`: Logger instance

**Returns**: Configured Reddit client or None on failure

**Description**: Reads `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` from environment variables. Creates read-only PRAW instance.

---

#### `check_nsfw_status(reddit: praw.Reddit, subreddit_name: str, logger: Logger) -> bool`
Check if subreddit is marked as NSFW.

**Parameters**:
- `reddit`: PRAW Reddit client
- `subreddit_name`: Subreddit name to check
- `logger`: Logger instance

**Returns**: True if NSFW, False if SFW

**Description**: Queries Reddit API for subreddit's `over18` flag. Handles API errors gracefully.

---

#### `extract_subreddit_data(reddit: praw.Reddit, subreddit_name: str, original_rank: int, mod_comment_count: int, logger: Logger) -> Optional[Dict[str, Any]]`
Extract subreddit metadata and information.

**Parameters**:
- `reddit`: PRAW Reddit client
- `subreddit_name`: Subreddit name
- `original_rank`: Rank from Stage 1 rankings
- `mod_comment_count`: Mod comment count from Stage 1
- `logger`: Logger instance

**Returns**: Dictionary with subreddit data or None on failure

**Description**: Extracts display_name, subscribers, created_utc, description, lang, over18 flag.

---

#### `extract_community_rules(reddit: praw.Reddit, subreddit_name: str, logger: Logger) -> List[Dict[str, Any]]`
Extract and clean community rules for a subreddit.

**Parameters**:
- `reddit`: PRAW Reddit client
- `subreddit_name`: Subreddit name
- `logger`: Logger instance

**Returns**: List of rule dictionaries

**Description**: Fetches rules via API, cleans text (removes markdown, HTML, URLs, prefixes), creates comprehensive rule text for embeddings. Each rule includes:
- `rule_index`: int (1-based)
- `short_name`: str (original)
- `description`: str (original)
- `violation_reason`: str (original)
- `short_name_clean`: str (cleaned)
- `description_clean`: str (cleaned)
- `violation_reason_clean`: str (cleaned)
- `rule_comprehensive`: str (combined for embeddings)

---

#### `process_subreddit_with_retry(reddit: praw.Reddit, subreddit_name: str, original_rank: int, mod_comment_count: int, logger: Logger) -> Optional[Dict[str, Any]]`
Process a single subreddit with retry logic.

**Parameters**:
- `reddit`: PRAW Reddit client
- `subreddit_name`: Subreddit name
- `original_rank`: Rank from Stage 1
- `mod_comment_count`: Mod comment count from Stage 1
- `logger`: Logger instance

**Returns**: Complete subreddit data dictionary or None

**Description**: Combines NSFW check, metadata extraction, and rule extraction with exponential backoff retry (up to 3 attempts). Skips NSFW subreddits.

---

#### `collect_sfw_subreddits(rankings_data: Dict, reddit: praw.Reddit, logger: Logger) -> List[Dict[str, Any]]`
Collect top N SFW subreddits with rules.

**Parameters**:
- `rankings_data`: Rankings from Stage 1
- `reddit`: PRAW Reddit client
- `logger`: Logger instance

**Returns**: List of subreddit data dictionaries

**Description**: Iterates through rankings, processes subreddits sequentially until reaching `TOP_N_SUBREDDITS_WITH_MOD_COMMENTS` SFW subreddits. Skips NSFW and failed subreddits.

---

#### `main() -> int`
Main execution function for Stage 2.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates API data collection: loads rankings, initializes Reddit client, collects SFW subreddits, writes output JSON with summary statistics.

---

### Input/Output

**Inputs**:
- `data/stage1_subreddit_mod_comment_rankings.json` from Stage 1
- Environment variables: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`

**Outputs**:
```
data/
└── stage2_top_N_sfw_subreddits.json
```

**Key Output**: `stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json` (e.g., `stage2_top_10000_sfw_subreddits.json`)

---

### Example Data Structures

**Subreddit Entry**:
```json
{
  "original_rank": 1,
  "original_mod_comment_count": 150000,
  "subreddit": {
    "display_name": "AskReddit",
    "subscribers": 45000000,
    "created_utc": 1201233600,
    "description": "r/AskReddit is the place to ask and answer thought-provoking questions.",
    "lang": "en",
    "over18": false
  },
  "rules": [
    {
      "rule_index": 1,
      "short_name": "Rule 1 - Post must be an open-ended question",
      "description": "All posts must be open-ended questions...",
      "violation_reason": "Not an open-ended question",
      "short_name_clean": "Post must be an open-ended question",
      "description_clean": "All posts must be open-ended questions...",
      "violation_reason_clean": "Not an open-ended question",
      "rule_comprehensive": "Short Name: Post must be an open-ended question\nDescription: All posts must be open-ended questions...\nViolation Reason: Not an open-ended question"
    }
  ]
}
```

**Stage 2 Summary Output**:
```json
{
  "summary": {
    "total_sfw_subreddits": 2000,
    "date_processed": "2025-01-09",
    "top_n_target": 2000
  },
  "subreddits": [
    { /* subreddit entry */ },
    { /* subreddit entry */ }
  ]
}
```

---

## Phase 1 Summary

**Total Stages**: 3
**Key Outputs**:
- RC/RS archives (~416 files)
- Mod comment extracts (~208 files)
- Subreddit rankings (JSON)
- Top N SFW subreddits with rules (JSON)

**Next Phase**: Phase 2 filters mod comments for target subreddits and matches them to rules using embeddings.
