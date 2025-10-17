# Phase 4: Finalization & Evaluation

**Stages**: 7-10
**Purpose**: Enrich dataset with submissions and media, create final datasets, setup human evaluation

---

## Stage 7: Collect Submissions

**Script**: `scripts/7_collect_submissions.py`

**Purpose**: Collect submission objects for all submissions referenced in discussion threads.

**Process Flow** (3-phase):
1. **Phase 1 - Extract IDs**: Extract submission IDs from discussion threads
2. **Phase 2 - Process RS Files**: Stream RS files → filter for target submissions → write to temp subdirectories
3. **Phase 3 - Consolidate**: For each subreddit → consolidate temp files → save as .zst

### Functions

#### `extract_submission_ids_from_threads(subreddit: str, logger: Logger) -> Set[str]`
Extract unique submission IDs from a subreddit's discussion threads.

**Parameters**:
- `subreddit`: Subreddit name
- `logger`: Logger instance

**Returns**: Set of unique submission IDs

**Description**: Loads `{subreddit}_discussion_threads.pkl`, extracts submission_id from each thread pair's metadata.

---

#### `collect_subreddit_submission_ids(logger: Logger) -> Dict[str, Set[str]]`
Collect submission IDs for all qualified subreddits from Stage 6.

**Parameters**:
- `logger`: Logger instance

**Returns**: Dictionary mapping subreddit -> set of submission IDs

**Description**: Uses `load_qualified_subreddits_from_stage6` helper, extracts submission IDs for each qualified subreddit.

---

#### `process_rs_file(args: Tuple) -> Dict[str, Any]`
Process single RS file to collect submissions (Phase 2).

**Parameters**:
- `args`: Tuple of (rs_file_path, subreddit_submission_ids, temp_dir) for parallel processing

**Returns**: Dictionary with keys:
- `rs_file`: str (filename)
- `total_submissions`: int
- `matched_submissions`: int
- `subreddits_with_data`: int
- `processing_time`: float
- `success`: bool

**Description**: Streams RS file, checks if submission is in target set, writes to temp subdirectory using multi-output pattern. Uses worker logger with RS file identifier.

---

#### `consolidate_subreddit_submissions(args: Tuple) -> Dict[str, Any]`
Consolidate temp files for a single subreddit (Phase 3).

**Parameters**:
- `args`: Tuple of (subreddit, temp_dir, needed_submission_ids) for parallel processing

**Returns**: Dictionary with keys:
- `subreddit`: str
- `submissions_needed`: int
- `submissions_collected`: int
- `coverage_rate`: float
- `output_file`: str
- `file_size_gb`: float
- `temp_files_processed`: int
- `processing_time`: float
- `success`: bool

**Description**: Reads all temp RS files for a subreddit in chronological order, consolidates into `{subreddit}_submissions.zst`, calculates coverage rate. Uses worker logger with subreddit identifier. Cleans up temp directory after consolidation.

---

#### `main() -> int`
Main execution function for Stage 7.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates 3-phase processing: extracts submission IDs from threads (Phase 1), processes RS files in parallel (Phase 2), consolidates per subreddit in parallel (Phase 3), writes summary JSON with coverage statistics.

---

### Input/Output

**Inputs**:
- `output/discussion_threads/{subreddit}_discussion_threads.pkl` from Stage 6
- `reddit_submissions/RS_*.zst` from Stage 0
- `data/stage6_trees_and_threads_summary.json` from Stage 6 (for qualified subreddits)

**Outputs**:
```
data/
├── submissions/
│   ├── {subreddit}_submissions.zst
│   └── temp/ (removed after consolidation)
│       └── {subreddit}/
│           └── RS_YYYY-MM.zst
└── stage7_submission_collection_stats.json
```

---

### Example Data Structures

**Submission Object**:
```python
{
    'id': 'sub123',
    'author': 'username',
    'title': 'Interesting question about...',
    'selftext': 'Full text of self post...',
    'url': 'https://reddit.com/r/AskReddit/...',
    'subreddit': 'AskReddit',
    'created_utc': 1640000000,
    'score': 1250,
    'num_comments': 345,
    'is_video': False,
    'is_self': True,
    'media_metadata': {...},
    'preview': {...}
}
```

**Summary Output**:
```json
{
  "summary": {
    "total_qualified_subreddits": 125,
    "subreddits_with_submissions": 123,
    "total_unique_submission_ids": 105000,
    "total_submissions_needed": 105000,
    "total_submissions_found": 98000,
    "overall_coverage_rate": 0.933,
    "total_rs_files_processed": 208,
    "total_submissions_processed": 250000000,
    "total_submissions_collected": 98000,
    "collection_rate": 0.000392,
    "total_output_size_gb": 15.5,
    "processing_time_seconds": 4500,
    "collection_date": "2025-01-09 14:30:22"
  },
  "subreddit_stats": [
    {
      "subreddit": "askreddit",
      "submissions_needed": 850,
      "submissions_collected": 845,
      "coverage_rate": 0.994,
      "file_size_gb": 0.125
    }
  ]
}
```

---

## Stage 8: Collect Media

**Script**: `scripts/8_collect_media.py`

**Purpose**: Download media files for submissions using priority-based collection strategy.

**Process Flow**:
1. Load qualified subreddits from Stage 6
2. For each subreddit:
   - Load submissions
   - For each submission:
     - Check NSFW/crosspost/URL-only selfpost (skip)
     - Extract media URLs using priority hierarchy
     - Download and validate files
3. Track detailed statistics
4. Save successful submission IDs for Stage 9

### Media Priority Hierarchy (Early Stopping)
1. **media_metadata** - Gallery/inline images (original source, 1-N items)
2. **url field** - Direct image posts (original source, 1 item)
3. **oembed** - Video thumbnails from YouTube/Vimeo (original source, 1 item)
4. **preview** - Reddit's cached images (fallback, 1 item)

### Functions

#### `with_percentages(counts: Dict[str, int], total: int) -> Dict[str, Dict]`
Add percentages to count dictionary.

**Parameters**:
- `counts`: Dictionary of category -> count
- `total`: Total count for percentage calculation

**Returns**: Dictionary with count and percentage for each category

**Description**: Helper for statistics formatting.

---

#### `count_by_field(items: List[Dict], field: str, filter_status: str = None) -> Dict[str, int]`
Count occurrences of field values.

**Parameters**:
- `items`: List of result dictionaries
- `field`: Field name to count
- `filter_status`: Optional status filter (e.g., 'complete')

**Returns**: Dictionary of field_value -> count

**Description**: Helper for statistics aggregation.

---

#### `count_where(items: List[Dict], field: str, value: Any, filter_status: str = None) -> int`
Count items where field equals value.

**Parameters**:
- `items`: List of result dictionaries
- `field`: Field name to check
- `value`: Value to match
- `filter_status`: Optional status filter

**Returns**: Count of matching items

**Description**: Helper for statistics aggregation.

---

#### `aggregate_nested(results: List[Dict], key: str) -> Dict[str, int]`
Aggregate counts from nested dictionaries.

**Parameters**:
- `results`: List of result dictionaries
- `key`: Key containing nested counts

**Returns**: Aggregated dictionary

**Description**: Helper for combining error counts from multiple subreddits.

---

#### `categorize_error(error_msg: str) -> str`
Categorize error message into standard types.

**Parameters**:
- `error_msg`: Raw error message

**Returns**: Categorized error type (e.g., '404 Not Found', 'Timeout')

**Description**: Standardizes error messages for statistics.

---

#### `create_session() -> requests.Session`
Create robust requests session with retry logic.

**Parameters**: None

**Returns**: Configured requests session

**Description**: Creates session with retry strategy (3 attempts, exponential backoff) for status codes 429, 500-504.

---

#### `download_file(url: str, output_path: str, session: requests.Session) -> Dict[str, Any]`
Download and validate file.

**Parameters**:
- `url`: Download URL
- `output_path`: Local file path
- `session`: Requests session

**Returns**: Dictionary with keys:
- `success`: bool
- `file_size`: int (if success)
- `error`: str (if failure)

**Description**: Downloads file with streaming, validates Content-Type, validates with `file` command, removes invalid files.

---

#### `is_video_submission(submission: Dict) -> bool`
Check if submission is a video submission.

**Parameters**:
- `submission`: Submission dictionary

**Returns**: True if video submission

**Description**: Checks `is_video` flag, URL domain (v.redd.it, youtube.com, etc.), media_metadata for video types.

---

#### `extract_media_metadata_urls(submission: Dict) -> List[Dict]`
Extract URLs from media_metadata field (galleries).

**Parameters**:
- `submission`: Submission dictionary

**Returns**: List of URL dictionaries with keys: url, media_id, source, extension, index

**Description**: Priority 1 - Extracts gallery images and animated images from media_metadata.

---

#### `extract_url_field(submission: Dict) -> List[Dict]`
Extract URL from direct url field.

**Parameters**:
- `submission`: Submission dictionary

**Returns**: List with single URL dictionary (or empty)

**Description**: Priority 2 - Extracts direct image URLs from url field. Skips video domains and reddit.com gallery links.

---

#### `extract_oembed_url(submission: Dict) -> List[Dict]`
Extract oembed thumbnail URL.

**Parameters**:
- `submission`: Submission dictionary

**Returns**: List with single URL dictionary (or empty)

**Description**: Priority 3 - Extracts video thumbnails from oembed data (YouTube, Vimeo, etc.).

---

#### `extract_preview_url(submission: Dict) -> List[Dict]`
Extract preview URL (fallback for link posts only).

**Parameters**:
- `submission`: Submission dictionary

**Returns**: List with single URL dictionary (or empty)

**Description**: Priority 4 - Extracts Reddit's cached preview images. Only for link posts (not self posts).

---

#### `extract_download_urls(submission: Dict) -> Tuple[List[Dict], str]`
Extract downloadable URLs using priority hierarchy.

**Parameters**:
- `submission`: Submission dictionary

**Returns**: Tuple of (urls, source)
- `urls`: List of URL dictionaries
- `source`: Source name ('media_metadata', 'url', 'oembed', 'preview', or None)

**Description**: Applies priority hierarchy with early stopping. Returns first non-empty result.

---

#### `download_submission_media(submission: Dict, media_dir: str, session: requests.Session) -> Dict[str, Any]`
Download all media for a single submission.

**Parameters**:
- `submission`: Submission dictionary
- `media_dir`: Directory to save media
- `session`: Requests session

**Returns**: Dictionary with keys:
- `submission_id`: str
- `status`: str ('complete', 'partial', 'failed', 'no_media', 'skipped_nsfw', 'skipped_crosspost', 'skipped_url_only_selfpost')
- `files_downloaded`: int
- `total_size`: int
- `source`: str (media source)
- `has_multiple`: bool
- `is_video`: bool
- `errors`: List[str]

**Description**: Skips NSFW, crossposts, URL-only self posts. Extracts URLs, downloads each, tracks success/failure. Status is 'complete' only if all downloads succeed.

---

#### `process_subreddit(args: Tuple) -> Dict[str, Any]`
Process all submissions for a subreddit.

**Parameters**:
- `args`: Tuple of (subreddit,) for parallel processing

**Returns**: Dictionary with keys:
- `subreddit`: str
- `results`: List[Dict] (download results for each submission)
- `total_files`: int
- `total_size`: int
- `successful_ids`: List[str] (submissions with status 'complete' or 'no_media')
- `error_reasons`: Dict[str, int]
- `processing_time`: float

**Description**: Loads submissions, downloads media for each, aggregates statistics. Uses worker logger with subreddit identifier.

---

#### `format_subreddit_stats(subreddit_result: Dict) -> Dict`
Format statistics for a single subreddit.

**Parameters**:
- `subreddit_result`: Result dictionary from `process_subreddit`

**Returns**: Formatted statistics dictionary

**Description**: Calculates status breakdown, media sources, characteristics with percentages.

---

#### `format_global_stats(valid_results: List[Dict]) -> Dict`
Format global statistics from all subreddits.

**Parameters**:
- `valid_results`: List of successful subreddit results

**Returns**: Global statistics dictionary

**Description**: Aggregates across all subreddits, calculates totals, percentages, top errors.

---

#### `main() -> int`
Main execution function for Stage 8.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Loads qualified subreddits from Stage 6, processes in parallel, aggregates statistics, saves summary JSON and successful submission IDs JSON for Stage 9 filtering.

---

### Input/Output

**Inputs**:
- `data/submissions/{subreddit}_submissions.zst` from Stage 7
- `data/stage6_trees_and_threads_summary.json` from Stage 6 (for qualified subreddits)

**Outputs**:
```
output/
├── media/
│   └── {subreddit}/
│       ├── {submission_id}_{source}.{ext}
│       └── {submission_id}_{index}_{media_id}_{source}.{ext}
└── data/
    ├── stage8_media_collection_stats.json
    └── stage8_successful_submission_ids.json
```

**Filename Patterns**:
- Single media: `{submission_id}_{source}.{ext}` (e.g., `sub123_url.jpg`)
- Gallery: `{submission_id}_{index}_{media_id}_{source}.{ext}` (e.g., `sub123_0_abc_media_metadata.jpg`)

---

### Example Data Structures

**Download Result**:
```python
{
    'submission_id': 'sub123',
    'status': 'complete',
    'files_downloaded': 3,
    'total_size': 2500000,
    'source': 'media_metadata',
    'has_multiple': True,
    'is_video': False,
    'errors': []
}
```

**Successful Submission IDs Output**:
```json
{
  "metadata": {
    "total_subreddits": 123,
    "total_successful_submissions": 85000,
    "creation_date": "2025-01-09 14:30:22",
    "criteria": "Submissions with status: complete or no_media ONLY"
  },
  "subreddit_submission_ids": {
    "askreddit": ["sub123", "sub456", "..."],
    "pics": ["sub789", "..."]
  }
}
```

**Summary Output**:
```json
{
  "summary": {
    "total_submissions": 98000,
    "total_files_downloaded": 45000,
    "total_size_bytes": 25000000000,
    "total_size_gb": 23.28,
    "status_breakdown": {
      "complete": {"count": 40000, "percentage": 40.82},
      "no_media": {"count": 45000, "percentage": 45.92},
      "partial": {"count": 2000, "percentage": 2.04},
      "failed": {"count": 1000, "percentage": 1.02},
      "skipped_nsfw": {"count": 8000, "percentage": 8.16},
      "skipped_crosspost": {"count": 2000, "percentage": 2.04}
    },
    "media_sources": {
      "media_metadata": {"count": 15000, "percentage": 37.5},
      "url": {"count": 20000, "percentage": 50.0},
      "oembed": {"count": 3000, "percentage": 7.5},
      "preview": {"count": 2000, "percentage": 5.0}
    },
    "error_reasons": {
      "404 Not Found": 500,
      "Connection Error": 300,
      "Timeout": 200
    }
  }
}
```

---

## Stage 9: Create Final Datasets

**Script**: `scripts/9_create_unhydrated_dataset.py`

**Purpose**: Create final hydrated (full objects) and unhydrated (IDs only) datasets.

**Process Flow**:
1. Load successful submissions from Stage 8
2. Load submission objects from Stage 7
3. Load subreddit rules from Stage 2
4. For each subreddit:
   - Filter thread pairs to successful submissions
   - Require ≥500 pairs (sample exactly 500 if more)
   - Build hydrated structure with full objects
   - Collect media file paths
5. Rank ALL subreddits by JSD (lower = better)
6. Create hydrated dataset
7. Strip to IDs for unhydrated dataset
8. Create sample datasets (top 5 subreddits)

### Functions

#### `load_successful_submissions(logger: Logger) -> Tuple[Dict, Dict]`
Load submission IDs from Stage 8 and submission objects from Stage 7.

**Parameters**:
- `logger`: Logger instance

**Returns**: Tuple of (subreddit_successful_ids, subreddit_submission_objects)
- `subreddit_successful_ids`: Dict[str, Set[str]]
- `subreddit_submission_objects`: Dict[str, Dict[str, Dict]]

**Description**: Loads `stage8_successful_submission_ids.json`, then loads corresponding submission objects from Stage 7 .zst files.

---

#### `load_subreddit_rules(logger: Logger) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]`
Load rules for all subreddits from Stage 2.

**Parameters**:
- `logger`: Logger instance

**Returns**: Tuple of (subreddit_rules, subreddit_languages)

**Description**: Loads `stage2_top_N_sfw_subreddits.json`, extracts rules and language for each subreddit.

---

#### `filter_thread_pairs(thread_pairs: List[Dict], successful_submission_ids: Set[str]) -> List[Dict]`
Filter thread pairs to only include those with successful Stage 8 submissions.

**Parameters**:
- `thread_pairs`: List of thread pair dictionaries
- `successful_submission_ids`: Set of successful submission IDs

**Returns**: Filtered list of thread pairs

**Description**: Checks each pair's submission_id against successful set.

---

#### `collect_media_files(subreddit: str, submission_id: str, logger: Logger) -> List[str]`
Collect media file paths for a submission.

**Parameters**:
- `subreddit`: Subreddit name
- `submission_id`: Submission ID
- `logger`: Logger instance

**Returns**: List of media file paths (relative: `{subreddit}/{filename}`)

**Description**: Lists files in media directory matching `{submission_id}_*` pattern, returns sorted paths.

---

#### `strip_to_ids(hydrated_dataset: Dict) -> Dict`
Strip hydrated dataset to IDs only for public release.

**Parameters**:
- `hydrated_dataset`: Full hydrated dataset

**Returns**: Unhydrated dataset with `[NEEDS_HYDRATION]` placeholders

**Description**: Replaces all text fields (comment objects, submission objects) with `[NEEDS_HYDRATION]` placeholder. Preserves:
- IDs
- Timestamps (created_utc fields)
- Rule metadata
- Media file count
- Tree structure (depths, counts)

---

#### `process_subreddit(subreddit: str, successful_submission_ids: Set[str], subreddit_rules: List[Dict], submission_objects: Dict[str, Dict], logger: Logger) -> Dict[str, Any]`
Process a single subreddit and extract hydrated data.

**Parameters**:
- `subreddit`: Subreddit name
- `successful_submission_ids`: Set of successful submission IDs
- `subreddit_rules`: List of rule dictionaries
- `submission_objects`: Dictionary of submission_id -> submission object
- `logger`: Logger instance

**Returns**: Subreddit data dictionary or None if insufficient pairs

**Description**:
- Loads discussion threads
- Filters to successful submissions
- Requires ≥500 pairs (returns None if insufficient)
- Samples exactly 500 pairs if more (seed=0)
- Loads comment trees
- Creates shuffled rule options using mod_comment_id as seed
- Collects media files
- Calculates JSD from rule distribution
- Returns complete hydrated structure

---

#### `main() -> int`
Main execution function for Stage 9.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates dataset creation: loads all inputs, processes each subreddit, filters for ≥500 pairs, ranks ALL by JSD (lower = better), creates hydrated dataset, strips to unhydrated, creates sample datasets (top 5), writes all outputs with compression.

---

### Input/Output

**Inputs**:
- `output/discussion_threads/{subreddit}_discussion_threads.pkl` from Stage 6
- `output/comment_trees/{subreddit}_comment_trees.pkl` from Stage 6
- `data/submissions/{subreddit}_submissions.zst` from Stage 7
- `output/media/{subreddit}/*` from Stage 8
- `data/stage8_successful_submission_ids.json` from Stage 8
- `data/stage2_top_N_sfw_subreddits.json` from Stage 2

**Outputs**:
```
data/
├── reddit_moderation_dataset_hydrated_v1.0.json.zst
├── reddit_moderation_dataset_unhydrated_v1.0.json.zst
├── reddit_moderation_dataset_hydrated_SAMPLE.json
├── reddit_moderation_dataset_unhydrated_SAMPLE.json
└── stage9_final_datasets_stats.json
```

---

### Example Data Structures

**Hydrated Thread Pair**:
```python
{
    'mod_comment_id': 'mod_abc123',
    'mod_comment': { /* full comment object */ },
    'mod_comment_date': 1640000500,
    'submission_id': 'sub123',
    'submission_date': 1640000000,
    'matched_rule': {
        'rule_index': 1,
        'short_name': 'Post must be an open-ended question',
        'description': 'All posts must be...',
        'similarity_score': 0.8765
    },
    'rule_options': [
        {'label': '(a)', 'rule': 'Post must be an open-ended question'},
        {'label': '(b)', 'rule': 'No personal information'},
        {'label': '(c)', 'rule': 'Be respectful'}
    ],
    'moderated_thread': [ /* full comment objects */ ],
    'unmoderated_thread': [ /* full comment objects */ ],
    'unmod_thread_metadata': {
        'common_ancestors': 2,
        'moderated_comment_depth': 2,
        'target_length': 3,
        'moderated_score': 5,
        'unmoderated_score': 8
    }
}
```

**Unhydrated Thread Pair**:
```python
{
    'mod_comment_id': 'mod_abc123',
    'mod_comment': '[NEEDS_HYDRATION]',
    'mod_comment_date': 1640000500,
    'submission_id': 'sub123',
    'submission_date': 1640000000,
    'matched_rule': { /* preserved */ },
    'rule_options': [ /* preserved */ ],
    'moderated_thread': [
        {
            'comment_id': 'comment_1',
            'comment_date': 1640000100,
            'level': 0,
            'comment_object': '[NEEDS_HYDRATION]'
        }
    ],
    'unmoderated_thread': [ /* similar */ ],
    'unmod_thread_metadata': { /* preserved */ }
}
```

**Final Dataset Metadata**:
```json
{
  "version": "1.0",
  "creation_date": "2025-01-09",
  "total_subreddits": 115,
  "total_thread_pairs": 57500,
  "total_submissions": 45000,
  "total_submissions_with_media": 18000,
  "total_media_files": 25000,
  "embedding_model": "Qwen/Qwen3-Embedding-8B",
  "gold_threshold": 0.7234,
  "ambiguous_threshold": 0.6891,
  "date_range": ["2005-12", "2023-02"]
}
```

---

## Stage 10: Human Evaluation

**Script**: `scripts/10_human_evaluation.py`

**Purpose**: Create Google Forms for human evaluation of rule matching quality.

**Process Flow**:
1. Load final hydrated dataset from Stage 9
2. Sample 4 moderator comments per subreddit (seed=0)
3. Prepare subreddit pages (sorted alphabetically)
4. Split into forms (20 subreddits per form)
5. Authenticate with Google APIs
6. Create forms with rules displayed at top of each page
7. Save evaluation metadata

### Functions

#### `authenticate() -> Credentials`
Authenticate with Google APIs using OAuth2.

**Parameters**: None

**Returns**: OAuth2 credentials

**Description**: Uses client secrets file and token file. Runs local server for OAuth flow if needed. Saves credentials for future runs.

---

#### `load_final_dataset(logger: Logger) -> Dict[str, Any]`
Load the final hydrated dataset from Stage 9.

**Parameters**:
- `logger`: Logger instance

**Returns**: Hydrated dataset dictionary

**Description**: Reads compressed `reddit_moderation_dataset_hydrated_v1.0.json.zst` using multi-threaded zstandard decompressor.

---

#### `sample_comments_from_subreddit(subreddit_data: Dict, num_samples: int) -> List[Dict]`
Sample moderator comments from a subreddit's thread pairs.

**Parameters**:
- `subreddit_data`: Subreddit data from dataset
- `num_samples`: Number of comments to sample

**Returns**: List of sampled comment dictionaries

**Description**: Uses `random.sample` to select thread pairs, extracts moderator comments with matched rule info.

---

#### `prepare_subreddit_pages(dataset: Dict) -> List[Dict]`
Prepare data for each subreddit page.

**Parameters**:
- `dataset`: Full dataset dictionary

**Returns**: List of page data dictionaries

**Description**: Samples comments for each subreddit, sorts alphabetically by subreddit name. Uses seed=0 for reproducibility.

---

#### `create_rules_display_text(rules: List[Dict]) -> str`
Create formatted text for displaying rules at top of page.

**Parameters**:
- `rules`: List of rule dictionaries

**Returns**: Formatted rules text

**Description**: Creates multi-line text with rule index, short name, description, violation reason.

---

#### `create_evaluation_form(service: googleapiclient.discovery.Resource, subreddit_pages: List[Dict], form_part: int = 1, total_forms: int = 1) -> str`
Create Google Form with subreddit pages.

**Parameters**:
- `service`: Google Forms API service
- `subreddit_pages`: List of subreddit page data
- `form_part`: Form number (if split)
- `total_forms`: Total number of forms

**Returns**: Form ID

**Description**:
1. Creates basic form
2. Adds description with total subreddits/questions
3. For each subreddit page:
   - Adds page break (except first) with rules display
   - Adds checkbox question for each comment
   - Options = rule short names + "None of the above" + "Other"
4. Executes batch update

---

#### `save_evaluation_metadata(all_forms_data: List[Dict], subreddit_pages: List[Dict], output_dir: str) -> Tuple[List[str], List[str]]`
Save evaluation metadata for analysis.

**Parameters**:
- `all_forms_data`: List of form data dictionaries
- `subreddit_pages`: List of subreddit page data
- `output_dir`: Output directory

**Returns**: Tuple of (all_form_ids, all_public_urls)

**Description**: Saves individual form metadata files, combined summary file with form URLs and subreddit metadata.

---

#### `main() -> int`
Main execution function for Stage 10.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Loads dataset, samples comments, splits into forms (20 subreddits each), authenticates with Google, creates forms, saves metadata. Prints public URLs for distribution.

---

### Input/Output

**Inputs**:
- `data/reddit_moderation_dataset_hydrated_v1.0.json.zst` from Stage 9
- Google OAuth credentials (environment)

**Outputs**:
```
data/
└── evaluation/
    ├── stage9.5_human_evaluation_part1.json
    ├── stage9.5_human_evaluation_part2.json
    ├── ...
    └── stage9.5_human_evaluation_summary.json
```

**External Outputs**:
- Google Forms (created via API)

---

### Example Data Structures

**Form Data**:
```json
{
  "form_id": "abc123xyz",
  "form_url": "https://docs.google.com/forms/d/abc123xyz/edit",
  "public_url": "https://docs.google.com/forms/d/abc123xyz/viewform",
  "form_part": 1,
  "total_forms": 6,
  "num_subreddits": 20,
  "num_questions": 80,
  "subreddits": ["askreddit", "pics", "gaming", "..."]
}
```

**Summary Output**:
```json
{
  "metadata": {
    "total_forms": 6,
    "total_subreddits": 115,
    "total_questions": 460,
    "comments_per_subreddit": 4,
    "subreddits_per_form": 20,
    "random_seed": 0,
    "creation_date": "2025-01-09 14:30:22"
  },
  "forms": [ /* form data */ ],
  "subreddit_pages": [
    {
      "subreddit": "askreddit",
      "rank": 1,
      "jsd_score": 0.0234,
      "num_rules": 8,
      "num_comments_sampled": 4
    }
  ]
}
```

---

## Phase 4 Summary

**Total Stages**: 4
**Key Outputs**:
- Submission objects (~123 subreddits)
- Media files (~25,000 files, ~23 GB)
- Final hydrated dataset (full objects)
- Final unhydrated dataset (IDs only with placeholders)
- Sample datasets (top 5 subreddits)
- Human evaluation forms

**Dataset Structure**:
- Exactly 500 thread pairs per subreddit
- All subreddits ranked by JSD (lower = better)
- English and non-English subreddits included
- Shuffled rule options for evaluation (using mod_comment_id as seed)
- Media files with source tracking

**Evaluation Setup**:
- 4 comments sampled per subreddit
- 20 subreddits per Google Form
- Alphabetically sorted
- Rules displayed at top of each page
- Checkbox selection with "None" and "Other" options

**Final Dataset Characteristics**:
- **Hydrated**: Full comment/submission objects for internal use
- **Unhydrated**: IDs only with `[NEEDS_HYDRATION]` placeholders for public release
- **Timestamps preserved**: All `*_date` fields preserved for chronological analysis
- **Media tracked**: File counts and paths (with placeholders in unhydrated)
- **Reproducible**: All random operations use seed=0

---

*This completes the 4-phase, 11-stage Reddit Mod Collection Pipeline documentation.*
