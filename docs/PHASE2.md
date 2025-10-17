# Phase 2: Rule Matching & Filtering

**Stages**: 3-4
**Purpose**: Match moderator comments to community rules using embeddings and rank subreddits by distribution quality

---

## Stage 3: Filter and Consolidate

**Script**: `scripts/3_filter_and_consolidate.py`

**Purpose**: Filter mod comments for target subreddits and consolidate chronologically into per-subreddit files.

**Process Flow** (3-phase):
1. **Phase 1 - Filter**: Stream each RC file → filter for target subreddits → write to temp subdirectories
2. **Phase 2 - Consolidate**: For each subreddit → read temp files chronologically → consolidate to final file
3. **Phase 3 - Cleanup**: Remove temp directories

### Functions

#### `load_target_subreddits(logger: Logger) -> Set[str]`
Load target subreddit names from Stage 2 output.

**Parameters**:
- `logger`: Logger instance

**Returns**: Set of normalized subreddit names

**Description**: Reads `stage2_top_N_sfw_subreddits.json`, extracts subreddit names, normalizes to lowercase.

---

#### `process_single_file(args: Tuple[str, Set[str]]) -> Dict[str, Any]`
Process a single RC mod comments file (Phase 1 - Filter).

**Parameters**:
- `args`: Tuple of (file_path, target_subreddits) for parallel processing

**Returns**: Dictionary with keys:
- `file`: str (filename)
- `lines_processed`: int
- `lines_matched`: int
- `subreddits_with_data`: int (number of subreddits that got data)
- `processing_time`: float
- `success`: bool

**Description**: Streams RC mod comments file, filters for target subreddits, writes to temp subdirectories using multi-output pattern. Each target subreddit gets its own temp subdirectory. Uses `process_zst_file_multi` with lazy-open writers.

---

#### `consolidate_subreddit(args: Tuple[str, Set[str]]) -> Dict[str, Any]`
Consolidate temp files for a single subreddit (Phase 2 - Consolidate).

**Parameters**:
- `args`: Tuple of (subreddit_name, target_subreddits) for parallel processing

**Returns**: Dictionary with keys:
- `subreddit`: str
- `total_comments`: int
- `temp_files_processed`: int
- `output_file`: str
- `file_size_gb`: float
- `processing_time`: float
- `success`: bool

**Description**: Reads all temp files for a subreddit in chronological order, consolidates into single output file `{subreddit}_mod_comments.jsonl.zst`. Maintains chronological order across all RC files.

---

#### `cleanup_temp_files(logger: Logger) -> None`
Remove temporary directories (Phase 3 - Cleanup).

**Parameters**:
- `logger`: Logger instance

**Returns**: None

**Description**: Recursively removes temp directory created in Phase 1. Logs cleanup statistics.

---

#### `main() -> int`
Main execution function for Stage 3.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates 3-phase processing: loads target subreddits, processes RC files in parallel (Phase 1), consolidates per subreddit in parallel (Phase 2), cleans up temp files (Phase 3). Writes summary JSON with statistics.

---

### Input/Output

**Inputs**:
- `data/mod_comments/RC_*_mod_comments.zst` from Stage 1
- `data/stage2_top_N_sfw_subreddits.json` from Stage 2

**Outputs**:
```
data/
├── top_subreddits/
│   ├── {subreddit}_mod_comments.jsonl.zst (one per subreddit)
│   └── temp/ (removed after consolidation)
│       └── {subreddit}/
│           └── RC_YYYY-MM_mod_comments.zst
└── stage3_filter_and_consolidate_summary.json
```

**Key Output**: Per-subreddit mod comment files in `top_subreddits/`

---

### Example Data Structures

**Filter Result (Phase 1)**:
```python
{
    'file': 'RC_2023-02_mod_comments.zst',
    'lines_processed': 5000000,
    'lines_matched': 125000,
    'subreddits_with_data': 1850,
    'processing_time': 245.3,
    'success': True
}
```

**Consolidation Result (Phase 2)**:
```python
{
    'subreddit': 'askreddit',
    'total_comments': 45000,
    'temp_files_processed': 208,
    'output_file': '/path/to/top_subreddits/askreddit_mod_comments.jsonl.zst',
    'file_size_gb': 0.125,
    'processing_time': 15.7,
    'success': True
}
```

**Summary Output**:
```json
{
  "summary": {
    "total_target_subreddits": 2000,
    "subreddits_with_data": 1850,
    "total_comments_filtered": 8500000,
    "processing_time_seconds": 3245.6,
    "collection_date": "2025-01-09 14:30:22"
  },
  "phase1_filter_stats": {
    "rc_files_processed": 208,
    "total_lines_processed": 125000000,
    "total_lines_matched": 8500000
  },
  "phase2_consolidate_stats": {
    "subreddits_consolidated": 1850,
    "total_output_size_gb": 12.5
  },
  "subreddit_stats": [
    {
      "subreddit": "askreddit",
      "total_comments": 45000,
      "file_size_gb": 0.125
    }
  ]
}
```

---

## Stage 4: Match Rules

**Script**: `scripts/4_match_rules.py`

**Purpose**: Match moderator comments to community rules using embeddings with global threshold application and JSD-based ranking.

**Process Flow** (2-phase):
1. **Phase 1 - Similarity Matrices**: Parallel subprocess per subreddit → compute embeddings → save `.pt` files
2. **Phase 2 - Global Matching**: Load all matrices → compute global thresholds → apply thresholds → match comments → rank by JSD

### Main Script Functions

#### `extract_submission_ids(comments: List[Dict[str, Any]]) -> List[str]`
Extract unique submission IDs from matched comments.

**Parameters**:
- `comments`: List of comment dictionaries

**Returns**: List of unique submission IDs

**Description**: Extracts `link_id` from each comment, removes `t3_` prefix, returns unique set as list.

---

#### `is_cuda_memory_available(device_id: int, threshold: float = 0.85) -> bool`
Check if CUDA device has available memory.

**Parameters**:
- `device_id`: CUDA device ID to check
- `threshold`: Memory usage threshold (0.0-1.0)

**Returns**: True if memory usage below threshold

**Description**: Uses `nvidia-smi` to query GPU memory. Returns True if available memory > (1 - threshold) * total memory.

---

#### `get_available_cuda_devices() -> List[int]`
Get list of available CUDA devices with sufficient memory.

**Parameters**: None

**Returns**: List of available CUDA device IDs

**Description**: Checks all CUDA devices, returns those with <85% memory usage. Used for dynamic device assignment in worker pool.

---

#### `create_distribution_plot(output_dir: str, all_similarities: np.ndarray, gold_percentile: int, ambiguous_percentile: int, logger: Logger = None) -> Tuple[float, float]`
Create similarity distribution plot with threshold lines.

**Parameters**:
- `output_dir`: Directory to save plot
- `all_similarities`: Numpy array of all similarity scores (flattened)
- `gold_percentile`: Percentile for gold threshold (e.g., 99)
- `ambiguous_percentile`: Percentile for ambiguous threshold (e.g., 95)
- `logger`: Optional logger instance

**Returns**: Tuple of (gold_threshold, ambiguous_threshold)

**Description**: Creates histogram of cosine similarity distribution, draws vertical lines for thresholds, saves as PNG. Calculates and returns threshold values.

---

#### `load_similarity_matrix(matrix_file: str) -> Dict[str, Any]`
Load similarity matrix from .pt file.

**Parameters**:
- `matrix_file`: Path to .pt file

**Returns**: Dictionary with keys:
- `subreddit`: str
- `matrix_file`: str
- `cosine_similarity_matrix`: torch.Tensor
- `comment_mapping`: Dict[str, int]
- `rule_indices`: List[int]
- `num_comments`: int
- `num_rules`: int
- `success`: bool

**Description**: Loads torch .pt file containing similarity matrix and metadata. Returns loaded data or error dictionary.

---

#### `process_subreddit_matching(load_result: Dict, logger: Logger = None) -> Dict[str, Any]`
Process a single subreddit's similarity matrix to create matches (Phase 2).

**Parameters**:
- `load_result`: Dictionary from `load_similarity_matrix`
- `logger`: Optional logger instance

**Returns**: Dictionary with keys:
- `subreddit`: str
- `total_comments`: int
- `matched_comments`: int
- `gold_matches`: int
- `ambiguous_rejected`: int
- `rule_matches`: Dict[str, int] (rule -> count)
- `jsd_from_uniform`: float
- `submission_ids`: List[str]
- `output_file`: str
- `stats_file`: str
- `success`: bool

**Description**: Applies global thresholds to similarity matrix, matches comments to best rule above gold threshold, rejects ambiguous matches, samples up to MAX_MATCHED_COMMENTS, calculates JSD, extracts submission IDs. Writes matched comments to .jsonl.zst and stats to JSON.

---

#### `main() -> int`
Main execution function for Stage 4.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Orchestrates 2-phase matching:
- **Phase 1**: Spawns isolated subprocesses (via `4_match_rules_single.py`) with dynamic CUDA device assignment via queue
- **Phase 2**: Loads all similarity matrices, computes global thresholds, creates distribution plot, processes each subreddit in parallel, ranks by JSD, writes summary

Supports `--phase2-only` CLI flag to skip Phase 1.

---

### Subprocess Script (4_match_rules_single.py)

**Purpose**: Isolated subprocess for computing similarity matrix for a single subreddit.

#### Class: `SimpleCommentRuleMatcher`

#### `__init__(self, model_name: str = None, max_model_len: int = 2048)`
Initialize matcher with vLLM model.

**Parameters**:
- `model_name`: Embedding model name (default from config)
- `max_model_len`: Maximum sequence length

**Description**: Initializes vLLM LLM instance for embedding generation. Sets deterministic seed.

---

#### `save_similarity_matrix(cosine_similarities: torch.Tensor, comments: List, rules: List, subreddit_name: str) -> None`
Save similarity matrix to .pt file.

**Parameters**:
- `cosine_similarities`: Torch tensor of shape (num_comments, num_rules)
- `comments`: List of comment dictionaries
- `rules`: List of rule dictionaries
- `subreddit_name`: Subreddit name for filename

**Description**: Creates comment_id -> row_index mapping, extracts rule indices, saves to `{subreddit}_similarity_matrix.pt` with metadata.

---

#### `calculate_similarities_pretokenized(self, comments: List, rules: List, tokenized_comments: List, tokenized_rules: List) -> bool`
Calculate cosine similarities using pretokenized inputs.

**Parameters**:
- `comments`: List of comment dictionaries
- `rules`: List of rule dictionaries
- `tokenized_comments`: Pretokenized comment texts
- `tokenized_rules`: Pretokenized rule texts

**Returns**: True on success, False on failure

**Description**: Generates embeddings using vLLM, computes cosine similarity matrix, saves to .pt file. Uses batch processing for efficiency.

---

#### `pretokenize_inputs(cls, comments: List, rules: List, model_name: str, task_description: str) -> Tuple[List, List, int]`
Pretokenize comments and rules to optimize max_model_len.

**Parameters**:
- `comments`: List of comment dictionaries
- `rules`: List of rule dictionaries
- `model_name`: Tokenizer model name
- `task_description`: Task prefix for embeddings

**Returns**: Tuple of (tokenized_comments, tokenized_rules, max_length_found)

**Description**: Uses transformers AutoTokenizer to pretokenize all inputs, finds maximum token length, enables optimal `max_model_len` configuration.

---

#### `main() -> int`
Main execution for subprocess.

**Returns**: Exit code (0 = success, 1 = failure)

**Description**: Parses CLI arguments (subreddit, cuda_device), loads comments and rules, pretokenizes, initializes matcher, computes similarities, saves matrix. Redirects vLLM logs to subreddit-specific log file. Explicit cleanup to prevent SIGABRT.

**Environment Variables Set**:
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `VLLM_CONFIGURE_LOGGING=0`
- `PYTHONHASHSEED=0`

---

### Input/Output

**Inputs**:
- `data/top_subreddits/{subreddit}_mod_comments.jsonl.zst` from Stage 3
- `data/stage2_top_N_sfw_subreddits.json` from Stage 2 (for rules)

**Outputs**:
```
output/
├── matched_comments/
│   ├── {subreddit}_similarity_matrix.pt (Phase 1)
│   ├── {subreddit}_match.jsonl.zst (Phase 2)
│   ├── {subreddit}_stats.json (Phase 2)
│   └── cosine_similarity_distribution_all_percentiles.png (Phase 2)
└── data/
    ├── stage4_matching_summary.json
    └── stage4_subreddit_submission_ids.json
```

---

### Example Data Structures

**Similarity Matrix (.pt file)**:
```python
{
    'cosine_similarity_matrix': torch.Tensor,  # Shape: (45000, 8)
    'comment_mapping': {
        'abc123': 0,
        'def456': 1,
        # ... comment_id -> row_index
    },
    'rule_indices': [1, 2, 3, 4, 5, 6, 7, 8],  # rule_index for each column
    'subreddit': 'askreddit',
    'num_comments': 45000,
    'num_rules': 8
}
```

**Matched Comment**:
```json
{
  "id": "abc123",
  "author": "ModeratorName",
  "body": "Your post has been removed for violating Rule 1.",
  "subreddit": "askreddit",
  "created_utc": 1640000000,
  "parent_id": "t1_xyz789",
  "link_id": "t3_sub123",
  "distinguished": "moderator",
  "score": 5,
  "matched_rule": {
    "rule_index": 1,
    "short_name": "Post must be an open-ended question",
    "description": "All posts must be open-ended questions...",
    "similarity_score": 0.8765
  }
}
```

**Subreddit Stats**:
```json
{
  "subreddit": "askreddit",
  "total_comments": 45000,
  "matched_comments": 1000,
  "match_percentage": 2.22,
  "gold_matches": 1200,
  "ambiguous_rejected": 200,
  "rule_matches": {
    "Post must be an open-ended question": 450,
    "No personal information": 250,
    "Be respectful": 300
  },
  "jsd_from_uniform": 0.0234,
  "gold_threshold": 0.7234,
  "ambiguous_threshold": 0.6891,
  "submission_ids": ["sub123", "sub456", "..."]
}
```

**Stage 4 Summary**:
```json
{
  "summary": {
    "total_subreddits_processed": 1800,
    "total_matched": 1500000,
    "gold_threshold": 0.7234,
    "ambiguous_threshold": 0.6891,
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "date_range": ["2005-12", "2023-02"],
    "processing_time_seconds": 45000,
    "collection_date": "2025-01-09 14:30:22"
  },
  "subreddit_stats": [
    {
      "rank": 1,
      "subreddit": "askreddit",
      "jsd_from_uniform": 0.0234,
      "matched_comments": 1000,
      "total_rules": 8,
      "total_submission_ids": 850
    }
  ]
}
```

**Submission IDs Output**:
```json
{
  "metadata": {
    "total_subreddits": 1800,
    "total_unique_submission_ids": 1250000,
    "creation_date": "2025-01-09 14:30:22"
  },
  "subreddit_submission_ids": {
    "askreddit": ["sub123", "sub456", "..."],
    "pics": ["sub789", "..."]
  }
}
```

---

## Phase 2 Summary

**Total Stages**: 2
**Key Outputs**:
- Per-subreddit mod comments (~1800 files)
- Similarity matrices (.pt files)
- Matched comments with rule assignments
- JSD-ranked subreddits
- Submission IDs for thread collection

**Key Algorithms**:
- **Embedding-based Matching**: Qwen3-Embedding-8B for semantic similarity
- **Global Thresholds**: 99th percentile (gold), 95th percentile (ambiguous)
- **JSD Ranking**: Jensen-Shannon Divergence from uniform distribution (0 = uniform, 1 = maximum divergence)

**Next Phase**: Phase 3 collects all comments for target submissions and builds discussion thread pairs.
