"""
Simple configuration for Reddit mod collection pipeline.

Edit the base directories below for your environment.
All other paths are generated automatically based on data flow.
"""

import os
import multiprocessing

# =============================================================================
# BASE CONFIGURATION - Edit these for your environment
# =============================================================================

# Base directories
BASE_DATA = "/N/scratch/zkachwal/data-reddit-mod-collection-pipeline"
REDDIT_DATA = "/N/project/cnets/datasets/reddit/Pushshift"
ARCTIC_SHIFT_DATA = "/N/project/cnets/datasets/reddit/Arcticshift/Subreddits/subreddits"

# Processing settings
DATE_RANGE = ("2005-12", "2023-02")  # (start, end) inclusive PushshiftDumps
MIN_RULES_FOR_MATCHING = 2  # Minimum rules needed for semantic matching (skip subreddits with ≤1 rule)
GOLD_PERCENTILE = 99  # Top 2% of similarity scores considered gold matches (Stage 4 Phase 2)
AMBIGUOUS_PERCENTILE = 98  # Top 2% of similarity scores considered ambiguous matches (Stage 4 Phase 2)
MIN_MATCHED_COMMENTS = 1 # Minimum matched comments for subreddit inclusion in Stage 4
MAX_MATCHED_COMMENTS = 500  # Max sample size for matched comments in Stage 4

# Stage 9: Dataset split configuration
MIN_TEST_THREAD_PAIRS = 25  # Minimum pairs required to include a subreddit (also used in Stage 6 filtering)
TEST_PAIRS_PER_SUBREDDIT = 25  # First 25 pairs from each subreddit go to test
VAL_SPLIT_RATIO = 0.1  # 10% of remaining pairs (after test) go to val
TRAIN_SPLIT_RATIO = 0.9  # 90% of remaining pairs (after test) go to train

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"  # Model used in Stage 4 for semantic matching
# Auto-detect number of CPU cores (use all available cores)
PROCESSES = multiprocessing.cpu_count()

# Alternative: Use 75% of available cores to leave some for system
# PROCESSES = max(1, int(multiprocessing.cpu_count() * 0.75))

# =============================================================================
# DATA FLOW MAPPING - Shows what each stage produces and consumes
# =============================================================================

DATA_FLOW = {
    # Phase 1: Data Collection
    'stage0_download_data': {
        'name': 'Download Reddit Data from Internet Archive',
        'script': '0_download_data.py',
        'input_paths': [],  # No inputs - downloads from internet
        'output_dir': 'reddit_data',
        'produces': [
            'comments/YYYY/RC_*.zst',  # Reddit comment files organized by year
            'submissions/YYYY/RS_*.zst',  # Reddit submission files organized by year
            'stage0_download_log.json'
        ]
    },

    'stage1_mod_comments': {
        'name': 'Collect Moderator Comments from Arctic Shift',
        'script': '1_collect_mod_comments.py',
        'input_paths': [],  # Uses Arctic Shift data directly
        'output_dir': 'top_subreddits',
        'produces': [
            '{subreddit}_mod_comments.jsonl.zst',  # One per subreddit
            'stage1_subreddit_mod_comment_rankings.json'
        ],
        'notes': 'Reads Arctic Shift subreddit files, filters mod comments. Replaces old Stage 1 + Stage 3.'
    },

    'stage2_top_sfw': {
        'name': 'Get SFW Subreddits with Minimum Mod Comments',
        'script': '2_get_top_sfw_subreddits.py',
        'input_files': ['stage1_subreddit_mod_comment_rankings.json'],
        'output_dir': 'data',
        'produces': ['stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'],
        'notes': 'Uses Reddit API to filter NSFW, collect subreddit metadata and rules'
    },

    # Phase 2: Comment Matching
    # NOTE: Stage 3 (filter_and_consolidate) is now obsolete - Stage 1 directly outputs to top_subreddits/

    'stage4_match_rules': {
        'name': 'Match Comments to Rules (2-Phase: Similarity Matrices + Global Thresholds)',
        'script': '4_match_rules.py',
        'helper_scripts': ['4_match_rules_single.py'],
        'input_paths': ['top_subreddits'],
        'input_files': [
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'
        ],
        'output_dir': 'matched_comments',
        'produces': [
            '{subreddit}_match.jsonl.zst',
            '{subreddit}_stats.json',
            '{subreddit}_similarity_matrix.pt',
            'stage4_matching_summary.json',
            'stage4_subreddit_submission_ids.json',
            'cosine_similarity_distribution_all_percentiles.png'
        ],
        'notes': 'Phase 1: Create similarity matrices using vLLM embeddings. Phase 2: Apply global percentile thresholds for matching. Filters ambiguous matches, ranks by JSD.'
    },

    # Phase 3: Thread Construction
    'stage5_collect_submission_comments': {
        'name': 'Collect and Organize Submission Comments from Arctic Shift',
        'script': '5_collect_submission_comments.py',
        'input_paths': [],  # Uses Arctic Shift data directly
        'input_files': ['stage4_subreddit_submission_ids.json'],
        'output_dir': 'organized_comments',
        'produces': [
            '{subreddit}_submission_comments.pkl',
            'stage5_submission_comment_collection_stats.json'
        ],
        'notes': '2-pass per subreddit: filter with process_zst_file_multi → deduplicate with [removed]/[deleted] preservation'
    },

    'stage6_build_trees_and_threads': {
        'name': 'Build Comment Trees and Discussion Threads',
        'script': '6_build_trees_and_threads.py',
        'input_paths': ['organized_comments', 'matched_comments'],
        'input_files': ['stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json'],
        'output_dir': 'comment_trees',
        'alternate_output_dirs': ['discussion_threads'],  # Also outputs here
        'produces': [
            'comment_trees/{subreddit}_comment_trees.pkl',
            'discussion_threads/{subreddit}_discussion_threads.pkl',
            'stage6_trees_and_threads_summary.json'
        ],
        'notes': 'Builds trees (parent-child, depth levels), creates moderated/unmoderated pairs, requires 500+ pairs, ranks by JSD'
    },

    # Phase 4: Dataset Finalization
    'stage7_collect_submissions': {
        'name': 'Collect Submissions from Discussion Threads',
        'script': '7_collect_submissions.py',
        'input_paths': ['reddit_submissions', 'discussion_threads'],
        'input_files': ['stage6_trees_and_threads_summary.json'],
        'output_dir': 'submissions',
        'produces': [
            '{subreddit}_submissions.zst',
            'stage7_submission_collection_stats.json'
        ],
        'notes': '3-phase: extract IDs from threads → process RS files → consolidate by subreddit'
    },

    'stage8_collect_media': {
        'name': 'Collect Media for Submissions',
        'script': '8_collect_media.py',
        'input_paths': ['submissions'],
        'input_files': ['stage6_trees_and_threads_summary.json'],
        'output_dir': 'media',
        'produces': [
            '{subreddit}/{submission_id}_{media_id}_{source}.{ext}',  # Downloaded media files
            'stage8_media_collection_stats.json',
            'stage8_successful_submission_ids.json'
        ],
        'notes': 'Priority: media_metadata → url → oembed → preview. Skips NSFW/crosspost/URL-only selfposts. Validates file types.'
    },

    'stage9_create_datasets': {
        'name': 'Create Final Datasets (Hydrated + Unhydrated)',
        'script': '9_create_unhydrated_dataset.py',
        'input_paths': ['discussion_threads', 'comment_trees', 'submissions', 'media'],
        'input_files': [
            'stage2_sfw_subreddits_min_{MIN_MATCHED_COMMENTS}_comments.json',
            'stage8_successful_submission_ids.json',
            'stage1_subreddit_mod_comment_rankings.json',
            'stage4_matching_summary.json',
            'stage6_trees_and_threads_summary.json',
            'stage7_submission_collection_stats.json',
            'stage8_media_collection_stats.json'
        ],
        'output_dir': 'data',
        'produces': [
            'reddit_moderation_dataset_hydrated_v1.0.json.zst',
            'reddit_moderation_dataset_unhydrated_v1.0.json.zst',
            'reddit_moderation_dataset_hydrated_SAMPLE.json',
            'reddit_moderation_dataset_unhydrated_SAMPLE.json',
            'stage9_final_datasets_stats.json'
        ],
        'notes': 'Samples exactly 500 pairs per subreddit, ranks ALL by JSD. Hydrated: full objects. Unhydrated: IDs with [NEEDS_HYDRATION] placeholders.'
    },

    'stage10_human_evaluation': {
        'name': 'Human Evaluation Setup (Google Forms)',
        'script': '10_human_evaluation.py',
        'input_files': ['reddit_moderation_dataset_hydrated_v1.0.json.zst'],
        'output_dir': 'data/evaluation',
        'produces': [
            'stage9.5_human_evaluation_part*.json',
            'stage9.5_human_evaluation_summary.json'
        ],
        'notes': 'Samples 4 comments per subreddit, creates Google Forms with 20 subreddits each. Requires OAuth2 credentials.'
    }
}

# =============================================================================
# AUTO-GENERATED PATHS - Don't edit these
# =============================================================================

def _generate_paths():
    """Generate all paths based on base directories and data flow."""
    paths = {
        # Input data sources
        'reddit_comments': f"{REDDIT_DATA}/comments",
        'reddit_submissions': f"{REDDIT_DATA}/submissions",
        'reddit_data': f"{REDDIT_DATA}",  # Base directory for downloaded data

        # Base output directories
        'data': f"{BASE_DATA}/data",
        'logs': f"{BASE_DATA}/logs",

        # Stage output directories (auto-generated from DATA_FLOW)
        'mod_comments': f"{BASE_DATA}/data/mod_comments",
        'top_subreddits': f"{BASE_DATA}/data/top_subreddits",
        'matched_comments': f"{BASE_DATA}/output/matched_comments",
        'matched_comments_sample': f"{BASE_DATA}/output/matched_comments_sample",
        'submission_comments': f"{BASE_DATA}/data/submission_comments",
        'organized_comments': f"{BASE_DATA}/output/organized_comments",
        'comment_trees': f"{BASE_DATA}/output/comment_trees",
        'discussion_threads': f"{BASE_DATA}/output/discussion_threads",
        'submissions': f"{BASE_DATA}/data/submissions",
        'media': f"{BASE_DATA}/output/media",
        'final_dataset': f"{BASE_DATA}/output/final_dataset"
    }

    return paths

PATHS = _generate_paths()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_stage_info(stage_num):
    """Get information for a specific stage number (0-13)."""
    stage_key = f"stage{stage_num}_" + list(DATA_FLOW.keys())[stage_num].split('_', 1)[1]
    return DATA_FLOW.get(stage_key)

def get_input_paths_for_stage(stage_num):
    """Get resolved input paths for a stage."""
    stage_info = get_stage_info(stage_num)
    if not stage_info:
        return []

    input_paths = []

    # Add directory paths
    for path_key in stage_info.get('input_paths', []):
        input_paths.append(PATHS[path_key])

    # Add specific files
    for file_name in stage_info.get('input_files', []):
        # Substitute template variables
        resolved_file_name = file_name.format(
            MIN_MATCHED_COMMENTS=MIN_MATCHED_COMMENTS
        )
        input_paths.append(os.path.join(PATHS['data'], resolved_file_name))

    return input_paths

def get_output_path_for_stage(stage_num):
    """Get resolved output path for a stage."""
    stage_info = get_stage_info(stage_num)
    if not stage_info:
        return None

    output_dir = stage_info.get('output_dir')
    return PATHS.get(output_dir)

def create_directories():
    """Create necessary output directories (excludes read-only input paths)."""
    # Skip input directories that should already exist
    skip_paths = {'reddit_comments', 'reddit_submissions', 'reddit_data'}

    for name, path in PATHS.items():
        if name not in skip_paths:
            os.makedirs(path, exist_ok=True)

def validate_stage_inputs(stage_num):
    """Check if inputs exist for a stage."""
    input_paths = get_input_paths_for_stage(stage_num)

    for path in input_paths:
        if os.path.isfile(path):
            if not os.path.exists(path):
                return False, f"Missing file: {path}"
        elif os.path.isdir(path):
            if not os.path.exists(path) or not os.listdir(path):
                return False, f"Missing or empty directory: {path}"
        else:
            return False, f"Path doesn't exist: {path}"

    return True, "All inputs available"

def print_pipeline_status():
    """Print status of entire pipeline."""
    print("Reddit Mod Collection Pipeline Status")
    print("=" * 80)
    print()

    for i in range(0, 11):  # Now 0-10 stages (including stage 10)
        stage_info = get_stage_info(i)
        if stage_info:
            valid, msg = validate_stage_inputs(i)
            output_path = get_output_path_for_stage(i)
            output_exists = os.path.exists(output_path) if output_path else False

            status = "✓" if valid else "✗"
            output_status = "✓" if output_exists else "✗"

            print(f"Stage {i:2d}: {stage_info['name']}")
            print(f"         Script: {stage_info.get('script', 'N/A')}")
            print(f"         Input: {status} | Output: {output_status}")
            if not valid:
                print(f"         Issue: {msg}")
            if stage_info.get('notes'):
                print(f"         Notes: {stage_info['notes']}")
            print()

    print("=" * 80)

if __name__ == "__main__":
    # When run directly, show pipeline status
    create_directories()
    print_pipeline_status()