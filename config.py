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
BASE_DATA = "/data3/zkachwal/reddit-mod-collection-pipeline"
REDDIT_DATA = "/gpfs/slate-cnets/datasets/reddit/Pushshift"

# Processing settings
DATE_RANGE = ("2005-12", "2023-02")  # (start, end) inclusive PushshiftDumps
TOP_N_SUBREDDITS_WITH_MOD_COMMENTS = 10000
GOLD_PERCENTILE = 99  # Top 2% of similarity scores considered gold matches
AMBIGUOUS_PERCENTILE = 95  # Top 5% of similarity scores considered ambiguous matches
MIN_MATCHED_COMMENTS = FINAL_THREAD_PAIRS_PER_SUBREDDIT = 50
MAX_MATCHED_COMMENTS = 100
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
FINAL_SUBREDDITS = 100
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
        'produces': ['RC_*.zst', 'RS_*.zst']  # Reddit comment and submission files
    },

    'stage1_mod_comments': {
        'name': 'Collect Moderator Comments',
        'script': '1_collect_mod_comments.py',
        'input_paths': ['reddit_comments'],
        'output_dir': 'mod_comments',
        'produces': [
            'stage1_subreddit_mod_comment_rankings.json',
            '*_mod_comments.zst'  # One per RC file
        ]
    },

    'stage2_top_sfw': {
        'name': 'Get Top N SFW Subreddits',
        'script': '2_get_top_sfw_subreddits.py',
        'input_files': ['stage1_subreddit_mod_comment_rankings.json'],
        'output_dir': 'data',
        'produces': ['stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json']
    },

    # Phase 2: Comment Filtering & Matching
    'stage3_filter_and_consolidate': {
        'name': 'Filter and Consolidate Top N Subreddits',
        'script': '3_filter_and_consolidate.py',
        'input_paths': ['mod_comments'],
        'input_files': ['stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json'],
        'output_dir': 'top_subreddits',
        'produces': [
            '{subreddit}_mod_comments.jsonl.zst',
            'stage3_filter_and_consolidate_summary.json'
        ]
    },

    'stage4_match_rules': {
        'name': 'Match Comments to Rules, Rank by JSD, Collect Submission IDs',
        'script': '4_match_rules.py',
        'input_paths': ['top_subreddits'],
        'input_files': ['stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json'],
        'output_dir': 'matched_comments',
        'produces': [
            '{subreddit}_match.jsonl.zst',
            '{subreddit}_stats.json',
            'stage4_matching_summary.json',
            'stage4_subreddit_submission_ids.json'
        ]
    },

    # Phase 3: Thread Construction
    'stage5_collect_submission_comments': {
        'name': 'Collect Submission Comments',
        'script': '5_collect_submission_comments.py',
        'input_paths': ['reddit_comments'],
        'input_files': ['stage4_subreddit_submission_ids.json'],
        'output_dir': 'organized_comments',
        'produces': [
            '{subreddit}_submission_comments.pkl',
            'stage5_submission_comment_organization_stats.json'
        ]
    },

    'stage6_build_trees_and_threads': {
        'name': 'Build Comment Trees and Discussion Threads',
        'script': '6_build_trees_and_threads.py',
        'input_paths': ['organized_comments', 'matched_comments'],
        'input_files': ['stage2_top_{TOP_N_SUBREDDITS_WITH_MOD_COMMENTS}_sfw_subreddits.json'],
        'output_dir': 'comment_trees',  # Primary output
        'produces': [
            '{subreddit}_comment_trees.pkl',
            '{subreddit}_discussion_threads.pkl',
            'stage6_trees_and_threads_summary.json'
        ]
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
        ]
    },

    'stage8_collect_media': {
        'name': 'Download Media Files for Submissions',
        'script': '8_collect_media.py',
        'input_paths': ['submissions'],
        'input_files': ['stage7_submission_collection_stats.json'],
        'output_dir': 'media',
        'produces': [
            '{subreddit}/{submission_id}.{ext}',  # Downloaded media files
            'stage8_media_collection_stats.json'
        ]
    },

    # Phase 4: Dataset Finalization
    'stage9_finalize_dataset': {
        'name': 'Finalize Dataset',
        'script': '9_finalize_dataset.py',
        'input_paths': ['discussion_threads', 'submissions', 'media'],
        'input_files': ['stage6_trees_and_threads_summary.json', 'stage7_submission_collection_stats.json', 'stage8_media_collection_stats.json'],
        'output_dir': 'final_dataset',
        'produces': [
            'reddit_moderation_dataset.pkl',
            'stage9_dataset_manifest.json',
            'stage9_final_statistics.json'
        ]
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
        resolved_file_name = file_name.format(TOP_N_SUBREDDITS_WITH_MOD_COMMENTS=TOP_N_SUBREDDITS_WITH_MOD_COMMENTS)
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
    print("=" * 50)
    print()

    for i in range(0, 10):  # Now 0-9 stages
        stage_info = get_stage_info(i)
        if stage_info:
            valid, msg = validate_stage_inputs(i)
            output_path = get_output_path_for_stage(i)
            output_exists = os.path.exists(output_path) if output_path else False

            status = "✓" if valid else "✗"
            output_status = "✓" if output_exists else "✗"

            print(f"Stage {i:2d}: {stage_info['name']}")
            print(f"         Input: {status} | Output: {output_status}")
            if not valid:
                print(f"         Issue: {msg}")

    print()

if __name__ == "__main__":
    # When run directly, show pipeline status
    create_directories()
    print_pipeline_status()