#!/usr/bin/env python3
"""
Extract worst matches per subreddit for analysis.

Gets the bottom N worst matches from each subreddit (lowest similarity scores).
"""

import sys
import os
import json
import glob
from multiprocessing import Pool
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PATHS
from utils.files import read_json_file, read_zst_lines, json_loads

def process_subreddit(subreddit):
    """Process one subreddit and return its worst matches + date stats."""
    matches = []
    total_count = 0
    march_2023_count = 0
    march_2023_timestamp = 1677628800  # March 1, 2023 00:00:00 UTC

    match_file = os.path.join(PATHS['matched_comments'], f"{subreddit}_match.jsonl.zst")
    if not os.path.exists(match_file):
        return subreddit, [], 0, 0

    # Load Stage 2 data to get rule details
    stage2_file = os.path.join(PATHS['data'], 'stage2_sfw_subreddits_min_25_comments.json')
    stage2_data = read_json_file(stage2_file)

    # Find rules for this subreddit
    rules = {}
    for entry in stage2_data['subreddits']:
        if entry['subreddit']['display_name'].lower() == subreddit.lower():
            for rule in entry['rules']:
                rules[rule['rule_index']] = rule['rule_comprehensive']
            break

    lines = read_zst_lines(match_file)
    for line in lines:
        if line.strip():
            try:
                comment = json_loads(line)
                matched_rule = comment.get('matched_rule', {})
                rule_index = matched_rule.get('rule_index')

                total_count += 1
                created_utc = int(comment.get('created_utc', 0))
                if created_utc >= march_2023_timestamp:
                    march_2023_count += 1

                matches.append({
                    'body': comment.get('body_clean', ''),
                    'score': matched_rule.get('similarity_score', 0),
                    'rule': rules.get(rule_index, 'Unknown rule')
                })
            except Exception:
                continue

    # Sort by score and get bottom N
    matches.sort(key=lambda x: x['score'])
    worst_n = matches[:5]  # Bottom 5

    return subreddit, worst_n, total_count, march_2023_count


def main():
    # Load all stats files
    stats_files = glob.glob(os.path.join(PATHS['matched_comments'], '*_stats.json'))
    print(f"Found {len(stats_files)} stats files")

    # Filter subreddits with >=25 matches
    qualifying_subreddits = []
    for stats_file in stats_files:
        stats = read_json_file(stats_file)
        if stats.get('matched_comments', 0) >= 25:
            qualifying_subreddits.append(stats['subreddit'])

    print(f"Found {len(qualifying_subreddits)} subreddits with >=25 matches")

    # Process in parallel using 48 workers
    print(f"Processing {len(qualifying_subreddits)} subreddits with 48 workers...")
    with Pool(48) as pool:
        results = pool.map(process_subreddit, qualifying_subreddits)

    # Organize results by subreddit and collect date stats
    subreddit_worst = {}
    total_matched = 0
    total_march_2023 = 0

    for subreddit, matches, total, march_count in results:
        if matches:
            subreddit_worst[subreddit] = matches
        total_matched += total
        total_march_2023 += march_count

    print(f"Total subreddits with matches: {len(subreddit_worst)}")
    print(f"Total matched comments: {total_matched:,}")
    print(f"Comments from March 2023+: {total_march_2023:,} ({total_march_2023/total_matched*100:.1f}%)")

    # Write to text file
    output_file = '/data3/zkachwal/reddit-mod-collection-pipeline/worst_5_per_subreddit.txt'
    print(f"Writing bottom 5 matches per subreddit to file...")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"BOTTOM 5 WORST MATCHES PER SUBREDDIT ({len(subreddit_worst)} subreddits)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total matched comments: {total_matched:,}\n")
        f.write(f"Comments from March 2023+: {total_march_2023:,} ({total_march_2023/total_matched*100:.1f}%)\n")
        f.write("=" * 100 + "\n\n")

        # Sort subreddits alphabetically
        for subreddit in sorted(subreddit_worst.keys()):
            matches = subreddit_worst[subreddit]

            f.write(f"\n{'='*100}\n")
            f.write(f"SUBREDDIT: r/{subreddit}\n")
            f.write(f"{'='*100}\n\n")

            for i, match in enumerate(matches, 1):
                f.write(f"Rank #{i} | Score: {match['score']:.4f}\n")
                f.write(f"Mod Comment: {match['body']}\n")
                f.write(f"Matched Rule: {match['rule']}\n")
                f.write("-" * 100 + "\n\n")

    print(f"✅ Done! Written to: {output_file}")


if __name__ == "__main__":
    main()
