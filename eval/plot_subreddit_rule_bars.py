#!/usr/bin/env python3
"""
Unified plotting script for subreddit vs rule bar plots (ACL two-column format).

Generates consistent two-column figures with:
- Left: Subreddit data (orange bars)
- Right: Rule data (teal bars)
- Shared y-axis, no titles, (a)(b) labels only

Usage:
    python plot_subreddit_rule_bars.py distribution
    python plot_subreddit_rule_bars.py cluster-analysis --model qwen3-vl-30b --split test --metric overall_accuracy
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS
from plotting_config import create_two_column_figure, save_figure, PUBLICATION_DPI


def plot_two_column_bars(ax_left, ax_right, sub_labels, sub_values, rule_labels, rule_values, ylabel, bar_values=None, show_baseline=False, log_scale=False):
    """Standardized two-column bar plot."""
    x_sub = np.arange(len(sub_labels))
    x_rule = np.arange(len(rule_labels))

    # LEFT: Subreddit (Orange)
    ax_left.bar(x_sub, sub_values, color='#EE7733', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_left.set_ylabel(ylabel, fontsize=5)
    ax_left.set_xticks(x_sub)
    ax_left.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=3.5)
    ax_left.tick_params(axis='y', labelsize=3.5, pad=0.5, length=3)
    ax_left.tick_params(axis='x', pad=0.5, length=3)
    ax_left.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_left.set_xlim(-0.8, len(sub_labels) - 0.2)
    if log_scale:
        ax_left.set_yscale('log')
        ax_left.set_ylim(bottom=1, top=10000)

    # RIGHT: Rule (Teal)
    ax_right.bar(x_rule, rule_values, color='#0077BB', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax_right.set_xticks(x_rule)
    ax_right.set_xticklabels(rule_labels, rotation=45, ha='right', fontsize=3.5)
    ax_right.tick_params(axis='y', labelsize=3.5, pad=0.5, length=3)
    ax_right.tick_params(axis='x', pad=0.5, length=3)
    ax_right.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_right.set_xlim(-0.8, len(rule_labels) - 0.2)
    if log_scale:
        ax_right.set_yscale('log')
        ax_right.set_ylim(bottom=1, top=10000)

    # Add baseline if needed
    if show_baseline:
        for ax in [ax_left, ax_right]:
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Add value labels on bars
    if bar_values:
        sub_bar_values, rule_bar_values = bar_values
        for ax, values in [(ax_left, sub_bar_values), (ax_right, rule_bar_values)]:
            for bar, val in zip(ax.patches, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.0f}', ha='center', va='bottom', fontsize=3)

    # Labels in top right corner (no bold)
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.98, f'({label})', transform=ax.transAxes,
               fontsize=7, verticalalignment='top', horizontalalignment='right')


def plot_distribution():
    """Plot cluster distribution from stage10 stats."""
    stats_file = Path(PATHS['data']) / 'stage10_cluster_assignment_stats.json'
    if not stats_file.exists():
        print(f"❌ Stats file not found: {stats_file}")
        return 1

    with open(stats_file) as f:
        data = json.load(f)
    all_stats = data.get('cluster_assignment_statistics', {})

    # Extract distributions
    for cluster_type in ['subreddit', 'rule']:
        total_counts = Counter()
        for split, stats in all_stats.items():
            key = f'{cluster_type}_clusters'
            if key in stats:
                for label, count in stats[key].items():
                    total_counts[label] += count

        filtered = {l: c for l, c in total_counts.items() if l.lower().strip() != 'other'}
        sorted_data = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        if cluster_type == 'subreddit':
            sub_labels, sub_counts = zip(*sorted_data)
        else:
            rule_labels, rule_counts = zip(*sorted_data)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot', gridspec_kw={'wspace': 0.07})
    plot_two_column_bars(ax_left, ax_right, sub_labels, sub_counts, rule_labels, rule_counts,
                         'Number of Thread Pairs', log_scale=True)

    fig.tight_layout(pad=0.3, w_pad=0.75)
    save_figure(fig, Path(PATHS['data']) / 'stage10_cluster_distribution', dpi=PUBLICATION_DPI)
    plt.close(fig)
    print("✅ Distribution plot saved")
    return 0


def plot_cluster_analysis(model, split, context, metric, phrase='baseline', mode='prefill'):
    """Plot accuracy by cluster from evaluation results."""
    eval_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    perf_dir = eval_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    perf_files = sorted(perf_dir.glob('performance_*.json'))
    if not perf_files:
        print(f"❌ No performance files in {perf_dir}")
        return 1

    with open(perf_files[-1]) as f:
        data = json.load(f)

    # Extract metrics
    for cluster_type in ['subreddit', 'rule']:
        clusters = data['metrics'][f'per_{cluster_type}_cluster']
        sorted_data = sorted([(name, info[metric]*100, info.get('count', 0))
                             for name, info in clusters.items() if metric in info],
                            key=lambda x: x[1], reverse=True)

        if cluster_type == 'subreddit':
            sub_labels, sub_accs, _ = zip(*sorted_data) if sorted_data else ([], [], [])
        else:
            rule_labels, rule_accs, _ = zip(*sorted_data) if sorted_data else ([], [], [])

    if not (sub_labels and rule_labels):
        print(f"❌ No cluster metrics found")
        return 1

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot', gridspec_kw={'wspace': 0.05})
    plot_two_column_bars(ax_left, ax_right, sub_labels, sub_accs, rule_labels, rule_accs,
                         'Accuracy (%)', bar_values=(sub_accs, rule_accs), show_baseline=True)

    fig.tight_layout(pad=0.3, w_pad=0.75)
    filename = f"cluster_analysis_{model}_{split}_{context}_{phrase if phrase=='baseline' else f'{phrase}_{mode}'}_{metric}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI)
    plt.close(fig)
    print("✅ Cluster analysis plot saved")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures (subreddit vs rule bars)')
    parser.add_argument('type', choices=['distribution', 'cluster-analysis'], help='Plot type')
    parser.add_argument('--model', default='qwen3-vl-30b', help='Model name')
    parser.add_argument('--split', default='test', help='Dataset split')
    parser.add_argument('--context', default='subreddit-submission-media-discussion-user', help='Context')
    parser.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')
    parser.add_argument('--phrase', default='baseline', help='Phrase type')
    parser.add_argument('--mode', default='prefill', help='Mode')
    args = parser.parse_args()

    if args.type == 'distribution':
        return plot_distribution()
    else:
        return plot_cluster_analysis(args.model, args.split, args.context, args.metric, args.phrase, args.mode)


if __name__ == '__main__':
    exit(main())
