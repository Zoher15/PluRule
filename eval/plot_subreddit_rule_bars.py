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
    python plot_subreddit_rule_bars.py drilldown --model gpt5.2-high --rule-cluster civility
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


def plot_two_column_bars(ax_left, ax_right, sub_labels, sub_values, rule_labels, rule_values, xlabel, bar_values=None, show_baseline=False, log_scale=False, ci_errors=None):
    """Standardized two-column horizontal bar plot.

    Args:
        ci_errors: Optional tuple of (sub_ci_errors, rule_ci_errors) where each is
                   a 2xN array with [lower_errors, upper_errors] for error bars.
    """
    y_sub = np.arange(len(sub_labels))
    y_rule = np.arange(len(rule_labels))

    # LEFT: Subreddit (Reddit Lapis Lazuli) - horizontal bars
    ax_left.barh(y_sub, sub_values, height=0.8, color='#336699', edgecolor='none')
    ax_left.set_xlabel(xlabel, fontsize=8)
    ax_left.set_yticks(y_sub)
    ax_left.set_yticklabels(sub_labels, fontsize=6)
    ax_left.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_left.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_left.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_left.set_ylim(-0.45, len(sub_labels) - 0.2)
    ax_left.invert_yaxis()  # Highest values at top
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(0.25)
    ax_left.spines['bottom'].set_linewidth(0.25)
    if log_scale:
        ax_left.set_xscale('log')
        ax_left.set_xlim(left=10, right=10000)
    elif show_baseline:
        ax_left.set_xlim(left=25, right=70)
    else:
        ax_left.set_xlim(left=10)

    # RIGHT: Rule (Orange Red) - horizontal bars
    ax_right.barh(y_rule, rule_values, height=0.8, color='#FF4500', edgecolor='none')
    ax_right.set_xlabel(xlabel, fontsize=8)
    ax_right.set_yticks(y_rule)
    ax_right.set_yticklabels(rule_labels, fontsize=6)
    ax_right.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_right.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_right.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_right.set_ylim(-0.45, len(rule_labels) - 0.2)
    ax_right.invert_yaxis()  # Highest values at top
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_linewidth(0.25)
    ax_right.spines['bottom'].set_linewidth(0.25)
    if log_scale:
        ax_right.set_xscale('log')
        ax_right.set_xlim(left=10, right=10000)
    elif show_baseline:
        ax_right.set_xlim(left=25, right=70)
    else:
        ax_right.set_xlim(left=10)

    # Add baseline if needed
    if show_baseline:
        for ax in [ax_left, ax_right]:
            ax.axvline(x=50, color='lightgray', linestyle='--', alpha=0.9, linewidth=1)

    # Add CI error bars if provided
    if ci_errors is not None:
        sub_ci, rule_ci = ci_errors
        if sub_ci is not None:
            ax_left.errorbar(sub_values, y_sub, xerr=sub_ci, fmt='none',
                           ecolor='black', elinewidth=0.5, capsize=1.5, capthick=0.5)
        if rule_ci is not None:
            ax_right.errorbar(rule_values, y_rule, xerr=rule_ci, fmt='none',
                            ecolor='black', elinewidth=0.5, capsize=1.5, capthick=0.5)

    # Add value labels inside bars
    if bar_values:
        sub_bar_values, rule_bar_values = bar_values
        for ax, values in [(ax_left, sub_bar_values), (ax_right, rule_bar_values)]:
            for bar, val in zip(ax.patches, values):
                ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2.,
                       f'{val:.0f}', ha='right', va='center', fontsize=5, color='white')

    # Labels in bottom right corner (no bold)
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right')


def plot_two_column_forest(ax_left, ax_right, sub_labels, sub_values, sub_cis,
                           rule_labels, rule_values, rule_cis, xlabel):
    """Forest plot with dots for point estimates and horizontal lines for 95% CIs.

    Args:
        sub_labels, rule_labels: Cluster names
        sub_values, rule_values: Point estimates (accuracy %)
        sub_cis, rule_cis: List of (ci_lower, ci_upper) tuples for each cluster
    """
    y_sub = np.arange(len(sub_labels))
    y_rule = np.arange(len(rule_labels))

    # LEFT: Subreddit clusters
    # Faint horizontal lines at each cluster position
    for i in y_sub:
        ax_left.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
    ax_left.scatter(sub_values, y_sub, color='#336699', s=30, zorder=3)
    # Add accuracy values on top of circles
    for i, val in enumerate(sub_values):
        ax_left.text(val, i, f'{val:.0f}', ha='center', va='center', fontsize=4, color='white', fontweight='bold', zorder=4)
    for i, (ci_low, ci_high) in enumerate(sub_cis):
        ax_left.hlines(i, ci_low, ci_high, color='#336699', linewidth=1, zorder=2)
        # Caps at ends
        ax_left.vlines([ci_low, ci_high], i - 0.25, i + 0.25, color='#336699', linewidth=1, zorder=2)

    ax_left.set_xlabel(xlabel, fontsize=8)
    ax_left.set_yticks(y_sub)
    ax_left.set_yticklabels(sub_labels, fontsize=6)
    ax_left.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_left.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_left.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_left.set_ylim(-0.5, len(sub_labels) - 0.5)
    ax_left.set_xlim(0, 100)
    ax_left.invert_yaxis()
    ax_left.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(0.25)
    ax_left.spines['bottom'].set_linewidth(0.25)

    # RIGHT: Rule clusters
    # Faint horizontal lines at each cluster position
    for i in y_rule:
        ax_right.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
    ax_right.scatter(rule_values, y_rule, color='#FF4500', s=30, zorder=3)
    # Add accuracy values on top of circles
    for i, val in enumerate(rule_values):
        ax_right.text(val, i, f'{val:.0f}', ha='center', va='center', fontsize=4, color='white', fontweight='bold', zorder=4)
    for i, (ci_low, ci_high) in enumerate(rule_cis):
        ax_right.hlines(i, ci_low, ci_high, color='#FF4500', linewidth=1, zorder=2)
        # Caps at ends
        ax_right.vlines([ci_low, ci_high], i - 0.25, i + 0.25, color='#FF4500', linewidth=1, zorder=2)

    ax_right.set_xlabel(xlabel, fontsize=8)
    ax_right.set_yticks(y_rule)
    ax_right.set_yticklabels(rule_labels, fontsize=6)
    ax_right.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_right.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_right.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_right.set_ylim(-0.5, len(rule_labels) - 0.5)
    ax_right.set_xlim(0, 100)
    ax_right.invert_yaxis()
    ax_right.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_linewidth(0.25)
    ax_right.spines['bottom'].set_linewidth(0.25)

    # Labels in bottom right corner
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right')


def plot_two_column_stacked(ax_left, ax_right, sub_labels, sub_mod, sub_overall, sub_unmod,
                            rule_labels, rule_mod, rule_overall, rule_unmod, xlabel):
    """Overlapping bar plot showing moderated, overall, and unmoderated accuracy.

    Bars are drawn back to front: unmoderated (lightest) -> overall -> moderated (darkest).
    Annotations in black at end of each bar.
    """
    y_sub = np.arange(len(sub_labels))
    y_rule = np.arange(len(rule_labels))

    # LEFT: Subreddit clusters
    # Draw bars back to front
    ax_left.barh(y_sub, sub_unmod, height=0.8, color='#336699', alpha=0.2, edgecolor='none', zorder=2)
    ax_left.barh(y_sub, sub_overall, height=0.8, color='#336699', alpha=0.55, edgecolor='none', zorder=3)
    ax_left.barh(y_sub, sub_mod, height=0.8, color='#336699', alpha=1.0, edgecolor='none', zorder=4)
    # Annotate only overall accuracy (white, inside bar at end)
    for i, ovr in enumerate(sub_overall):
        ax_left.text(ovr - 1, i, f'{ovr:.0f}', fontsize=4.75, color='white', fontweight='bold', va='center', ha='right', zorder=5)

    ax_left.set_xlabel(xlabel, fontsize=8)
    ax_left.set_yticks(y_sub)
    ax_left.set_yticklabels(sub_labels, fontsize=6)
    ax_left.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_left.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_left.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_left.set_ylim(-0.5, len(sub_labels) - 0.5)
    ax_left.set_xlim(0, 105)
    ax_left.invert_yaxis()
    ax_left.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(0.25)
    ax_left.spines['bottom'].set_linewidth(0.25)

    # RIGHT: Rule clusters
    # Draw bars back to front
    ax_right.barh(y_rule, rule_unmod, height=0.8, color='#FF4500', alpha=0.2, edgecolor='none', zorder=2)
    ax_right.barh(y_rule, rule_overall, height=0.8, color='#FF4500', alpha=0.55, edgecolor='none', zorder=3)
    ax_right.barh(y_rule, rule_mod, height=0.8, color='#FF4500', alpha=1.0, edgecolor='none', zorder=4)
    # Annotate only overall accuracy (white, inside bar at end)
    for i, ovr in enumerate(rule_overall):
        ax_right.text(ovr - 1, i, f'{ovr:.0f}', fontsize=4.75, color='white', fontweight='bold', va='center', ha='right', zorder=5)

    ax_right.set_xlabel(xlabel, fontsize=8)
    ax_right.set_yticks(y_rule)
    ax_right.set_yticklabels(rule_labels, fontsize=6)
    ax_right.tick_params(axis='x', labelsize=6, pad=0.5, length=3, width=0.25)
    ax_right.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax_right.grid(axis='x', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_right.set_ylim(-0.5, len(rule_labels) - 0.5)
    ax_right.set_xlim(0, 105)
    ax_right.invert_yaxis()
    ax_right.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_linewidth(0.25)
    ax_right.spines['bottom'].set_linewidth(0.25)

    # Labels in bottom right corner
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right')


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

        sorted_data = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_data = [('other' if l.lower() == 'other' else l, c) for l, c in sorted_data]

        if cluster_type == 'subreddit':
            sub_labels, sub_counts = zip(*sorted_data)
        else:
            rule_labels, rule_counts = zip(*sorted_data)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='scatter')
    plot_two_column_bars(ax_left, ax_right, sub_labels, sub_counts, rule_labels, rule_counts,
                         'Number of Thread Pairs', log_scale=True)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.99, bottom=0.11, wspace=0.30)
    save_figure(fig, Path(PATHS['data']) / 'stage10_cluster_distribution', dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("✅ Distribution plot saved")
    return 0


def plot_cluster_analysis(model, split, context, metric, phrase='baseline', mode='prefill'):
    """Plot accuracy by cluster from evaluation results using bar plots."""
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
        # Extract (name, accuracy, count) tuples
        cluster_data = []
        for name, info in clusters.items():
            if metric in info:
                acc = info[metric] * 100
                count = info.get('count', 0)
                cluster_data.append((name, acc, count))

        sorted_data = sorted(cluster_data, key=lambda x: x[1], reverse=True)
        # Lowercase "Other" to "other" for consistency
        sorted_data = [('other' if n.lower() == 'other' else n, a, c) for n, a, c in sorted_data]

        if cluster_type == 'subreddit':
            if sorted_data:
                sub_labels, sub_accs, _ = zip(*sorted_data)
            else:
                sub_labels, sub_accs = [], []
        else:
            if sorted_data:
                rule_labels, rule_accs, _ = zip(*sorted_data)
            else:
                rule_labels, rule_accs = [], []

    if not (sub_labels and rule_labels):
        print(f"❌ No cluster metrics found")
        return 1

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='scatter')
    plot_two_column_bars(ax_left, ax_right, sub_labels, sub_accs, rule_labels, rule_accs,
                         'Accuracy (%)', bar_values=(sub_accs, rule_accs), show_baseline=True)

    fig.subplots_adjust(left=0.13, right=0.98, top=0.99, bottom=0.11, wspace=0.28)
    filename = f"cluster_analysis_{model}_{split}_{context}_{phrase if phrase=='baseline' else f'{phrase}_{mode}'}_{metric}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("✅ Cluster analysis plot saved")
    return 0


def plot_cluster_forest(model, split, context, metric, phrase='baseline', mode='prefill'):
    """Forest plot showing accuracy with 95% CI by cluster."""
    eval_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    perf_dir = eval_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    perf_files = sorted(perf_dir.glob('performance_*.json'))
    if not perf_files:
        print(f"No performance files in {perf_dir}")
        return 1

    with open(perf_files[-1]) as f:
        data = json.load(f)

    # Extract metrics with CIs
    for cluster_type in ['subreddit', 'rule']:
        clusters = data['metrics'][f'per_{cluster_type}_cluster']
        cluster_data = []
        for name, info in clusters.items():
            if metric in info:
                acc = info[metric] * 100
                ci_key = f'{metric}_ci'
                if ci_key in info:
                    ci_low, ci_high = info[ci_key]
                    ci_low *= 100
                    ci_high *= 100
                else:
                    # No CI available, use point estimate
                    ci_low, ci_high = acc, acc
                cluster_data.append((name, acc, ci_low, ci_high))

        sorted_data = sorted(cluster_data, key=lambda x: x[1], reverse=True)
        sorted_data = [('other' if n.lower() == 'other' else n, a, cl, ch) for n, a, cl, ch in sorted_data]

        if cluster_type == 'subreddit':
            if sorted_data:
                sub_labels, sub_accs, sub_ci_low, sub_ci_high = zip(*sorted_data)
                sub_cis = list(zip(sub_ci_low, sub_ci_high))
            else:
                sub_labels, sub_accs, sub_cis = [], [], []
        else:
            if sorted_data:
                rule_labels, rule_accs, rule_ci_low, rule_ci_high = zip(*sorted_data)
                rule_cis = list(zip(rule_ci_low, rule_ci_high))
            else:
                rule_labels, rule_accs, rule_cis = [], [], []

    if not (sub_labels and rule_labels):
        print("No cluster metrics found")
        return 1

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='scatter')
    plot_two_column_forest(ax_left, ax_right, sub_labels, sub_accs, sub_cis,
                           rule_labels, rule_accs, rule_cis, 'Accuracy (%)')

    fig.subplots_adjust(left=0.13, right=0.98, top=0.99, bottom=0.11, wspace=0.28)
    filename = f"cluster_forest_{model}_{split}_{context}_{phrase if phrase=='baseline' else f'{phrase}_{mode}'}_{metric}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("✅ Cluster forest plot saved")
    return 0


def plot_cluster_stacked(model, split, context, phrase='baseline', mode='prefill'):
    """Stacked bar plot showing moderated, overall, and unmoderated accuracy by cluster."""
    eval_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    perf_dir = eval_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    perf_files = sorted(perf_dir.glob('performance_*.json'))
    if not perf_files:
        print(f"No performance files in {perf_dir}")
        return 1

    with open(perf_files[-1]) as f:
        data = json.load(f)

    # Extract all three metrics per cluster
    for cluster_type in ['subreddit', 'rule']:
        clusters = data['metrics'][f'per_{cluster_type}_cluster']
        cluster_data = []
        for name, info in clusters.items():
            if 'overall_accuracy' in info:
                mod = info.get('moderated_accuracy', 0) * 100
                overall = info['overall_accuracy'] * 100
                unmod = info.get('unmoderated_accuracy', 0) * 100
                cluster_data.append((name, mod, overall, unmod))

        # Sort by overall accuracy descending
        sorted_data = sorted(cluster_data, key=lambda x: x[2], reverse=True)
        sorted_data = [('other' if n.lower() == 'other' else n, m, o, u) for n, m, o, u in sorted_data]

        if cluster_type == 'subreddit':
            if sorted_data:
                sub_labels, sub_mod, sub_overall, sub_unmod = zip(*sorted_data)
            else:
                sub_labels, sub_mod, sub_overall, sub_unmod = [], [], [], []
        else:
            if sorted_data:
                rule_labels, rule_mod, rule_overall, rule_unmod = zip(*sorted_data)
            else:
                rule_labels, rule_mod, rule_overall, rule_unmod = [], [], [], []

    if not (sub_labels and rule_labels):
        print("No cluster metrics found")
        return 1

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='scatter')
    plot_two_column_stacked(ax_left, ax_right, sub_labels, sub_mod, sub_overall, sub_unmod,
                            rule_labels, rule_mod, rule_overall, rule_unmod, 'Accuracy (%)')

    # Add legend with gray swatches showing alpha levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=1.0, label='moderated'),
        Patch(facecolor='gray', alpha=0.55, label='overall'),
        Patch(facecolor='gray', alpha=0.2, label='unmoderated'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.11, wspace=0.25)
    filename = f"cluster_stacked_{model}_{split}_{context}_{phrase if phrase=='baseline' else f'{phrase}_{mode}'}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("✅ Cluster stacked plot saved")
    return 0


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures (subreddit vs rule bars)')
    parser.add_argument('type', choices=['distribution', 'cluster-analysis', 'cluster-forest', 'cluster-stacked'], help='Plot type')
    parser.add_argument('--model', default='gpt5.2-high', help='Model name')
    parser.add_argument('--split', default='test', help='Dataset split')
    parser.add_argument('--context', default='submission-media-discussion-user', help='Context')
    parser.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')
    parser.add_argument('--phrase', default='baseline', help='Phrase type')
    parser.add_argument('--mode', default='prefill', help='Mode')
    args = parser.parse_args()

    if args.type == 'distribution':
        return plot_distribution()
    elif args.type == 'cluster-forest':
        return plot_cluster_forest(args.model, args.split, args.context, args.metric, args.phrase, args.mode)
    elif args.type == 'cluster-stacked':
        return plot_cluster_stacked(args.model, args.split, args.context, args.phrase, args.mode)
    else:
        return plot_cluster_analysis(args.model, args.split, args.context, args.metric, args.phrase, args.mode)


if __name__ == '__main__':
    exit(main())
