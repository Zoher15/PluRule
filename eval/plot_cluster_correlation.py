#!/usr/bin/env python3
"""
Plot correlation between rule cluster accuracy and cluster size metrics.

Generates two-column scatter plots showing:
- Left: Accuracy vs. number of subreddits in cluster
- Right: Accuracy vs. number of rules in cluster

Each point represents one rule cluster with horizontal 95% CI error bars.

Usage:
    python plot_cluster_correlation.py
    python plot_cluster_correlation.py --model qwen3-vl-30b --split test
    python plot_cluster_correlation.py --show-regression --annotate-outliers
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from adjustText import adjust_text

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PATHS
from plotting_config import create_two_column_figure, save_figure, PUBLICATION_DPI


def get_latest_performance_file(perf_dir: Path) -> Path:
    """Get latest performance file, preferring _ci version if available."""
    perf_files_ci = sorted(perf_dir.glob('performance_*_ci.json'))
    perf_files = sorted(perf_dir.glob('performance_*.json'))

    if perf_files_ci:
        return perf_files_ci[-1]
    elif perf_files:
        return perf_files[-1]
    else:
        raise FileNotFoundError(f"No performance files in {perf_dir}")


def load_cluster_size_stats(split: str = 'all', cluster_type: str = 'rule') -> dict:
    """Load cached cluster size statistics.

    Args:
        split: Dataset split (train/val/test/all)
        cluster_type: 'rule' or 'subreddit'

    Returns:
        Dictionary with cluster stats
    """
    data_dir = Path(PATHS['data'])
    cache_file = data_dir / f'{cluster_type}_cluster_size_stats_{split}.json'

    if not cache_file.exists():
        raise FileNotFoundError(
            f"Cluster size stats cache not found: {cache_file}\n"
            f"Please run: python eval/compute_cluster_size_stats.py --cluster-type {cluster_type}"
            + (f" --split {split}" if split != 'all' else "")
        )

    with open(cache_file) as f:
        return json.load(f)


def merge_performance_and_sizes(perf_data: dict, size_stats: dict, metric: str = 'overall_accuracy', cluster_type: str = 'rule'):
    """Merge performance metrics with cluster size statistics.

    Args:
        perf_data: Performance JSON data
        size_stats: Cluster size statistics
        metric: Accuracy metric to use
        cluster_type: 'rule' or 'subreddit'

    Returns:
        List of tuples: (cluster_name, accuracy, ci_low, ci_high, n_subreddits, n_rules)
    """
    per_cluster_key = f'per_{cluster_type}_cluster'
    per_cluster = perf_data['metrics'].get(per_cluster_key, {})
    cluster_size_data = size_stats['cluster_stats']

    merged_data = []

    for cluster_name, perf_info in per_cluster.items():
        if metric not in perf_info:
            continue

        # Get accuracy
        accuracy = perf_info[metric] * 100

        # Get confidence interval
        ci_key = f'{metric}_ci'
        if ci_key in perf_info:
            ci_low, ci_high = perf_info[ci_key]
            ci_low *= 100
            ci_high *= 100
        else:
            # No CI available
            ci_low, ci_high = accuracy, accuracy

        # Get cluster sizes
        if cluster_name not in cluster_size_data:
            print(f"  ⚠ Cluster '{cluster_name}' not found in size stats, skipping")
            continue

        n_subreddits = cluster_size_data[cluster_name]['n_subreddits']
        n_rules = cluster_size_data[cluster_name]['n_rules']

        merged_data.append((cluster_name, accuracy, ci_low, ci_high, n_subreddits, n_rules))

    return merged_data


def compute_correlation(x, y):
    """Compute Spearman correlation coefficient and p-value.

    Args:
        x: Array of x values
        y: Array of y values

    Returns:
        Tuple of (rho, p_value)
    """
    return stats.spearmanr(x, y)


def plot_scatter(ax, x, y, color, label):
    """Plot scatter points.

    Args:
        ax: Matplotlib axis
        x: X values (accuracy)
        y: Y values (count metric)
        color: Point color
        label: Legend label
    """
    ax.scatter(x, y, color=color, s=20, alpha=0.7, marker='o', zorder=3, label=label)


def add_regression_line(ax, x, y, color):
    """Add linear regression line with 95% confidence band.

    Args:
        ax: Matplotlib axis
        x: X values
        y: Y values
        color: Line color
    """
    # Fit regression
    slope, intercept, r_value, _, _ = stats.linregress(x, y)

    # Generate line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    # Plot line
    ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.6,
            linestyle='--', zorder=1, label=f'Linear fit (r={r_value:.2f})')


def annotate_outliers(ax, x, y, labels, n_outliers=3):
    """Annotate outlier points.

    Args:
        ax: Matplotlib axis
        x: X values
        y: Y values
        labels: Cluster names
        n_outliers: Number of outliers to label
    """
    # Compute residuals from regression line
    slope, intercept, _, _, _ = stats.linregress(x, y)
    y_pred = slope * np.array(x) + intercept
    residuals = np.abs(np.array(y) - y_pred)

    # Get indices of top outliers
    outlier_indices = np.argsort(residuals)[-n_outliers:]

    # Annotate
    for idx in outlier_indices:
        ax.annotate(labels[idx],
                   xy=(x[idx], y[idx]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=6,
                   alpha=0.8)


def plot_cluster_correlation(model, split, context, metric='overall_accuracy',
                             phrase='baseline', mode='prefill',
                             show_regression=False, annotate_outliers_flag=False,
                             cluster_split='all', cluster_type='rule'):
    """Generate correlation scatter plots.

    Args:
        model: Model name
        split: Dataset split
        context: Context type
        metric: Accuracy metric
        phrase: Phrase type
        mode: Mode type
        show_regression: Whether to show regression lines
        annotate_outliers_flag: Whether to annotate outlier points

    Returns:
        0 on success, 1 on error
    """
    # Load data
    eval_dir = Path(PATHS['data']).parent / 'output' / 'eval'
    perf_dir = eval_dir / model / split / context / ('baseline' if phrase == 'baseline' else f'{phrase}_{mode}')

    try:
        perf_file = get_latest_performance_file(perf_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    print(f"Loading performance data: {perf_file.name}")
    with open(perf_file) as f:
        perf_data = json.load(f)

    print(f"Loading cluster size stats: split={cluster_split}, type={cluster_type}")
    try:
        size_stats = load_cluster_size_stats(split=cluster_split, cluster_type=cluster_type)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Merge data
    merged_data = merge_performance_and_sizes(perf_data, size_stats, metric=metric, cluster_type=cluster_type)

    # Filter out "Other" cluster
    merged_data = [d for d in merged_data if d[0] != 'Other']

    if not merged_data:
        print(f"❌ No cluster data available")
        return 1

    print(f"  ✓ Merged {len(merged_data)} clusters (excluded 'Other')")

    # Unpack data
    cluster_names, accuracies, ci_lows, ci_highs, n_subreddits, n_rules = zip(*merged_data)

    # Convert to numpy arrays
    accuracies = np.array(accuracies)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    n_subreddits = np.array(n_subreddits)
    n_rules = np.array(n_rules)

    # Compute axis limits with 5% padding
    acc_min, acc_max = accuracies.min(), accuracies.max()
    acc_range = acc_max - acc_min
    xlim = (acc_min - 0.05 * acc_range, acc_max + 0.05 * acc_range)

    sub_min, sub_max = n_subreddits.min(), n_subreddits.max()
    sub_range = sub_max - sub_min
    ylim_sub = (sub_min - 0.05 * sub_range, sub_max + 0.05 * sub_range)

    rules_min, rules_max = n_rules.min(), n_rules.max()
    rules_range = rules_max - rules_min
    ylim_rules = (rules_min - 0.05 * rules_range, rules_max + 0.05 * rules_range)

    # Create figure
    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    # LEFT: Accuracy vs. # Subreddits
    plot_scatter(ax_left, accuracies, n_subreddits,
                color='#FF4500', label='Rule clusters')

    # Annotate all points with cluster names (with automatic adjustment)
    texts_left = []
    for i, name in enumerate(cluster_names):
        txt = ax_left.text(accuracies[i], n_subreddits[i], name,
                          fontsize=4.5, alpha=0.8)
        texts_left.append(txt)
    adjust_text(texts_left, ax=ax_left,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    # Compute and display correlation
    rho_sub, p_sub = compute_correlation(accuracies, n_subreddits)
    ax_left.text(0.05, 0.88, f'ρ = {rho_sub:.3f}\np = {p_sub:.3f}',
                transform=ax_left.transAxes,
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        add_regression_line(ax_left, accuracies, n_subreddits, '#FF4500')

    ax_left.set_xlabel('Accuracy (%)', fontsize=8)
    ax_left.set_ylabel('Number of Subreddits', fontsize=8)
    ax_left.tick_params(axis='both', labelsize=7, pad=0.5, length=3, width=0.25)
    ax_left.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_left.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    ax_left.spines['left'].set_linewidth(0.25)
    ax_left.spines['bottom'].set_linewidth(0.25)
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim_sub)

    # RIGHT: Accuracy vs. # Rules
    plot_scatter(ax_right, accuracies, n_rules,
                color='#FF4500', label='Rule clusters')

    # Annotate all points with cluster names (with automatic adjustment)
    texts_right = []
    for i, name in enumerate(cluster_names):
        txt = ax_right.text(accuracies[i], n_rules[i], name,
                           fontsize=4.5, alpha=0.8)
        texts_right.append(txt)
    adjust_text(texts_right, ax=ax_right,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    # Compute and display correlation
    rho_rules, p_rules = compute_correlation(accuracies, n_rules)
    ax_right.text(0.05, 0.88, f'ρ = {rho_rules:.3f}\np = {p_rules:.3f}',
                 transform=ax_right.transAxes,
                 fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        add_regression_line(ax_right, accuracies, n_rules, '#FF4500')

    ax_right.set_xlabel('Accuracy (%)', fontsize=8)
    ax_right.set_ylabel('Number of Rules', fontsize=8)
    ax_right.tick_params(axis='both', labelsize=7, pad=0.5, length=3, width=0.25)
    ax_right.grid(axis='both', alpha=0.2, linestyle='--', linewidth=0.5)
    ax_right.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_linewidth(0.25)
    ax_right.spines['bottom'].set_linewidth(0.25)
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim_rules)

    # Subplot labels - top left above correlation scores
    ax_left.text(0.02, 0.98, '(a)', transform=ax_left.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left')
    ax_right.text(0.02, 0.98, '(b)', transform=ax_right.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='left')

    # Adjust layout
    fig.subplots_adjust(left=0.075, right=0.98, top=0.99, bottom=0.10, wspace=0.15)

    # Save
    filename = f"{cluster_type}_cluster_correlation_{model}_{split}_{context}_{phrase if phrase=='baseline' else f'{phrase}_{mode}'}_{metric}"
    plots_dir = eval_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)

    print(f"✅ Saved: {plots_dir / filename}.pdf")
    print(f"\nSpearman Correlation Results:")
    print(f"  Accuracy vs. Subreddits: ρ = {rho_sub:.3f}, p = {p_sub:.3f}")
    print(f"  Accuracy vs. Rules:      ρ = {rho_rules:.3f}, p = {p_rules:.3f}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation between cluster accuracy and size metrics'
    )
    parser.add_argument('--model', default='gpt5.2-high', help='Model name')
    parser.add_argument('--split', default='test', help='Dataset split for accuracy')
    parser.add_argument('--context', default='submission-media-discussion-user', help='Context')
    parser.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')
    parser.add_argument('--phrase', default='baseline', help='Phrase type')
    parser.add_argument('--mode', default='prefill', help='Mode')
    parser.add_argument('--cluster-split', default='all',
                       help='Dataset split for cluster sizes (default: all)')
    parser.add_argument('--cluster-type', default='rule',
                       choices=['rule', 'subreddit'],
                       help='Cluster type (default: rule)')
    parser.add_argument('--show-regression', action='store_true',
                       help='Show regression lines')
    parser.add_argument('--annotate-outliers', action='store_true',
                       help='Annotate outlier cluster names')
    args = parser.parse_args()

    return plot_cluster_correlation(
        args.model, args.split, args.context, args.metric,
        args.phrase, args.mode,
        show_regression=args.show_regression,
        annotate_outliers_flag=args.annotate_outliers,
        cluster_split=args.cluster_split,
        cluster_type=args.cluster_type
    )


if __name__ == '__main__':
    exit(main())
