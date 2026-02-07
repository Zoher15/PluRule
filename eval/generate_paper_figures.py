#!/usr/bin/env python3
"""
Generate paper figures from evaluation results.

Subcommands:
    forest          Cluster forest plot (accuracy + 95% CI per cluster)
    stacked         Cluster stacked bar plot (violating/overall/compliant)
    correlation     Cluster correlation scatter (accuracy vs. cluster size)
    language        Language analysis diverging plot
    all             Generate all figures

Usage:
    python eval/generate_paper_figures.py all
    python eval/generate_paper_figures.py forest --model gpt5.2-high --split test
    python eval/generate_paper_figures.py correlation --show-regression
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plotting_config import (
    create_two_column_figure, save_figure, style_clean_axis, PUBLICATION_DPI
)
from eval.eval_utils import (
    get_eval_dir, load_performance, extract_cluster_metrics,
    extract_three_metrics, load_cluster_size_stats,
)


# ============================================================================
# COLORS
# ============================================================================

COLOR_SUBREDDIT = '#FF4500'  # Orange Red
COLOR_RULE = '#336699'       # Lapis Lazuli


# ============================================================================
# FOREST PLOT
# ============================================================================

def _plot_forest_panel(ax, labels, values, cis, color, xlabel):
    """Draw one panel of the forest plot (shared between subreddit and rule)."""
    y_pos = np.arange(len(labels))

    # Faint horizontal lines at each cluster position
    for i in y_pos:
        ax.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    ax.scatter(values, y_pos, color=color, s=30, marker='s', zorder=3)

    # Accuracy values inside squares
    for i, val in enumerate(values):
        ax.text(val, i, f'{val:.0f}', ha='center', va='center',
                fontsize=4.5, color='white', fontweight='bold', zorder=4)

    # CI error bars with caps
    for i, (ci_low, ci_high) in enumerate(cis):
        ax.hlines(i, ci_low, ci_high, color=color, linewidth=1, zorder=2)
        ax.vlines([ci_low, ci_high], i - 0.25, i + 0.25, color=color, linewidth=1, zorder=2)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax, grid_axis='x')


def plot_forest(model, split, context, metric, phrase='baseline', mode='prefill'):
    """Forest plot showing accuracy with 95% CI by cluster."""
    perf_data = load_performance(model, split, context, phrase, mode)

    sub_data = extract_cluster_metrics(perf_data, 'subreddit', metric)
    rule_data = extract_cluster_metrics(perf_data, 'rule', metric)

    if not sub_data or not rule_data:
        print("No cluster metrics found")
        return 1

    sub_labels, sub_accs, sub_ci_low, sub_ci_high = zip(*sub_data)
    sub_cis = list(zip(sub_ci_low, sub_ci_high))
    rule_labels, rule_accs, rule_ci_low, rule_ci_high = zip(*rule_data)
    rule_cis = list(zip(rule_ci_low, rule_ci_high))

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    _plot_forest_panel(ax_left, sub_labels, sub_accs, sub_cis, COLOR_SUBREDDIT, 'Accuracy (%)')
    _plot_forest_panel(ax_right, rule_labels, rule_accs, rule_cis, COLOR_RULE, 'Accuracy (%)')

    # Subplot labels
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    fig.subplots_adjust(left=0.15, right=0.98, top=0.99, bottom=0.1, wspace=0.32)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"cluster_forest_{model}_{split}_{context}_{phrase_suffix}_{metric}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("Done: cluster forest plot")
    return 0


# ============================================================================
# STACKED BAR PLOT
# ============================================================================

def _plot_stacked_panel(ax, labels, violating, overall, compliant, color, xlabel):
    """Draw one panel of the stacked bar plot."""
    y_pos = np.arange(len(labels))

    # Draw bars back to front: compliant (lightest) -> overall -> violating (darkest)
    ax.barh(y_pos, compliant, height=0.8, color=color, alpha=0.2, edgecolor='none', zorder=2)
    ax.barh(y_pos, overall, height=0.8, color=color, alpha=0.55, edgecolor='none', zorder=3)
    ax.barh(y_pos, violating, height=0.8, color=color, alpha=1.0, edgecolor='none', zorder=4)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax, grid_axis='x')


def plot_stacked(model, split, context, phrase='baseline', mode='prefill'):
    """Stacked bar plot showing violating, overall, and compliant accuracy by cluster."""
    perf_data = load_performance(model, split, context, phrase, mode)

    sub_data = extract_three_metrics(perf_data, 'subreddit')
    rule_data = extract_three_metrics(perf_data, 'rule')

    if not sub_data or not rule_data:
        print("No cluster metrics found")
        return 1

    sub_labels, sub_vio, sub_overall, sub_comp = zip(*sub_data)
    rule_labels, rule_vio, rule_overall, rule_comp = zip(*rule_data)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    _plot_stacked_panel(ax_left, sub_labels, sub_vio, sub_overall, sub_comp, COLOR_SUBREDDIT, '%')
    _plot_stacked_panel(ax_right, rule_labels, rule_vio, rule_overall, rule_comp, COLOR_RULE, '%')

    # Subplot labels
    for ax, label in zip([ax_left, ax_right], ['a', 'b']):
        ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    # Legend with gray swatches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=1.0, label='violating recall'),
        Patch(facecolor='gray', alpha=0.55, label='accuracy'),
        Patch(facecolor='gray', alpha=0.2, label='compliant recall'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.subplots_adjust(left=0.15, right=0.98, top=0.94, bottom=0.09, wspace=0.32)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"cluster_stacked_{model}_{split}_{context}_{phrase_suffix}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print("Done: cluster stacked plot")
    return 0


# ============================================================================
# CORRELATION SCATTER PLOT
# ============================================================================

def plot_correlation(model, split, context, metric='overall_accuracy',
                     phrase='baseline', mode='prefill',
                     show_regression=False, cluster_split='all', cluster_type='rule'):
    """Correlation scatter plots: accuracy vs. cluster size metrics."""
    from adjustText import adjust_text

    perf_data = load_performance(model, split, context, phrase, mode)

    print(f"Loading cluster size stats: split={cluster_split}, type={cluster_type}")
    size_stats = load_cluster_size_stats(split=cluster_split, cluster_type=cluster_type)

    # Merge performance + size data
    per_cluster = perf_data['metrics'].get(f'per_{cluster_type}_cluster', {})
    cluster_size_data = size_stats['cluster_stats']
    merged = []

    for name, info in per_cluster.items():
        if metric not in info or name.lower() == 'other':
            continue
        acc = info[metric] * 100
        ci_key = f'{metric}_ci'
        if ci_key in info:
            ci_low, ci_high = info[ci_key]
            ci_low *= 100
            ci_high *= 100
        else:
            ci_low, ci_high = acc, acc
        if name not in cluster_size_data:
            print(f"  Warning: Cluster '{name}' not found in size stats, skipping")
            continue
        n_sub = cluster_size_data[name]['n_subreddits']
        n_rules = cluster_size_data[name]['n_rules']
        merged.append((name, acc, ci_low, ci_high, n_sub, n_rules))

    if not merged:
        print("No cluster data available")
        return 1

    print(f"  Merged {len(merged)} clusters (excluded 'Other')")

    names, accs, _, _, n_subs, n_rules = zip(*merged)
    accs = np.array(accs)
    n_subs = np.array(n_subs)
    n_rules = np.array(n_rules)

    # Axis limits with 5% padding
    acc_range = accs.max() - accs.min()
    xlim = (accs.min() - 0.05 * acc_range, accs.max() + 0.05 * acc_range)
    sub_range = n_subs.max() - n_subs.min()
    ylim_sub = (n_subs.min() - 0.05 * sub_range, n_subs.max() + 0.05 * sub_range)
    rules_range = n_rules.max() - n_rules.min()
    ylim_rules = (n_rules.min() - 0.05 * rules_range, n_rules.max() + 0.05 * rules_range)

    fig, (ax_left, ax_right) = create_two_column_figure(plot_type='barplot')

    # LEFT: Accuracy vs. # Subreddits
    ax_left.scatter(accs, n_subs, color=COLOR_SUBREDDIT, s=20, alpha=0.7, marker='o', zorder=3)
    texts_left = [ax_left.text(accs[i], n_subs[i], name, fontsize=4.5, alpha=0.8)
                  for i, name in enumerate(names)]
    adjust_text(texts_left, ax=ax_left,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    rho_sub, p_sub = stats.spearmanr(accs, n_subs)
    ax_left.text(0.05, 0.88, f'\u03c1 = {rho_sub:.3f}\np = {p_sub:.3f}',
                 transform=ax_left.transAxes, fontsize=7, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        slope, intercept, *_ = stats.linregress(accs, n_subs)
        x_line = np.linspace(accs.min(), accs.max(), 100)
        ax_left.plot(x_line, slope * x_line + intercept, color=COLOR_SUBREDDIT,
                     linewidth=1.5, alpha=0.6, linestyle='--', zorder=1)

    ax_left.set_xlabel('Accuracy (%)', fontsize=8)
    ax_left.set_ylabel('Number of Subreddits', fontsize=8)
    ax_left.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    style_clean_axis(ax_left, grid_axis='both')
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim_sub)

    # RIGHT: Accuracy vs. # Rules
    ax_right.scatter(accs, n_rules, color=COLOR_SUBREDDIT, s=20, alpha=0.7, marker='o', zorder=3)
    texts_right = [ax_right.text(accs[i], n_rules[i], name, fontsize=4.5, alpha=0.8)
                   for i, name in enumerate(names)]
    adjust_text(texts_right, ax=ax_right,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6))

    rho_rules, p_rules = stats.spearmanr(accs, n_rules)
    ax_right.text(0.05, 0.88, f'\u03c1 = {rho_rules:.3f}\np = {p_rules:.3f}',
                  transform=ax_right.transAxes, fontsize=7, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    if show_regression:
        slope, intercept, *_ = stats.linregress(accs, n_rules)
        x_line = np.linspace(accs.min(), accs.max(), 100)
        ax_right.plot(x_line, slope * x_line + intercept, color=COLOR_SUBREDDIT,
                      linewidth=1.5, alpha=0.6, linestyle='--', zorder=1)

    ax_right.set_xlabel('Accuracy (%)', fontsize=8)
    ax_right.set_ylabel('Number of Rules', fontsize=8)
    ax_right.axvline(x=50, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    style_clean_axis(ax_right, grid_axis='both')
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim_rules)

    # Subplot labels
    ax_left.text(0.02, 0.98, '(a)', transform=ax_left.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='left')
    ax_right.text(0.02, 0.98, '(b)', transform=ax_right.transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='left')

    fig.subplots_adjust(left=0.075, right=0.98, top=0.99, bottom=0.10, wspace=0.15)

    phrase_suffix = 'baseline' if phrase == 'baseline' else f'{phrase}_{mode}'
    filename = f"{cluster_type}_cluster_correlation_{model}_{split}_{context}_{phrase_suffix}_{metric}"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)

    print(f"Done: cluster correlation plot")
    print(f"  Accuracy vs. Subreddits: \u03c1 = {rho_sub:.3f}, p = {p_sub:.3f}")
    print(f"  Accuracy vs. Rules:      \u03c1 = {rho_rules:.3f}, p = {p_rules:.3f}")
    return 0


# ============================================================================
# LANGUAGE ANALYSIS DIVERGING PLOT
# ============================================================================

LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'French', 'de': 'German', 'pt': 'Portuguese',
    'es': 'Spanish', 'nl': 'Dutch', 'it': 'Italian', 'pl': 'Polish',
    'tr': 'Turkish', 'sv': 'Swedish', 'da': 'Danish', 'el': 'Greek',
    'uk': 'Ukrainian', 'ro': 'Romanian', 'eo': 'Esperanto', 'hu': 'Hungarian',
    'hr': 'Croatian', 'sk': 'Slovak', 'zh': 'Chinese', 'fi': 'Finnish',
    'cs': 'Czech', 'ru': 'Russian', 'no': 'Norwegian', 'sl': 'Slovenian'
}


def _load_language_distribution(use_cache=True):
    """Load language distribution from hydrated datasets across all splits."""
    import zstandard as zstd

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    cache_file = data_dir / 'language_distribution_stats.json'

    if use_cache and cache_file.exists():
        print(f"  Loading from cache: {cache_file.name}")
        with open(cache_file) as f:
            cached_data = json.load(f)
        return [(lang, count) for lang, count in cached_data['language_distribution']]

    print("  Computing language distribution from datasets...")
    language_counts = Counter()

    for split in ['train', 'val', 'test']:
        dataset_file = data_dir / f'{split}_hydrated_clustered.json.zst'
        with open(dataset_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                dataset = json.loads(reader.read())

        for sub_data in dataset['subreddits']:
            language = sub_data.get('language', 'unknown')
            normalized_lang = language.replace('_', '-').split('-')[0]
            n_pairs = len(sub_data['thread_pairs'])
            language_counts[normalized_lang] += n_pairs

    sorted_data = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)

    cache_data = {
        'language_distribution': sorted_data,
        'total_pairs': sum(language_counts.values()),
        'n_languages': len(language_counts)
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)
    print(f"  Saved cache to: {cache_file.name}")

    return sorted_data


def plot_language(model='gpt5.2-high'):
    """Diverging plot: distribution (left) + language labels (middle) + accuracy forest (right)."""
    print("Loading language distribution...")
    lang_distribution = _load_language_distribution()

    perf_data = load_performance(model, 'test', 'submission-media-discussion-user')
    per_language = perf_data['metrics'].get('per_language', {})
    if not per_language:
        print("No per_language metrics found in performance JSON")
        return 1

    # Merge distribution + accuracy
    language_data = {}
    for lang, count in lang_distribution:
        if lang in per_language:
            acc = per_language[lang]['overall_accuracy'] * 100
            ci_key = 'overall_accuracy_ci'
            if ci_key in per_language[lang]:
                ci_low, ci_high = per_language[lang][ci_key]
                ci_low *= 100
                ci_high *= 100
            else:
                ci_low, ci_high = acc, acc
            language_data[lang] = (count, acc, ci_low, ci_high)

    sorted_langs = [(lang, *language_data[lang])
                    for lang, _ in lang_distribution if lang in language_data]
    # Filter languages with < 10 instances
    sorted_langs = [(l, c, a, cl, ch) for l, c, a, cl, ch in sorted_langs if c >= 10]

    if not sorted_langs:
        print("No language data to plot")
        return 1

    languages, counts, accs, ci_lows, ci_highs = zip(*sorted_langs)
    language_labels = [LANGUAGE_NAMES.get(lang, lang) for lang in languages]

    fig, (ax_left, ax_right) = create_two_column_figure(figsize=(6.3, 1.7))
    y_pos = np.arange(len(languages))

    # LEFT: Distribution bars extending leftward
    ax_left.barh(y_pos, counts, height=0.8, color=COLOR_SUBREDDIT, edgecolor='none')
    ax_left.set_xlabel('Number of Instances', fontsize=8)
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels([])  # Labels go in the gap
    ax_left.tick_params(axis='y', length=0)
    ax_left.set_ylim(-0.45, len(languages) - 0.2)
    ax_left.invert_yaxis()
    ax_left.spines['left'].set_visible(False)
    style_clean_axis(ax_left, grid_axis='x', grid_width=1.0)
    ax_left.set_xscale('log')
    ax_left.set_xlim(left=10000, right=1)  # Reversed: bars extend leftward

    # RIGHT: Forest plot with accuracies and CIs
    for i in y_pos:
        ax_right.axhline(y=i, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

    ax_right.scatter(accs, y_pos, color=COLOR_SUBREDDIT, s=50, marker='s', zorder=3)
    for i, acc in enumerate(accs):
        ax_right.text(acc, i, f'{acc:.0f}', ha='center', va='center',
                      fontsize=4.5, color='white', fontweight='bold', zorder=4)
    for i, (cl, ch) in enumerate(zip(ci_lows, ci_highs)):
        ax_right.hlines(i, cl, ch, color=COLOR_SUBREDDIT, linewidth=1, zorder=2)
        ax_right.vlines([cl, ch], i - 0.25, i + 0.25, color=COLOR_SUBREDDIT, linewidth=1, zorder=2)

    ax_right.set_xlabel('Accuracy (%)', fontsize=8)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels([])
    ax_right.tick_params(axis='y', length=0)
    ax_right.set_ylim(-0.5, len(languages) - 0.5)
    ax_right.set_xlim(0, 100)
    ax_right.invert_yaxis()
    ax_right.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    style_clean_axis(ax_right, grid_axis='x')

    # Language labels centered in the gap
    for i, label_text in enumerate(language_labels):
        ax_left.text(1.125, i, label_text, transform=ax_left.get_yaxis_transform(),
                     fontsize=7, ha='center', va='center')

    # Subplot labels
    ax_left.text(0.02, 0.02, '(a)', transform=ax_left.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='left')
    ax_right.text(0.98, 0.02, '(b)', transform=ax_right.transAxes,
                  fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.2, wspace=0.25)

    filename = f"language_analysis_{model}_test_submission-media-discussion-user"
    plots_dir = get_eval_dir() / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, plots_dir / filename, dpi=PUBLICATION_DPI, bbox_inches=None)
    plt.close(fig)
    print(f"Done: language diverging plot")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures from evaluation results'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Shared args for plot commands
    def add_common_args(p):
        p.add_argument('--model', default='gpt5.2-high', help='Model name')
        p.add_argument('--split', default='test', help='Dataset split')
        p.add_argument('--context', default='submission-media-discussion-user', help='Context')
        p.add_argument('--phrase', default='baseline', help='Phrase type')
        p.add_argument('--mode', default='prefill', help='Mode')

    # forest
    p_forest = subparsers.add_parser('forest', help='Cluster forest plot (accuracy + 95%% CI)')
    add_common_args(p_forest)
    p_forest.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')

    # stacked
    p_stacked = subparsers.add_parser('stacked', help='Cluster stacked bar plot')
    add_common_args(p_stacked)

    # correlation
    p_corr = subparsers.add_parser('correlation', help='Cluster correlation scatter')
    add_common_args(p_corr)
    p_corr.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')
    p_corr.add_argument('--cluster-split', default='all', help='Split for cluster sizes')
    p_corr.add_argument('--cluster-type', default='rule', choices=['rule', 'subreddit'])
    p_corr.add_argument('--show-regression', action='store_true')

    # language
    p_lang = subparsers.add_parser('language', help='Language analysis diverging plot')
    p_lang.add_argument('--model', default='gpt5.2-high', help='Model name')

    # all
    p_all = subparsers.add_parser('all', help='Generate all figures')
    add_common_args(p_all)
    p_all.add_argument('--metric', default='overall_accuracy', help='Accuracy metric')

    args = parser.parse_args()

    if args.command == 'forest':
        return plot_forest(args.model, args.split, args.context, args.metric, args.phrase, args.mode)
    elif args.command == 'stacked':
        return plot_stacked(args.model, args.split, args.context, args.phrase, args.mode)
    elif args.command == 'correlation':
        return plot_correlation(args.model, args.split, args.context, args.metric,
                                args.phrase, args.mode,
                                show_regression=args.show_regression,
                                cluster_split=args.cluster_split,
                                cluster_type=args.cluster_type)
    elif args.command == 'language':
        return plot_language(args.model)
    elif args.command == 'all':
        results = []
        print("=" * 60)
        print("Generating all paper figures")
        print("=" * 60)

        print("\n--- Forest plot ---")
        results.append(plot_forest(args.model, args.split, args.context, args.metric, args.phrase, args.mode))

        print("\n--- Stacked plot ---")
        results.append(plot_stacked(args.model, args.split, args.context, args.phrase, args.mode))

        print("\n--- Correlation plot ---")
        results.append(plot_correlation(args.model, args.split, args.context, args.metric, args.phrase, args.mode))

        print("\n--- Language plot ---")
        results.append(plot_language(args.model))

        failures = sum(1 for r in results if r != 0)
        print(f"\n{'=' * 60}")
        print(f"Complete: {len(results) - failures}/{len(results)} succeeded")
        if failures:
            print(f"  {failures} failed")
        return 1 if failures else 0


if __name__ == '__main__':
    exit(main())
