#!/usr/bin/env python3
"""
Plot Cluster Visualizations

Creates 2D visualizations of clustered embeddings with semantic labels.
Generates a single two-column figure with subreddit clusters (left) and rule clusters (right).

Usage:
    python plot_clusters.py                    # Plot both subreddits and rules (default)
    python plot_clusters.py --rotate 45       # Rotate color assignment by 45 degrees

Input:
- output/embeddings/all_subreddit_embeddings_reduced.tsv
- output/embeddings/all_subreddit_metadata.tsv (with cluster_id and cluster_label)
- output/embeddings/all_rule_embeddings_reduced.tsv
- output/embeddings/all_rule_metadata.tsv (with cluster_id and cluster_label)
- output/clustering/subreddit_grid_search_results.json (for UMAP params)
- output/clustering/rule_grid_search_results.json (for UMAP params)

Output:
- output/clustering/clusters_2d.pdf (ACL two-column format, editable)
- output/clustering/clusters_2d.png (quick preview)
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSES
from plotting_config import create_two_column_figure, save_figure, add_subplot_labels, FIGURE_HEIGHT_SCATTER, TWO_COLUMN_WIDTH
import umap

from adjustText import adjust_text
from coloring import assign_colors_with_conflicts
from scipy.spatial import ConvexHull
from collections import Counter
from config import PATHS


def rotate_coordinates(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate 2D coordinates by specified angle (in degrees)."""
    angle_rad = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return coords @ rotation_matrix.T


def load_cluster_distribution(cluster_type: str) -> list:
    """Load cluster distribution from stage10 stats.

    Args:
        cluster_type: 'subreddit' or 'rule'

    Returns:
        List of (name, count) tuples sorted by count descending
    """
    stats_file = Path(PATHS['data']) / 'stage10_cluster_assignment_stats.json'
    with open(stats_file) as f:
        data = json.load(f)

    all_stats = data.get('cluster_assignment_statistics', {})

    # Sum counts across all splits
    total_counts = Counter()
    for split, stats in all_stats.items():
        key = f'{cluster_type}_clusters'
        if key in stats:
            for label, count in stats[key].items():
                total_counts[label] += count

    # Sort by count descending, lowercase "Other" to "other"
    sorted_data = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_data = [('other' if l == 'Other' else l, c) for l, c in sorted_data]

    return sorted_data


def get_bar_colors(labels: list, color_json_path: Path) -> list:
    """Get bar colors from cluster color JSON file.

    Args:
        labels: List of cluster names
        color_json_path: Path to cluster colors JSON

    Returns:
        List of hex color strings
    """
    with open(color_json_path) as f:
        color_data = json.load(f)

    # Build name -> color map
    name_to_color = {}
    for cluster_id, info in color_data['clusters'].items():
        name_to_color[info['name']] = info['color']

    # Map labels to colors, default gray for "other"
    colors = []
    for label in labels:
        if label.lower() == 'other':
            colors.append('#DDDDDD')
        else:
            colors.append(name_to_color.get(label, '#DDDDDD'))
    return colors


def plot_distribution_bars(ax, labels: list, counts: list, colors: list, xlabel: str = 'Number of Instances'):
    """Plot horizontal distribution bars with cluster colors.

    Args:
        ax: Matplotlib axes
        labels: Cluster names
        counts: Thread pair counts
        colors: Hex colors for each bar
        xlabel: X-axis label
    """
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, counts, height=0.8, color=colors, edgecolor='none')
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis='x', labelsize=7, pad=0.5, length=3, width=0.25)
    ax.tick_params(axis='y', pad=0.5, length=3, width=0.25)
    ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=1.0)
    ax.set_ylim(-0.45, len(labels) - 0.2)
    ax.invert_yaxis()  # Highest values at top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.25)
    ax.spines['bottom'].set_linewidth(0.25)
    ax.set_xscale('log')
    ax.set_xlim(left=10, right=10000)


def apply_umap_2d(embeddings: np.ndarray, umap_params: dict, entity_type: str, logger, cache_dir: Path) -> np.ndarray:
    """Apply UMAP reduction to 2D for visualization with caching.

    Args:
        embeddings: Original embeddings to reduce
        umap_params: Dict with 'n_neighbors' and 'min_dist' from best params
        entity_type: 'subreddit' or 'rule'
        logger: Logger instance
        cache_dir: Directory to store cached results

    Returns:
        2D coordinates
    """
    min_dist = 2.0
    n_neighbors = umap_params['n_neighbors']

    # Cache file based on parameters (rotation applied after UMAP, so not in cache key)
    cache_file = cache_dir / f'{entity_type}_umap_2d_n{n_neighbors}_d{min_dist}_rs42.npy'

    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached 2D coordinates from {cache_file.name}...")
        coords_2d = np.load(cache_file)
        logger.info(f"  ✅ Loaded shape {coords_2d.shape}")
        return coords_2d

    # Compute UMAP
    logger.info(f"Reducing to 2D with UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_jobs={PROCESSES})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, spread=2.0,
                        metric='cosine', random_state=42, n_jobs=PROCESSES)
    coords_2d = reducer.fit_transform(embeddings)
    logger.info(f"  ✅ Reduced to shape {coords_2d.shape}")

    # Save to cache
    np.save(cache_file, coords_2d)
    logger.info(f"  💾 Cached to {cache_file.name}")

    return coords_2d

def plot_cluster_on_axes(ax, coords_2d: np.ndarray, metadata: pd.DataFrame, entity_type: str,
                         clustering_dir: Path, logger, rotation_angle: float = 0) -> None:
    """Plot 2D cluster visualization on given axes with spatially-coherent colors.

    Args:
        ax: Matplotlib axes to plot on
        coords_2d: 2D coordinates
        metadata: Cluster metadata
        entity_type: 'subreddit' or 'rule'
        clustering_dir: Output directory (for saving color JSON)
        logger: Logger instance
        rotation_angle: Rotation angle in degrees (default: 0)
    """
    labels = metadata['cluster_id'].values

    # Rotate coordinates if specified (affects color assignment based on spatial position)
    if rotation_angle != 0:
        logger.info(f"Rotating {entity_type} coordinates by {rotation_angle}° (affects color assignment)...")
        coords_2d = rotate_coordinates(coords_2d, rotation_angle)

    # Get cluster labels if available
    if 'cluster_label' in metadata.columns:
        cluster_label_map = metadata.groupby('cluster_id')['cluster_label'].first().to_dict()
    else:
        cluster_label_map = {}

    # Cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    logger.info(f"  {entity_type.upper()}: {n_clusters} clusters, {n_noise} noise points")

    # Separate noise and non-noise points
    noise_mask = labels == -1
    non_noise_mask = ~noise_mask

    # Get unique cluster IDs (excluding noise)
    unique_clusters = sorted([c for c in set(labels) if c != -1])

    # Calculate median centroids for all clusters (for color assignment and labels)
    median_centroids = {}
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_points = coords_2d[cluster_mask]
        if len(cluster_points) > 0:
            median_centroid = np.median(cluster_points, axis=0)
            median_centroids[cluster_id] = tuple(median_centroid)

    # Compute convex hulls for conflict detection
    cluster_polygons = []
    cluster_id_list = []
    for cluster_id in unique_clusters:
        if cluster_id in median_centroids:
            cluster_mask = labels == cluster_id
            cluster_points = coords_2d[cluster_mask]

            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_vertices = cluster_points[hull.vertices].tolist()
                    cluster_polygons.append(hull_vertices)
                    cluster_id_list.append(cluster_id)
                except Exception as e:
                    logger.debug(f"Could not compute hull for {entity_type} cluster {cluster_id}: {e}")
                    cluster_polygons.append(cluster_points.tolist())
                    cluster_id_list.append(cluster_id)
            else:
                cluster_polygons.append(cluster_points.tolist())
                cluster_id_list.append(cluster_id)

    # Prepare centroid data for color assignment
    color_assignment_data = [median_centroids[cid] for cid in cluster_id_list]

    # Get spatially-coherent colors with conflict resolution
    hex_colors = assign_colors_with_conflicts(color_assignment_data, cluster_polygons)

    # Create cluster_id -> color mapping
    cluster_colors = {}
    for i, cluster_id in enumerate(cluster_id_list):
        cluster_colors[cluster_id] = hex_colors[i]

    # Save cluster colors to JSON for reuse (with cluster names)
    color_file = clustering_dir / f'{entity_type}_cluster_colors.json'
    clusters_with_names = {}
    for cluster_id, color in cluster_colors.items():
        cluster_name = cluster_label_map.get(cluster_id, f'Cluster {cluster_id}')
        clusters_with_names[int(cluster_id)] = {
            'color': color,
            'name': cluster_name
        }

    color_data = {
        'entity_type': entity_type,
        'rotation_angle': rotation_angle,
        'clusters': clusters_with_names,
        'color_palette': 'rainbow_PuBr',
        'assignment_method': 'median_centroid_x_position'
    }
    with open(color_file, 'w') as f:
        json.dump(color_data, f, indent=2)
    logger.debug(f"  💾 Saved cluster colors to {color_file.name}")

    # Map each point to its cluster color
    point_colors = np.array([cluster_colors.get(label, '#DDDDDD') for label in labels])

    # Plot noise points first (very faint)
    point_size = 50  # Larger dots for better visibility
    if n_noise > 0:
        ax.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1], c='lightgray', s=point_size, alpha=0.15)

    # Plot cluster points on top (more prominent colors)
    ax.scatter(coords_2d[non_noise_mask, 0], coords_2d[non_noise_mask, 1],
               c=point_colors[non_noise_mask], s=point_size, alpha=1.0, edgecolors='white', linewidths=0.15)

    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add cluster labels as text annotations (all clusters)
    # Use median centroids (same as color assignment)
    texts = []
    for cluster_id in sorted([c for c in set(labels) if c != -1]):
        if cluster_id in median_centroids:
            centroid = median_centroids[cluster_id]
            label_text = cluster_label_map.get(cluster_id, f'Cluster {cluster_id}')
            # Remove number prefix - just show label text
            text = ax.text(centroid[0], centroid[1], f'{label_text}',
                          fontsize=6, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.025))
            texts.append(text)

    # Adjust text positions to avoid overlap (very gentle adjustment)
    if texts:
        adjust_text(texts,
                    expand_text=(0.95, 0.95),  # Allow slight overlap
                    lim=50,  # Very few iterations
                    force_text=(0.005, 0.005),  # Extremely weak text-text repulsion
                    autoalign=False,  # Don't auto-align non-overlapping labels
                    only_move={'text': 'xy'},  # Only move text if needed
                    avoid_self=False,  # Don't move to avoid own anchor point
                    ax=ax)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot cluster visualizations (two-column ACL format)')
    parser.add_argument('--rotate', type=float, default=-45, help='Rotation angle in degrees (affects color assignment, default: -45)')
    parser.add_argument('--grey-bars', action='store_true', help='Use grey bars instead of cluster colors')
    args = parser.parse_args()

    # Create directories
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / 'logs' / 'clustering'
    embeddings_dir = base_dir / 'output' / 'embeddings'
    clustering_dir = base_dir / 'output' / 'clustering'

    logs_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'plot_clusters_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(str(log_file)), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    try:
        logger.info("="*80)
        logger.info("CLUSTER VISUALIZATION (2x2 Layout with Distribution)")
        logger.info("="*80)

        # Create 2x2 figure with 2/3 vs 1/3 width ratio
        logger.info(f"\nCreating 2x2 figure ({TWO_COLUMN_WIDTH}\" x {FIGURE_HEIGHT_SCATTER}\") with 2:1 width ratio...")
        fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, FIGURE_HEIGHT_SCATTER))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], wspace=0, hspace=0.025)

        # Create axes with custom width distribution
        ax_sub_scatter = fig.add_subplot(gs[0, 0])   # Top-left: 2/3 width
        ax_sub_bars = fig.add_subplot(gs[0, 1])      # Top-right: 1/3 width
        ax_rule_scatter = fig.add_subplot(gs[1, 0])  # Bottom-left: 2/3 width
        ax_rule_bars = fig.add_subplot(gs[1, 1])     # Bottom-right: 1/3 width

        # Load and plot SUBREDDIT clusters (top-left)
        logger.info("\nLoading subreddit data...")
        sub_metadata = pd.read_csv(embeddings_dir / 'all_subreddit_metadata.tsv', sep='\t')
        sub_embeddings = np.loadtxt(embeddings_dir / 'all_subreddit_embeddings_reduced.tsv', delimiter='\t')
        with open(clustering_dir / 'subreddit_grid_search_results.json') as f:
            sub_params = json.load(f)['best_params']['umap']
        sub_umap_params = {'n_neighbors': sub_params['n_neighbors'], 'min_dist': sub_params['min_dist']}
        sub_coords = apply_umap_2d(sub_embeddings, sub_umap_params, 'subreddit', logger, clustering_dir)
        logger.info("Plotting subreddit clusters (top-left)...")
        plot_cluster_on_axes(ax_sub_scatter, sub_coords, sub_metadata, 'subreddit', clustering_dir, logger, args.rotate)

        # Plot SUBREDDIT distribution bars (top-right)
        logger.info("Plotting subreddit distribution (top-right)...")
        sub_dist = load_cluster_distribution('subreddit')
        sub_labels, sub_counts = zip(*sub_dist)
        if args.grey_bars:
            sub_colors = ['#888888'] * len(sub_labels)
        else:
            sub_colors = get_bar_colors(sub_labels, clustering_dir / 'subreddit_cluster_colors.json')
        plot_distribution_bars(ax_sub_bars, sub_labels, sub_counts, sub_colors)

        # Load and plot RULE clusters (bottom-left)
        logger.info("\nLoading rule data...")
        rule_metadata = pd.read_csv(embeddings_dir / 'all_rule_metadata.tsv', sep='\t')
        rule_embeddings = np.loadtxt(embeddings_dir / 'all_rule_embeddings_reduced.tsv', delimiter='\t')
        with open(clustering_dir / 'rule_grid_search_results.json') as f:
            rule_params = json.load(f)['best_params']['umap']
        rule_umap_params = {'n_neighbors': rule_params['n_neighbors'], 'min_dist': rule_params['min_dist']}
        rule_coords = apply_umap_2d(rule_embeddings, rule_umap_params, 'rule', logger, clustering_dir)
        logger.info("Plotting rule clusters (bottom-left)...")
        plot_cluster_on_axes(ax_rule_scatter, rule_coords, rule_metadata, 'rule', clustering_dir, logger, args.rotate)

        # Plot RULE distribution bars (bottom-right)
        logger.info("Plotting rule distribution (bottom-right)...")
        rule_dist = load_cluster_distribution('rule')
        rule_labels, rule_counts = zip(*rule_dist)
        if args.grey_bars:
            rule_colors = ['#888888'] * len(rule_labels)
        else:
            rule_colors = get_bar_colors(rule_labels, clustering_dir / 'rule_cluster_colors.json')
        plot_distribution_bars(ax_rule_bars, rule_labels, rule_counts, rule_colors)

        # Add subplot labels (a), (b), (c), (d) in bottom right corners
        # Order: (a) top-left scatter, (b) bottom-left scatter, (c) top-right bars, (d) bottom-right bars
        logger.info("Adding subplot labels...")
        # Bar plots use axes transform
        for ax, label in zip([ax_sub_bars, ax_rule_bars], ['c', 'd']):
            ax.text(0.98, 0.02, f'({label})', transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right')
        # Scatter plots use axes transform at same position
        # After bar adjustment (y0 + 0.045, height - 0.05), calculate matching position
        # Bar plot height after adjustment: 0.5 - 0.05 = 0.45
        # Bar plot at 2% = 0.45 * 0.02 = 0.009
        # Scatter plot height: 0.5, so 0.009 / 0.5 = 0.018 (1.8% instead of 2%)
        # But we need to account for the upward shift of 0.045 in the bar plots
        # Match the bar plot's absolute position: 0.045 + 0.009 for bottom, 0.545 + 0.009 for top
        # For scatter: (0.045 + 0.009) / 0.5 = 0.108 for bottom, (0.545 + 0.009 - 0.5) / 0.5 = 0.108 for top
        for ax, label in zip([ax_sub_scatter, ax_rule_scatter], ['a', 'b']):
            ax.text(0.96, 0.108, f'({label})', transform=ax.transAxes,
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right')

        # Save combined figure (no gap between subplots)
        logger.info("\nSaving figure...")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.025)

        # Inward adjustment for bar plots to create spacing for axis labels
        for ax_bars in [ax_sub_bars, ax_rule_bars]:
            pos = ax_bars.get_position()
            ax_bars.set_position([pos.x0 + 0.10, pos.y0 + 0.05, pos.width - 0.115, pos.height - 0.05])

        # Add horizontal line between top and bottom panels
        from matplotlib.lines import Line2D
        line = Line2D([0, 1], [0.5, 0.5], transform=fig.transFigure,
                      color='black', linestyle='-', linewidth=0.25, alpha=0.2)
        fig.add_artist(line)

        output_suffix = '_grey' if args.grey_bars else ''
        output_base = clustering_dir / f'clusters_2d{output_suffix}'
        save_figure(fig, output_base, dpi=300, bbox_inches=None)

        plt.close(fig)

        logger.info("\n" + "="*80)
        logger.info("✅ COMPLETE")
        logger.info("="*80)
        logger.info(f"Output: {output_base}.pdf")
        logger.info(f"        {output_base}.png")

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
