#!/usr/bin/env python3
"""
Plot Cluster Visualizations

Creates 2D visualizations of clustered embeddings with semantic labels.

Usage:
    python plot_clusters.py                         # Plot both subreddits and rules
    python plot_clusters.py --entity subreddit      # Plot only subreddits
    python plot_clusters.py --entity rule           # Plot only rules

Input:
- output/embeddings/all_subreddit_embeddings_reduced.tsv
- output/embeddings/all_subreddit_metadata.tsv (with cluster_id and cluster_label)
- output/embeddings/all_rule_embeddings_reduced.tsv
- output/embeddings/all_rule_metadata.tsv (with cluster_id and cluster_label)
- output/clustering/subreddit_grid_search_results.json (for UMAP params)
- output/clustering/rule_grid_search_results.json (for UMAP params)

Output:
- output/clustering/subreddit_clusters_2d.png
- output/clustering/rule_clusters_2d.png
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime
import argparse

# Configure matplotlib for editable PDFs
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts (editable in Illustrator)
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts for EPS as well

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSES
import umap

from adjustText import adjust_text
from coloring import assign_colors_by_position


def rotate_coordinates(coords: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate 2D coordinates by specified angle (in degrees)."""
    angle_rad = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return coords @ rotation_matrix.T


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
    min_dist = umap_params['min_dist']
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
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist,
                        metric='cosine', random_state=42, n_jobs=PROCESSES)
    coords_2d = reducer.fit_transform(embeddings)
    logger.info(f"  ✅ Reduced to shape {coords_2d.shape}")

    # Save to cache
    np.save(cache_file, coords_2d)
    logger.info(f"  💾 Cached to {cache_file.name}")

    return coords_2d

def create_cluster_visualization(coords_2d: np.ndarray, metadata: pd.DataFrame, entity_type: str,
                                 clustering_dir: Path, logger, rotation_angle: float = 0) -> None:
    """Create and save 2D cluster visualization with spatially-coherent colors.

    Args:
        coords_2d: 2D coordinates
        metadata: Cluster metadata
        entity_type: 'subreddit' or 'rule'
        clustering_dir: Output directory
        logger: Logger instance
        rotation_angle: Rotation angle in degrees (default: 0)
    """
    labels = metadata['cluster_id'].values

    # Rotate coordinates if specified (affects color assignment based on spatial position)
    if rotation_angle != 0:
        logger.info(f"Rotating coordinates by {rotation_angle}° (affects color assignment)...")
        coords_2d = rotate_coordinates(coords_2d, rotation_angle)

    # Get cluster labels if available
    if 'cluster_label' in metadata.columns:
        cluster_label_map = metadata.groupby('cluster_id')['cluster_label'].first().to_dict()
    else:
        cluster_label_map = {}

    # Cluster statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    logger.info(f"Creating 2D plot ({n_clusters} clusters, {n_noise} noise points shown faintly)...")
    logger.info(f"Using Paul Tol's spatially-coherent rainbow_discrete colors")

    # Separate noise and non-noise points
    noise_mask = labels == -1
    non_noise_mask = ~noise_mask

    # Get unique cluster IDs (excluding noise)
    unique_clusters = sorted([c for c in set(labels) if c != -1])

    # Calculate median centroids for all clusters (for color assignment and labels)
    logger.info("Calculating median centroids...")
    median_centroids = {}
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_points = coords_2d[cluster_mask]
        if len(cluster_points) > 0:
            median_centroid = np.median(cluster_points, axis=0)
            median_centroids[cluster_id] = tuple(median_centroid)

    # Prepare data for color assignment (use median centroids)
    color_assignment_data = []
    cluster_id_list = []
    for cluster_id in unique_clusters:
        if cluster_id in median_centroids:
            color_assignment_data.append(median_centroids[cluster_id])
            cluster_id_list.append(cluster_id)

    # Get spatially-coherent colors using Paul Tol palette
    logger.info("Assigning spatially-coherent colors...")
    hex_colors = assign_colors_by_position(color_assignment_data)

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
    logger.info(f"  💾 Saved cluster colors to {color_file.name}")

    # Map each point to its cluster color
    point_colors = np.array([cluster_colors.get(label, '#DDDDDD') for label in labels])

    # Create plot (Nature double-column: 180mm = 7.09 inches width, ~6 inch height)
    plt.figure(figsize=(7.09, 6))

    point_size = 300 if entity_type == 'subreddit' else 150
    # Plot noise points first (very faint)
    if n_noise > 0:
        plt.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1], c='lightgray', s=point_size, alpha=0.10)

    # Plot cluster points on top (more prominent colors)
    if entity_type == 'subreddit':
        plt.scatter(coords_2d[non_noise_mask, 0], coords_2d[non_noise_mask, 1],
                    c=point_colors[non_noise_mask], s=point_size, alpha=1.0, edgecolors='white', linewidths=0.15)
    else:
        plt.scatter(coords_2d[non_noise_mask, 0], coords_2d[non_noise_mask, 1],
                    c=point_colors[non_noise_mask], s=point_size, alpha=1.0, edgecolors='white', linewidths=0.1)

    # No title - goes in figure caption per Nature guidelines
    plt.xticks([])
    plt.yticks([])
    plt.box(False)  # Remove box/border

    # Add cluster labels as text annotations (all clusters)
    # Use median centroids (same as color assignment)
    texts = []
    for cluster_id in sorted([c for c in set(labels) if c != -1]):
        if cluster_id in median_centroids:
            centroid = median_centroids[cluster_id]
            label_text = cluster_label_map.get(cluster_id, f'Cluster {cluster_id}')
            # Display cluster ID starting from 1 instead of 0
            display_id = cluster_id + 1
            text = plt.text(centroid[0], centroid[1], f'{display_id}: {label_text}',
                          fontsize=6, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='black', linewidth=0.1))
            texts.append(text)

    # Adjust text positions to avoid overlap (very gentle adjustment)
    # Only prevent text-text conflicts, don't move labels unnecessarily
    adjust_text(texts,
                expand_text=(0.95, 0.95),  # Allow slight overlap
                lim=50,  # Very few iterations
                force_text=(0.005, 0.005),  # Extremely weak text-text repulsion
                autoalign=False,  # Don't auto-align non-overlapping labels
                only_move={'text': 'xy'},  # Only move text if needed
                avoid_self=False)  # Don't move to avoid own anchor point

    # Save plot as PDF (editable in Illustrator)
    output_file_pdf = clustering_dir / f'{entity_type}_clusters_2d.pdf'
    plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', dpi=300)
    logger.info(f"  ✅ Saved PDF plot to: {output_file_pdf}")

    # Also save PNG for quick viewing (same DPI for consistency)
    output_file_png = clustering_dir / f'{entity_type}_clusters_2d.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✅ Saved PNG plot to: {output_file_png}")


def plot_entity_type(entity_type: str, embeddings_dir: Path, clustering_dir: Path, logger, rotation_angle: float = 0) -> None:
    """Plot clusters for a single entity type (subreddit or rule).

    Args:
        entity_type: 'subreddit' or 'rule'
        embeddings_dir: Path to embeddings directory
        clustering_dir: Path to clustering output directory
        logger: Logger instance
        rotation_angle: Rotation angle in degrees (default: 0)
    """
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} CLUSTERS")
    logger.info("="*80)

    # Load metadata
    # Use 'all_rule' for rules, 'all_subreddit' for subreddits (both from train/val/test)
    prefix = 'all_rule' if entity_type == 'rule' else 'all_subreddit'
    metadata_file = embeddings_dir / f'{prefix}_metadata.tsv'
    logger.info(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file, sep='\t')

    # Load reduced embeddings
    reduced_file = embeddings_dir / f'{prefix}_embeddings_reduced.tsv'
    logger.info(f"Loading reduced embeddings from {reduced_file}...")
    embeddings = np.loadtxt(reduced_file, delimiter='\t')
    logger.info(f"  Loaded shape {embeddings.shape}")

    # Load UMAP params from grid search results
    grid_search_file = clustering_dir / f'{entity_type}_grid_search_results.json'
    with open(grid_search_file) as f:
        best_params = json.load(f)['best_params']['umap']
    umap_params = {'n_neighbors': best_params['n_neighbors'], 'min_dist': best_params['min_dist']}

    # Reduce to 2D from reduced embeddings (with caching)
    coords_2d = apply_umap_2d(embeddings, umap_params, entity_type, logger, clustering_dir)

    # Create visualization
    create_cluster_visualization(coords_2d, metadata, entity_type, clustering_dir, logger, rotation_angle)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot cluster visualizations')
    parser.add_argument('--entity', choices=['subreddit', 'rule'], help='Plot only one entity type')
    parser.add_argument('--rotate', type=float, default=0, help='Rotation angle in degrees (affects color assignment, default: 0)')
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
        # Determine entity types to process
        entity_types = [args.entity] if args.entity else ['subreddit', 'rule']

        # Plot each entity type
        for entity_type in entity_types:
            plot_entity_type(entity_type, embeddings_dir, clustering_dir, logger, args.rotate)

        logger.info("\n" + "="*80)
        logger.info("COMPLETE")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
