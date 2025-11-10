#!/usr/bin/env python3
"""
Plot Cluster Visualizations

Creates 2D visualizations of clustered embeddings with semantic labels.

Usage:
    python plot_clusters.py                         # Plot both subreddits and rules
    python plot_clusters.py --entity subreddit      # Plot only subreddits
    python plot_clusters.py --entity rule           # Plot only rules

Input:
- output/embeddings/test_subreddit_embeddings_reduced.tsv
- output/embeddings/test_subreddit_metadata.tsv (with cluster_id and cluster_label)
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
from scipy.spatial import ConvexHull

from adjustText import adjust_text
from coloring import color_polygons_improved


def apply_umap_2d(embeddings: np.ndarray, umap_params: dict, entity_type: str, logger) -> np.ndarray:
    """Apply UMAP reduction to 2D for visualization.

    Args:
        embeddings: Original embeddings to reduce
        umap_params: Dict with 'n_neighbors' and 'min_dist' from best params
        entity_type: 'subreddit' or 'rule'
        logger: Logger instance

    Returns:
        2D coordinates
    """
    # Use tighter settings for subreddit visualization
    min_dist = 1 if entity_type == 'subreddit' else 1
    n_neighbors = umap_params['n_neighbors'] * 2
    logger.info(f"Reducing to 2D with UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, n_jobs={PROCESSES})...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist,
                        metric='cosine', random_state=0, n_jobs=PROCESSES)
    coords_2d = reducer.fit_transform(embeddings)
    logger.info(f"  ✅ Reduced to shape {coords_2d.shape}")

    return coords_2d

def create_cluster_visualization(coords_2d: np.ndarray, metadata: pd.DataFrame, entity_type: str,
                                 clustering_dir: Path, logger) -> None:
    """Create and save 2D cluster visualization with spatially-coherent colors."""
    labels = metadata['cluster_id'].values

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

    # Compute convex hulls for each cluster
    logger.info(f"Computing convex hulls for {n_clusters} clusters...")
    cluster_polygons = []
    cluster_id_to_index = {}

    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_points = coords_2d[cluster_mask]

        # Need at least 3 points for a convex hull
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices].tolist()
                cluster_polygons.append(hull_vertices)
                cluster_id_to_index[cluster_id] = len(cluster_polygons) - 1
            except Exception as e:
                logger.warning(f"Could not compute hull for cluster {cluster_id}: {e}")
                # Fallback: use all points as polygon
                cluster_polygons.append(cluster_points.tolist())
                cluster_id_to_index[cluster_id] = len(cluster_polygons) - 1
        else:
            # For small clusters, use points directly
            cluster_polygons.append(cluster_points.tolist())
            cluster_id_to_index[cluster_id] = len(cluster_polygons) - 1

    # Get spatially-coherent colors using Paul Tol palette
    logger.info("Assigning spatially-coherent colors...")
    hex_colors = color_polygons_improved(cluster_polygons)

    # Create cluster_id -> color mapping
    cluster_colors = {}
    for cluster_id in unique_clusters:
        if cluster_id in cluster_id_to_index:
            idx = cluster_id_to_index[cluster_id]
            cluster_colors[cluster_id] = hex_colors[idx]

    # Map each point to its cluster color
    point_colors = np.array([cluster_colors.get(label, '#DDDDDD') for label in labels])

    # Create plot (Nature double-column: 180mm = 7.09 inches width, ~6 inch height)
    plt.figure(figsize=(7.09, 6))

    point_size = 160 if entity_type == 'subreddit' else 40
    # Plot noise points first (very faint)
    if n_noise > 0:
        plt.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1], c='lightgray', s=point_size, alpha=0.30)

    # Plot cluster points on top (more prominent colors)
    if entity_type == 'subreddit':
        plt.scatter(coords_2d[non_noise_mask, 0], coords_2d[non_noise_mask, 1],
                    c=point_colors[non_noise_mask], s=point_size, alpha=1.0, edgecolors='white', linewidths=0.2)
    else:
        plt.scatter(coords_2d[non_noise_mask, 0], coords_2d[non_noise_mask, 1],
                    c=point_colors[non_noise_mask], s=point_size, alpha=1.0, edgecolors='white', linewidths=0.1)

    # No title - goes in figure caption per Nature guidelines
    plt.xticks([])
    plt.yticks([])
    plt.box(False)  # Remove box/border

    # Add cluster labels as text annotations (all clusters)
    texts = []
    for cluster_id in sorted([c for c in set(labels) if c != -1]):
        cluster_mask = labels == cluster_id
        if cluster_mask.sum() > 0:
            centroid = coords_2d[cluster_mask].mean(axis=0)
            label_text = cluster_label_map.get(cluster_id, f'Cluster {cluster_id}')
            text = plt.text(centroid[0], centroid[1], f'{cluster_id}: {label_text}',
                          fontsize=5, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='black', linewidth=0.1))
            texts.append(text)

    # Adjust text positions to avoid overlap (no arrows)
    adjust_text(texts,
                expand_text=(1.2, 1.2),
                expand_points=(1.2, 1.2))

    # Save plot as PDF (editable in Illustrator)
    output_file_pdf = clustering_dir / f'{entity_type}_clusters_2d.pdf'
    plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', dpi=300)
    logger.info(f"  ✅ Saved PDF plot to: {output_file_pdf}")

    # Also save PNG for quick viewing (same DPI for consistency)
    output_file_png = clustering_dir / f'{entity_type}_clusters_2d.png'
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✅ Saved PNG plot to: {output_file_png}")


def plot_entity_type(entity_type: str, embeddings_dir: Path, clustering_dir: Path, logger) -> None:
    """Plot clusters for a single entity type (subreddit or rule)."""
    logger.info("\n" + "="*80)
    logger.info(f"{entity_type.upper()} CLUSTERS")
    logger.info("="*80)

    # Load metadata
    # Use 'all_rule' for rules (train/val/test), 'test_subreddit' for subreddits (test only)
    prefix = 'all_rule' if entity_type == 'rule' else 'test_subreddit'
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

    # Reduce to 2D from reduced embeddings
    coords_2d = apply_umap_2d(embeddings, umap_params, entity_type, logger)

    # Create visualization
    create_cluster_visualization(coords_2d, metadata, entity_type, clustering_dir, logger)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot cluster visualizations')
    parser.add_argument('--entity', choices=['subreddit', 'rule'], help='Plot only one entity type')
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
            plot_entity_type(entity_type, embeddings_dir, clustering_dir, logger)

        logger.info("\n" + "="*80)
        logger.info("COMPLETE")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
