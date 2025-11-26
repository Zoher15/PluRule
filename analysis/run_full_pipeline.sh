#!/bin/bash
# Full Analysis Pipeline: Embed -> Cluster -> Label -> Plot
#
# Usage:
#   ./run_full_pipeline.sh              # Run full pipeline
#   ./run_full_pipeline.sh --skip-embed # Skip embedding step
#   ./run_full_pipeline.sh --skip-cluster # Skip clustering step

set -e  # Exit on any error

# Parse arguments
SKIP_EMBED=false
SKIP_CLUSTER=false
SKIP_LABEL=false
SKIP_PLOT=false

for arg in "$@"; do
    case $arg in
        --skip-embed)
            SKIP_EMBED=true
            ;;
        --skip-cluster)
            SKIP_CLUSTER=true
            ;;
        --skip-label)
            SKIP_LABEL=true
            ;;
        --skip-plot)
            SKIP_PLOT=true
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "REDDIT MOD COLLECTION PIPELINE - FULL ANALYSIS"
echo "================================================================================"
echo ""

# Step 1: Embed
if [ "$SKIP_EMBED" = false ]; then
    echo "================================================================================"
    echo "STEP 1/4: EMBEDDING"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python embed_test_1k.py
    if [ $? -ne 0 ]; then
        echo "❌ Embedding failed!"
        exit 1
    fi
    echo "✅ Embedding complete"
    echo ""
else
    echo "⏭️  Skipping embedding step"
    echo ""
fi

# Step 2: Cluster (grid search + apply best)
if [ "$SKIP_CLUSTER" = false ]; then
    echo "================================================================================"
    echo "STEP 2/4: CLUSTERING (Grid Search + Apply Best)"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python cluster_test_1k.py
    if [ $? -ne 0 ]; then
        echo "❌ Clustering failed!"
        exit 1
    fi
    echo "✅ Clustering complete"
    echo ""
else
    echo "⏭️  Skipping clustering step"
    echo ""
fi

# Step 3: Label clusters
if [ "$SKIP_LABEL" = false ]; then
    echo "================================================================================"
    echo "STEP 3/4: LABELING CLUSTERS"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python label_clusters.py
    if [ $? -ne 0 ]; then
        echo "❌ Labeling failed!"
        exit 1
    fi
    echo "✅ Labeling complete"
    echo ""
else
    echo "⏭️  Skipping labeling step"
    echo ""
fi

# Step 4: Plot
if [ "$SKIP_PLOT" = false ]; then
    echo "================================================================================"
    echo "STEP 4/4: PLOTTING"
    echo "================================================================================"
    conda run -n reddit-mod-pipeline python plot_clusters.py
    if [ $? -ne 0 ]; then
        echo "❌ Plotting failed!"
        exit 1
    fi
    echo "✅ Plotting complete"
    echo ""
else
    echo "⏭️  Skipping plotting step"
    echo ""
fi

echo "================================================================================"
echo "✅ FULL PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Outputs:"
echo "  - Subreddit Embeddings: output/embeddings/all_subreddit_embeddings.tsv"
echo "  - Rule Embeddings: output/embeddings/all_rule_embeddings.tsv"
echo "  - Metadata: output/embeddings/all_subreddit_metadata.tsv, output/embeddings/all_rule_metadata.tsv"
echo "  - Grid search: output/clustering/*_grid_search_results.json"
echo "  - Labels: output/clustering/*_cluster_labels.json"
echo "  - Analysis: output/clustering/*_cluster_analysis.txt"
echo "  - Plots: output/clustering/*_clusters_2d.png"
echo ""
