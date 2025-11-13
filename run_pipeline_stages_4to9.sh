#!/bin/bash
# Pipeline stages 4-9: Rule matching, comment collection, tree building, submission collection, media collection, and dataset creation

set -e  # Exit on any error

echo "=================================="
echo "Running Pipeline Stages 4-9"
echo "=================================="
echo ""

# echo "Stage 4: Matching rules (phase 2 only)..."
# python scripts/4_match_rules.py --phase2-only
# echo "✅ Stage 4 complete"
# echo ""

# echo "Stage 5: Collecting submission comments..."
# python scripts/5_collect_submission_comments.py
# echo "✅ Stage 5 complete"
# echo ""

echo "Stage 6: Building trees and threads..."
python scripts/6_build_trees_and_threads.py
echo "✅ Stage 6 complete"
echo ""

echo "Stage 7: Collecting submissions..."
python scripts/7_collect_submissions.py
echo "✅ Stage 7 complete"
echo ""

echo "Stage 8: Collecting media..."
python scripts/8_collect_media.py
echo "✅ Stage 8 complete"
echo ""

echo "Stage 9: Creating final datasets..."
python scripts/9_create_dehydrated_dataset.py
echo "✅ Stage 9 complete"
echo ""

echo "=================================="
echo "Pipeline stages 4-9 completed successfully!"
echo "=================================="


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
    conda run -n reddit-mod-pipeline python analysis/embed_test_1k.py
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
    conda run -n reddit-mod-pipeline python analysis/cluster_test_1k.py
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
    conda run -n reddit-mod-pipeline python analysis/label_clusters.py
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
    conda run -n reddit-mod-pipeline python analysis/plot_clusters.py
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
echo "  - Subreddit Embeddings: output/embeddings/test_subreddit_embeddings.tsv"
echo "  - Rule Embeddings: output/embeddings/all_rule_embeddings.tsv"
echo "  - Metadata: output/embeddings/test_subreddit_metadata.tsv, output/embeddings/all_rule_metadata.tsv"
echo "  - Grid search: output/clustering/*_grid_search_results.json"
echo "  - Labels: output/clustering/*_cluster_labels.json"
echo "  - Analysis: output/clustering/*_cluster_analysis.txt"
echo "  - Plots: output/clustering/*_clusters_2d.png"
echo ""

