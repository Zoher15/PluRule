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

echo "Stage 5: Collecting submission comments..."
python scripts/5_collect_submission_comments.py
echo "✅ Stage 5 complete"
echo ""

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
