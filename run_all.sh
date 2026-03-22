#!/usr/bin/env bash
# run_all.sh — One-click: Train all models → Evaluate → Compare
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════"
echo "  Step 1/8: Train TFT (32 seeds × 4 info sets)"
echo "═══════════════════════════════════════════════════════════"
python train.py --config config.json --model tft

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2/8: Train LSTM baseline (32 seeds × 4 info sets)"
echo "═══════════════════════════════════════════════════════════"
python train.py --config config.json --model lstm

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 3/8: Train traditional models (OLS, EN, PCA, PLS, RF, GBT)"
echo "═══════════════════════════════════════════════════════════"
python train_traditional.py --config config.json

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 4/8: Evaluate TFT"
echo "═══════════════════════════════════════════════════════════"
python evaluate.py --config config.json --model tft

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 5/8: Evaluate LSTM"
echo "═══════════════════════════════════════════════════════════"
python evaluate.py --config config.json --model lstm

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 6/8: Evaluate traditional models"
echo "═══════════════════════════════════════════════════════════"
for model in ols elasticnet pca_regression pls random_forest gradient_boosting; do
    echo "  --- $model ---"
    python evaluate.py --config config.json --model $model
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 7/8: Cross-model comparison"
echo "═══════════════════════════════════════════════════════════"
python evaluate.py --config config.json --compare-all

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 8/8: Done! Results in ./outputs/"
echo "═══════════════════════════════════════════════════════════"
