#!/bin/bash

# Script to finetune all 20 N20 tasks using SGD optimizer
#
# Usage: bash scripts/run_fts_sgd.sh

set -e  # Exit on error

# Checkpoint directory
CKPT_PATH="/home/ubuntu/thesis/MM/Mergeability-Bench/checkpoints/ViT-B-16_SGD"



datasets=(
    "SUN397"
    "Cars"
    "RESISC45"
    "EuroSAT"
    "SVHN"
    "GTSRB"
    "MNIST"
    "DTD"
    "Flowers102"
    "PCAM"
    "FER2013"
    "OxfordIIITPet"
    "STL10"
    "CIFAR100"
    "CIFAR10"
    "Food101"
    "FashionMNIST"
    "EMNIST"
    "KMNIST"
    "RenderedSST2"
)



datasets1=(
    "SUN397"
    "Cars"
    "RESISC45"
    "EuroSAT"
    "SVHN"
    "GTSRB"
    "MNIST"
    "DTD"
    "Flowers102"
    "PCAM"
)


datasets2=(
    "FER2013"
    "OxfordIIITPet"
    "STL10"
    "CIFAR100"
    "CIFAR10"
    "Food101"
    "FashionMNIST"
    "EMNIST"
    "KMNIST"
    "RenderedSST2"
)

# Log file
LOG_DIR="/home/ubuntu/thesis/MM/Mergeability-Bench/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/finetune_sgd_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting SGD finetuning for all N20 tasks" | tee -a "$LOG_FILE"
echo "Checkpoint path: $CKPT_PATH" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Total number of datasets
TOTAL=${#datasets[@]}
CURRENT=0

# Track successes and failures
SUCCESS_COUNT=0
FAILURE_COUNT=0
SKIPPED_COUNT=0
FAILED_DATASETS=()
SKIPPED_DATASETS=()

# Finetune each dataset
for dataset in "${datasets[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Check if checkpoint already exists
    EXPECTED_CKPT="${CKPT_PATH}/${dataset}/model.pt"
    if [ -f "$EXPECTED_CKPT" ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "[$CURRENT/$TOTAL] SKIPPING: $dataset (checkpoint exists: $EXPECTED_CKPT)" | tee -a "$LOG_FILE"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        SKIPPED_DATASETS+=("$dataset")
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$CURRENT/$TOTAL] Finetuning: $dataset" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Run finetuning with SGD
    if uv run python scripts/finetune.py dataset=$dataset misc.ckpt_path=$CKPT_PATH 2>&1 | tee -a "$LOG_FILE"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "" | tee -a "$LOG_FILE"
        echo "Successfully completed: $dataset" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        FAILED_DATASETS+=("$dataset")
        echo "" | tee -a "$LOG_FILE"
        echo "FAILED: $dataset" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "Progress: $CURRENT/$TOTAL completed (Success: $SUCCESS_COUNT, Skipped: $SKIPPED_COUNT, Failed: $FAILURE_COUNT)" | tee -a "$LOG_FILE"
done

# Final summary
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Total datasets: $TOTAL" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "Skipped: $SKIPPED_COUNT" | tee -a "$LOG_FILE"
echo "Failed: $FAILURE_COUNT" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $SKIPPED_COUNT -gt 0 ]; then
    echo "Skipped datasets (checkpoint already exists):" | tee -a "$LOG_FILE"
    for skipped in "${SKIPPED_DATASETS[@]}"; do
        echo "  - $skipped" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
fi

if [ $FAILURE_COUNT -gt 0 ]; then
    echo "Failed datasets:" | tee -a "$LOG_FILE"
    for failed in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
fi

echo "Completion time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

if [ $FAILURE_COUNT -gt 0 ]; then
    exit 1
fi

echo ""
echo "All finetuning tasks completed successfully!"
