#!/bin/bash
# Script to run AGM (Adaptive Gradient Modulator) implementation
# This script trains models with different optimizers enhanced with AGM
# and saves results for comparison with non-AGM versions

echo "=========================================="
echo "Running AGM Implementation"
echo "=========================================="

# Default parameters
PATIENCE=${1:-5}
MAX_EPOCHS=${2:-100}
BATCH_SIZE=${3:-128}
NUM_RUNS=${4:-5}
RESULTS_DIR=${5:-./agm_results}

# Run the AGM implementation script
python agm_implementation.py \
    --patience $PATIENCE \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \
    --results_dir $RESULTS_DIR

echo ""
echo "=========================================="
echo "AGM implementation completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

