#!/bin/bash
# Script to run optimizer comparison without AGM
# This script trains models with different optimizers (SGD, RMSProp, AdaGrad, Adam, AdamW, SGD+Momentum)
# and saves results for comparison

echo "=========================================="
echo "Running Optimizer Comparison (without AGM)"
echo "=========================================="

# Default parameters
PATIENCE=${1:-5}
MAX_EPOCHS=${2:-100}
BATCH_SIZE=${3:-128}
NUM_RUNS=${4:-5}
RESULTS_DIR=${5:-./results}

# Run the comparison script
python compare_optimizers.py \
    --patience $PATIENCE \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \
    --results_dir $RESULTS_DIR

echo ""
echo "=========================================="
echo "Optimizer comparison completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

