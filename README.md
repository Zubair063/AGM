# Adaptive Gradient Modulator (AGM) for Deep Learning Optimizers

**AGM: Adaptive Gradient Modulation for Faster Convergence and Deep Learning Optimization**

This repository contains the implementation of the Adaptive Gradient Modulator (AGM) and comprehensive comparisons with standard optimizers on the MNIST dataset.

## Overview

The Adaptive Gradient Modulator (AGM) is a gradient modification technique that adaptively adjusts gradients during training to improve optimization performance. This project compares AGM-enhanced optimizers against their standard counterparts across multiple optimizers: SGD, RMSProp, AdaGrad, Adam, AdamW, and SGD+Momentum.

## Repository Structure

```
AGM/
├── agm_implementation.py      # AGM implementation with optimizer comparisons
├── compare_optimizers.py       # Standard optimizer comparison (without AGM)
├── run_compare_optimizers.sh    # Bash script to run standard optimizer comparison
├── run_agm_implementation.sh   # Bash script to run AGM implementation
├── data/                       # MNIST dataset directory (auto-downloaded)
└── README.md                   # This file
```

## Features

- **AGM Implementation**: Adaptive gradient modulation with configurable hyperparameters
- **Multiple Optimizers**: Support for SGD, RMSProp, AdaGrad, Adam, AdamW, and SGD+Momentum
- **Reproducible Experiments**: Multiple runs with different random seeds for statistical significance
- **Comprehensive Metrics**: Tracks accuracy, F1 score, AUC, and test loss
- **Early Stopping**: Configurable patience for early stopping
- **Detailed Results**: Saves training history, model checkpoints, and aggregated statistics

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=0.24.0
tqdm>=4.62.0
```

Install dependencies:
```bash
pip install torch torchvision numpy scikit-learn tqdm
```

## Quick Start

### 1. Run Standard Optimizer Comparison

```bash
chmod +x run_compare_optimizers.sh
./run_compare_optimizers.sh
```

Or with custom parameters:
```bash
./run_compare_optimizers.sh [patience] [max_epochs] [batch_size] [num_runs] [results_dir]
```

Example:
```bash
./run_compare_optimizers.sh 5 100 128 5 ./results
```

### 2. Run AGM Implementation

```bash
chmod +x run_agm_implementation.sh
./run_agm_implementation.sh
```

Or with custom parameters:
```bash
./run_agm_implementation.sh [patience] [max_epochs] [batch_size] [num_runs] [results_dir]
```

Example:
```bash
./run_agm_implementation.sh 5 100 128 5 ./agm_results
```

### 3. Run with Python Directly

**Standard Optimizers:**
```bash
python compare_optimizers.py --patience 5 --max_epochs 100 --batch_size 128 --num_runs 5
```

**AGM Implementation:**
```bash
python agm_implementation.py --patience 5 --max_epochs 100 --batch_size 128 --num_runs 5
```

## Command Line Arguments

Both scripts support the following arguments:

- `--patience`: Early stopping patience (default: 5)
- `--max_epochs`: Maximum number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 128)
- `--num_runs`: Number of runs per optimizer with different seeds (default: 5)
- `--results_dir`: Directory to save results (default: `./results` for compare_optimizers.py, `./agm_results` for agm_implementation.py)
- `--seeds`: Custom random seeds (optional, default: [42, 123, 456, 789, 1011])

## Model Architecture

The implementation uses a simple Multi-Layer Perceptron (MLP) for MNIST classification:

- **Input Layer**: 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (10 classes)

## AGM Hyperparameters

The Adaptive Gradient Modulator uses the following default hyperparameters:

- `beta = 0.9`: EMA decay factor for gradient norm
- `alpha = 0.5`: Magnitude scaling exponent
- `lam = 0.7`: Gradient smoothing factor
- `kappa = 2.0`: Cosine similarity scaling factor
- `eps = 1e-8`: Numerical stability constant

## Optimizer Configurations

| Optimizer | Learning Rate | Additional Parameters |
|-----------|--------------|----------------------|
| SGD | 1e-2 | - |
| RMSprop | 1e-3 | - |
| AdaGrad | 1e-2 | - |
| Adam | 1e-3 | - |
| AdamW | 1e-3 | weight_decay=1e-4 |
| SGD+Momentum | 1e-2 | momentum=0.9 |

## Output Structure

Results are saved in the specified results directory with the following structure:

```
results/
├── [optimizer]_[timestamp]_run[1-5]/
│   ├── best_model.pth          # Best model checkpoint
│   ├── final_model.pth         # Final model checkpoint
│   ├── results.json            # Training results summary
│   └── training_history.txt    # Per-epoch metrics
├── [optimizer]_aggregated_[timestamp]/
│   └── aggregated_results.json # Aggregated statistics across runs
└── comparison_summary.txt      # Overall comparison summary
```

## Results Interpretation

The scripts generate:
- **Individual run results**: Per-run metrics and model checkpoints
- **Aggregated statistics**: Mean ± standard deviation across multiple runs
- **Comparison summaries**: Text files with tabulated results for easy comparison

Key metrics tracked:
- Best validation accuracy
- Best F1 score
- Best test loss
- Epochs to convergence
- Training stability



