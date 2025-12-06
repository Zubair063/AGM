"""
Author:Md Zubair
Date:2025-12-05
AGM (Adaptive Gradient Modulator) Implementation with Multiple Seeds
Compares optimizers with AGM: SGD, RMSProp, AdaGrad, Adam, AdamW, SGD+Momentum
Runs 5 seeds per optimizer and saves results similar to compare_optimizers.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm
import random
from collections import Counter
import argparse


# ============================================================
# 1. AGM MODULE
# ============================================================
class AdaptiveGradientModulator:
    def __init__(self, beta=0.9, alpha=0.5, lam=0.7, kappa=2.0, eps=1e-8):
        self.beta = beta
        self.alpha = alpha
        self.lam = lam
        self.kappa = kappa
        self.eps = eps

        self.v_norm = {}
        self.prev_grad = {}

    @torch.no_grad()
    def step(self, params):
        for p in params:
            if p.grad is None:
                continue

            g = p.grad
            pid = id(p)

            # 1. EMA of gradient norm
            g_norm = g.norm(p=2)
            v = self.v_norm.get(pid, torch.zeros_like(g_norm))
            v = self.beta * v + (1 - self.beta) * g_norm
            self.v_norm[pid] = v

            # 2. magnitude scaling
            s = ((g_norm + self.eps) / (v + self.eps)) ** (-self.alpha)

            # 3. previous gradient
            g_prev = self.prev_grad.get(pid, torch.zeros_like(g))

            # 4. smoothing
            g_smooth = self.lam * g + (1 - self.lam) * g_prev

            # 5. cosine similarity
            denom = (g.norm() * (g_prev.norm() + self.eps) + self.eps)
            cos_sim = (g * g_prev).sum() / denom
            gamma = torch.sigmoid(self.kappa * cos_sim)

            # 6. final gradient
            g_mod = (s * gamma) * g_smooth
            p.grad.copy_(g_mod)

            # save for next iteration
            self.prev_grad[pid] = g.detach().clone()


# ============================================================
# 2. MLP ARCHITECTURE
# ============================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. TRAINING AND EVALUATION FUNCTIONS
# ============================================================
def train_epoch(model, loader, optimizer, agm, device):
    """Train for one epoch with AGM"""
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        loss.backward()
        
        # Apply AGM if provided
        if agm is not None:
            agm.step(model.parameters())
        
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def compute_metrics(model, loader, device):
    """Compute accuracy, AUC, and F1 score"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return accuracy, auc, f1


# ============================================================
# 4. HELPER FUNCTION TO SET SEEDS
# ============================================================
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 5. TRAINING FUNCTION FOR EACH OPTIMIZER WITH AGM
# ============================================================
def train_optimizer_with_agm(optimizer_name, device, train_loader, test_loader, 
                              patience=5, max_epochs=100, results_base_dir="./agm_results", 
                              seed=42, run_id=1):
    """
    Train model with specified optimizer and AGM
    
    Args:
        optimizer_name: Name of optimizer ('sgd', 'rmsprop', 'adagrad', 'adam', 'adamw', 'sgd_momentum')
        device: torch device
        train_loader: Training data loader
        test_loader: Test data loader
        patience: Early stopping patience
        max_epochs: Maximum epochs
        results_base_dir: Base directory for results
        seed: Random seed for reproducibility
        run_id: Run number (for multiple runs)
    
    Returns:
        Dictionary with training results
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_base_dir, f"{optimizer_name}_agm_{timestamp}_run{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model
    model = MLP().to(device)
    
    # Initialize optimizer based on name
    optimizer_configs = {
        'sgd': {'optimizer': optim.SGD, 'params': {'lr': 1e-2}},
        'rmsprop': {'optimizer': optim.RMSprop, 'params': {'lr': 1e-3}},
        'adagrad': {'optimizer': optim.Adagrad, 'params': {'lr': 1e-2}},
        'adam': {'optimizer': optim.Adam, 'params': {'lr': 1e-3}},
        'adamw': {'optimizer': optim.AdamW, 'params': {'lr': 1e-3, 'weight_decay': 1e-4}},
        'sgd_momentum': {'optimizer': optim.SGD, 'params': {'lr': 1e-2, 'momentum': 0.9}}
    }
    
    if optimizer_name not in optimizer_configs:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    config = optimizer_configs[optimizer_name]
    optimizer = config['optimizer'](model.parameters(), **config['params'])
    
    # Initialize AGM
    agm = AdaptiveGradientModulator()
    
    # Hyperparameters
    hyperparams = {
        'batch_size': 128,
        'learning_rate': config['params']['lr'],
        'optimizer': optimizer_name,
        'agm_enabled': True,
        'agm_params': {
            'beta': agm.beta,
            'alpha': agm.alpha,
            'lam': agm.lam,
            'kappa': agm.kappa
        },
        'patience': patience,
        'max_epochs': max_epochs,
        'device': str(device)
    }
    if 'momentum' in config['params']:
        hyperparams['momentum'] = config['params']['momentum']
    if 'weight_decay' in config['params']:
        hyperparams['weight_decay'] = config['params']['weight_decay']
    
    # Training tracking
    best_val_acc = 0.0
    best_val_acc_epoch = 0
    best_val_acc_f1 = 0.0
    best_model_state = None
    
    best_test_loss = float('inf')
    best_test_loss_epoch = 0
    best_test_loss_val_acc = 0.0
    
    # Track 97.5% accuracy milestone
    epoch_97_5_percent = None
    model_97_5_percent_state = None
    accuracy_97_5_percent = None
    
    patience_counter = 0
    training_history = []
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(1, max_epochs + 1), 
                      desc=f"{optimizer_name.upper()}+AGM", 
                      unit="epoch",
                      leave=True,
                      ncols=100)
    
    for epoch in epoch_pbar:
        train_loss = train_epoch(model, train_loader, optimizer, agm, device)
        val_acc, val_auc, val_f1 = compute_metrics(model, test_loader, device)
        
        # Compute test loss
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = nn.CrossEntropyLoss()(logits, y)
                test_loss += loss.item()
        test_loss = test_loss / len(test_loader)
        
        # Record metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1
        }
        training_history.append(epoch_metrics)
        
        # Track 97.5% accuracy milestone (first time reached)
        if epoch_97_5_percent is None and val_acc >= 0.975:
            epoch_97_5_percent = epoch
            model_97_5_percent_state = model.state_dict().copy()
            accuracy_97_5_percent = val_acc
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            best_val_acc_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        
        # Track best test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_loss_epoch = epoch
            best_test_loss_val_acc = val_acc
        
        # Early stopping
        if val_acc <= best_val_acc:
            patience_counter += 1
        
        # Update progress bar description
        status = f"Acc:{val_acc*100:.2f}% | F1:{val_f1:.4f} | Loss:{train_loss:.4f}"
        if val_acc == best_val_acc and epoch == best_val_acc_epoch:
            status += " | ✓Best"
        if epoch_97_5_percent is not None and epoch == epoch_97_5_percent:
            status += " | 🎯97.5%"
        if patience_counter > 0:
            status += f" | P:{patience_counter}/{patience}"
        
        epoch_pbar.set_postfix_str(status)
        
        if patience_counter >= patience:
            epoch_pbar.set_postfix_str("Early stopping!")
            epoch_pbar.close()
            model.load_state_dict(best_model_state)
            break
    
    epoch_pbar.close()
    
    # Save best model
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'epoch': best_val_acc_epoch,
        'metrics': {
            'accuracy': best_val_acc,
            'f1': best_val_acc_f1
        },
        'hyperparams': hyperparams
    }, best_model_path)
    
    # Save model at 97.5% accuracy milestone (if reached)
    if epoch_97_5_percent is not None:
        model_97_5_path = os.path.join(results_dir, 'model_97_5_percent.pth')
        # Temporarily load the 97.5% model to compute F1
        temp_model = MLP().to(device)
        temp_model.load_state_dict(model_97_5_percent_state)
        _, _, f1_97_5 = compute_metrics(temp_model, test_loader, device)
        del temp_model
        
        torch.save({
            'model_state_dict': model_97_5_percent_state,
            'epoch': epoch_97_5_percent,
            'metrics': {
                'accuracy': accuracy_97_5_percent,
                'f1': f1_97_5
            },
            'hyperparams': hyperparams
        }, model_97_5_path)
    
    # Save final model
    final_model_path = os.path.join(results_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': {
            'accuracy': val_acc,
            'f1': val_f1
        },
        'hyperparams': hyperparams
    }, final_model_path)
    
    # Prepare results summary
    results = {
        'optimizer': optimizer_name,
        'seed': seed,
        'run_id': run_id,
        'best_accuracy': float(best_val_acc),
        'best_f1_score': float(best_val_acc_f1),
        'best_accuracy_epoch': int(best_val_acc_epoch),
        'best_test_loss': float(best_test_loss),
        'best_test_loss_epoch': int(best_test_loss_epoch),
        'best_test_loss_val_acc': float(best_test_loss_val_acc),
        'epoch_97_5_percent': epoch_97_5_percent,
        'final_epoch': epoch,
        'hyperparams': hyperparams
    }
    
    # Save results as JSON
    results_path = os.path.join(results_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history as CSV
    history_path = os.path.join(results_dir, 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTest_Loss\tVal_Accuracy\tVal_AUC\tVal_F1\n")
        for h in training_history:
            f.write(f"{h['epoch']}\t{h['train_loss']:.6f}\t{h['test_loss']:.6f}\t"
                   f"{h['val_accuracy']:.6f}\t{h['val_auc']:.6f}\t{h['val_f1']:.6f}\n")
    
    return results


# ============================================================
# 6. AGGREGATE RESULTS FUNCTION
# ============================================================
def aggregate_results(all_runs_results):
    """
    Aggregate results from multiple runs
    
    Args:
        all_runs_results: List of result dictionaries from multiple runs
    
    Returns:
        Dictionary with aggregated statistics
    """
    if not all_runs_results:
        return None
    
    # Extract metrics
    best_accuracies = [r['best_accuracy'] for r in all_runs_results]
    best_f1_scores = [r['best_f1_score'] for r in all_runs_results]
    best_test_losses = [r['best_test_loss'] for r in all_runs_results]
    
    # Extract epochs
    best_acc_epochs = [r['best_accuracy_epoch'] for r in all_runs_results]
    epoch_97_5_list = [r['epoch_97_5_percent'] for r in all_runs_results if r['epoch_97_5_percent'] is not None]
    best_test_loss_epochs = [r['best_test_loss_epoch'] for r in all_runs_results]
    
    # Calculate statistics for metrics
    aggregated = {
        'best_accuracy': {
            'mean': float(np.mean(best_accuracies)),
            'std': float(np.std(best_accuracies)),
            'values': best_accuracies
        },
        'best_f1_score': {
            'mean': float(np.mean(best_f1_scores)),
            'std': float(np.std(best_f1_scores)),
            'values': best_f1_scores
        },
        'best_test_loss': {
            'mean': float(np.mean(best_test_losses)),
            'std': float(np.std(best_test_losses)),
            'values': best_test_losses
        },
        # Mode for epochs
        'best_accuracy_epoch_mode': int(Counter(best_acc_epochs).most_common(1)[0][0]),
        'best_accuracy_epochs': best_acc_epochs,
        'epoch_97_5_percent_mode': int(Counter(epoch_97_5_list).most_common(1)[0][0]) if epoch_97_5_list else None,
        'epoch_97_5_percent_count': len(epoch_97_5_list),
        'best_test_loss_epoch_mode': int(Counter(best_test_loss_epochs).most_common(1)[0][0]),
        'best_test_loss_epochs': best_test_loss_epochs,
        'num_runs': len(all_runs_results),
        'reached_97_5_percent': len(epoch_97_5_list) > 0
    }
    
    return aggregated


# ============================================================
# 7. MAIN FUNCTION
# ============================================================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare optimizers with AGM on MNIST')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--results_dir', type=str, default='./agm_results',
                        help='Results directory (default: ./agm_results)')
    parser.add_argument('--num_runs', type=int, default=5,
                        help='Number of runs per optimizer (default: 5)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for each run (default: [42, 123, 456, 789, 1011])')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # MNIST DATA
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Optimizers to test with AGM
    optimizers_to_run = ['sgd', 'rmsprop', 'adagrad', 'adam', 'adamw', 'sgd_momentum']
    
    # Training parameters
    patience = args.patience
    max_epochs = args.max_epochs
    num_runs = args.num_runs
    seeds = args.seeds if args.seeds is not None else [42, 123, 456, 789, 1011]
    results_base_dir = args.results_dir
    
    # Ensure we have enough seeds
    if len(seeds) < num_runs:
        seeds = seeds + [seeds[-1] + i + 1 for i in range(num_runs - len(seeds))]
    
    # Create results directory
    os.makedirs(results_base_dir, exist_ok=True)
    
    print(f"\n{'='*110}")
    print("TRAINING OPTIMIZERS WITH AGM (Adaptive Gradient Modulator)")
    print(f"{'='*110}")
    print(f"Optimizers: {', '.join([opt.upper() + ' + AGM' for opt in optimizers_to_run])}")
    print(f"Number of runs per optimizer: {num_runs}")
    print(f"Seeds: {seeds}")
    print(f"Patience: {patience}, Max epochs: {max_epochs}")
    print(f"Results directory: {results_base_dir}")
    print(f"{'='*110}\n")
    
    all_aggregated_results = {}
    
    # Run each optimizer with multiple seeds
    optimizer_pbar = tqdm(optimizers_to_run, desc="Optimizers", unit="optimizer", ncols=100)
    
    for opt_name in optimizer_pbar:
        all_runs_results = []
        
        for run_id in range(1, num_runs + 1):
            seed = seeds[run_id - 1] if run_id - 1 < len(seeds) else seeds[-1]
            try:
                result = train_optimizer_with_agm(
                    optimizer_name=opt_name,
                    device=device,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    patience=patience,
                    max_epochs=max_epochs,
                    results_base_dir=results_base_dir,
                    seed=seed,
                    run_id=run_id
                )
                all_runs_results.append(result)
            except Exception as e:
                print(f"\nError in {opt_name} run {run_id}: {e}")
                continue
        
        # Aggregate results
        if all_runs_results:
            aggregated = aggregate_results(all_runs_results)
            all_aggregated_results[opt_name] = aggregated
            
            # Save aggregated results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agg_results_dir = os.path.join(results_base_dir, f"{opt_name}_agm_aggregated_{timestamp}")
            os.makedirs(agg_results_dir, exist_ok=True)
            
            agg_results_path = os.path.join(agg_results_dir, 'aggregated_results.json')
            with open(agg_results_path, 'w') as f:
                json.dump({
                    'optimizer': opt_name,
                    'num_runs': num_runs,
                    'seeds': seeds[:len(all_runs_results)],
                    'aggregated': aggregated,
                    'individual_runs': all_runs_results
                }, f, indent=2)
            
            optimizer_pbar.set_postfix_str(f"✓ {opt_name} ({len(all_runs_results)}/{num_runs} runs)")
        else:
            optimizer_pbar.set_postfix_str(f"✗ {opt_name} failed")
    
    optimizer_pbar.close()
    print("")
    
    # Print and save comparison summary
    if len(all_aggregated_results) > 0:
        summary_lines = []
        summary_lines.append("AGGREGATED COMPARISON SUMMARY (Mean ± Std over multiple runs)")
        summary_lines.append(f"{'='*110}")
        summary_lines.append(f"{'Optimizer':<15} {'Best Acc':<20} {'Best F1':<20} {'Best Acc Epoch':<18} "
                            f"{'97.5% Epoch':<15} {'Best Test Loss':<20}")
        summary_lines.append(f"{'-'*110}")
        
        for opt_name in optimizers_to_run:
            if opt_name in all_aggregated_results:
                agg = all_aggregated_results[opt_name]
                acc_str = f"{agg['best_accuracy']['mean']*100:.2f}±{agg['best_accuracy']['std']*100:.2f}%"
                f1_str = f"{agg['best_f1_score']['mean']*100:.2f}±{agg['best_f1_score']['std']*100:.2f}%"
                test_loss_str = f"{agg['best_test_loss']['mean']:.4f}±{agg['best_test_loss']['std']:.4f}"
                
                epoch_97_5_str = str(agg['epoch_97_5_percent_mode']) if agg['epoch_97_5_percent_mode'] is not None else "N/A"
                if agg['epoch_97_5_percent_mode'] is not None:
                    epoch_97_5_str += f" ({agg['epoch_97_5_percent_count']}/{agg['num_runs']})"
                
                line = (f"{opt_name:<15} {acc_str:<20} {f1_str:<20} "
                       f"{agg['best_accuracy_epoch_mode']:<18} {epoch_97_5_str:<15} {test_loss_str:<20}")
                summary_lines.append(line)
                if len(all_aggregated_results) > 1:
                    print(line)
        
        summary_lines.append(f"{'='*110}")
        
        if len(all_aggregated_results) > 1:
            print(f"{'='*110}\n")
        
        # Save summary to text file
        summary_path = os.path.join(results_base_dir, 'agm_comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"Summary saved to: {summary_path}")
    
    print("All training completed!")


if __name__ == "__main__":
    main()
