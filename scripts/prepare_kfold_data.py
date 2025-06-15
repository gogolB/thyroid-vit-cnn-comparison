#!/usr/bin/env python3
"""
Data preparation script for CARS thyroid dataset.
This version implements a true k-fold cross-validation scheme where train,
validation, and test sets are all rotated for each fold.
"""

import os
import sys
from pathlib import Path
import numpy as np
import argparse
import json
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Assumes dataset.py is in the same directory for import
from src.data.dataset import CARSThyroidDataset

console = Console()

def generate_kfold_splits(
    data_dir: Path,
    k: int,
    random_state: int = 42
):
    """
    Generates k-fold splits where each fold has a unique train, val, and test set.
    """
    console.print(Panel(f"[bold green]Generating {k}-Fold Rotating Splits[/bold green]", border_style="green"))
    splits_dir = data_dir.parent / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all file paths and labels
    full_dataset = CARSThyroidDataset(root_dir=data_dir, split='all')
    indices = np.arange(len(full_dataset.image_paths))
    labels = full_dataset.labels

    # Create k folds from the entire dataset
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_indices = [test_idx for _, test_idx in skf.split(indices, labels)]

    # Now, for each run, assign train, val, and test folds
    for i in range(k):
        fold_num = i + 1
        
        # Test set is the i-th fold
        test_idx = fold_indices[i]
        
        # Validation set is the next fold, wrapping around
        val_idx = fold_indices[(i + 1) % k]
        
        # Training set is all other folds
        train_folds_indices = [j for j in range(k) if j != i and j != (i + 1) % k]
        train_idx = np.concatenate([fold_indices[j] for j in train_folds_indices])
        
        fold_split_path = splits_dir / f'split_fold_{fold_num}.json'
        with open(fold_split_path, 'w') as f:
            json.dump({
                'train': train_idx.tolist(),
                'val': val_idx.tolist(),
                'test': test_idx.tolist()
            }, f, indent=2)
            
        console.print(f"✓ Saved Fold {fold_num} ({len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test) to [cyan]{fold_split_path}[/cyan]")


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare CARS thyroid dataset for training")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds to generate for cross-validation.")
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]CARS Thyroid Dataset Preparation[/bold cyan]", border_style="blue"))
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        console.print(f"[red]Error: Directory {data_dir} does not exist![/red]")
        return
        
    if args.k_folds:
        if args.k_folds < 3: # Need at least 3 for separate train/val/test
            console.print("[red]Error: --k-folds must be 3 or greater for rotating splits.[/red]")
            return
        generate_kfold_splits(data_dir, args.k_folds)
        console.print("\n[bold green]✓ K-fold rotating split generation complete![/bold green]")
        console.print(f"  You can now run your training script {args.k_folds} times, passing '--fold N' for N in 1 to {args.k_folds}.")
    else:
        console.print("[yellow]Please specify the number of folds with --k-folds N.[/yellow]")

if __name__ == "__main__":
    main()