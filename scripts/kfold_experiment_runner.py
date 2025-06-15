#!/usr/bin/env python3
"""
K-Fold Cross-Validation Runner for Thyroid CARS Classification
Runs experiments with k-fold validation and aggregates results
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time

console = Console()


class KFoldExperimentRunner:
    """Manages k-fold cross-validation experiments."""
    
    def __init__(self, 
                 model_name: str,
                 k_folds: int = 5,
                 base_config: Optional[str] = None,
                 output_dir: Optional[Path] = None,
                 wandb_project: Optional[str] = None):
        """
        Initialize k-fold runner.
        
        Args:
            model_name: Name of the model to train (e.g., 'swin_tiny', 'resnet50')
            k_folds: Number of folds for cross-validation
            base_config: Base configuration to override
            output_dir: Directory to save results
            wandb_project: W&B project name for tracking
        """
        self.model_name = model_name
        self.k_folds = k_folds
        self.base_config = base_config or "experiment/base"
        self.output_dir = output_dir or Path(f"experiments/kfold_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.wandb_project = wandb_project
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.fold_results = []
        self.aggregated_results = {}
        
    def run_single_fold(self, fold: int) -> Dict[str, Any]:
        """
        Run a single fold of the experiment.
        
        Args:
            fold: Fold number (0-indexed)
            
        Returns:
            Dictionary containing fold results
        """
        console.print(f"\n[bold cyan]Running Fold {fold + 1}/{self.k_folds}[/bold cyan]")
        
        # Construct command
        cmd = [
            "python", "train.py",  # Or path to your unified runner
            f"--config-name={self.base_config}",
            f"model={self.model_name}",
            f"dataset.k_fold={self.k_folds}",
            f"dataset.fold_idx={fold}",
            f"training.run_name={self.model_name}_fold{fold}",
            f"hydra.run.dir={self.output_dir}/fold_{fold}",
        ]
        
        # Add W&B configuration if provided
        if self.wandb_project:
            cmd.extend([
                f"wandb.project={self.wandb_project}",
                f"wandb.name={self.model_name}_fold{fold}",
                "wandb.mode=online"
            ])
        else:
            cmd.append("wandb.mode=disabled")
        
        # Run the experiment
        start_time = time.time()
        
        try:
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Parse results from output directory
            results_file = self.output_dir / f"fold_{fold}" / "results.json"
            
            # If results file doesn't exist, try to parse from stdout
            if results_file.exists():
                with open(results_file, 'r') as f:
                    fold_results = json.load(f)
            else:
                # Try to extract results from stdout
                fold_results = self._parse_results_from_output(result.stdout)
            
            # Add metadata
            fold_results.update({
                'fold': fold,
                'duration': duration,
                'status': 'success',
                'model': self.model_name
            })
            
            # Save individual fold results
            self._save_fold_results(fold, fold_results)
            
            console.print(f"[green]✓ Fold {fold + 1} completed successfully[/green]")
            console.print(f"  Accuracy: {fold_results.get('test_acc', 'N/A'):.4f}")
            console.print(f"  Duration: {duration/60:.1f} minutes")
            
            return fold_results
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Fold {fold + 1} failed[/red]")
            console.print(f"[red]Error: {e.stderr}[/red]")
            
            return {
                'fold': fold,
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def _parse_results_from_output(self, output: str) -> Dict[str, Any]:
        """
        Parse results from command output if results.json is not available.
        
        Args:
            output: stdout from the training command
            
        Returns:
            Dictionary of parsed results
        """
        results = {}
        
        # Common patterns to look for in output
        patterns = {
            'test_acc': r'Test Accuracy[:\s]+([0-9.]+)',
            'test_loss': r'Test Loss[:\s]+([0-9.]+)',
            'val_acc': r'Val Accuracy[:\s]+([0-9.]+)',
            'val_loss': r'Val Loss[:\s]+([0-9.]+)',
            'precision': r'Precision[:\s]+([0-9.]+)',
            'recall': r'Recall[:\s]+([0-9.]+)',
            'f1': r'F1[:\s]+([0-9.]+)',
        }
        
        import re
        for key, pattern in patterns.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results[key] = float(match.group(1))
        
        return results
    
    def _save_fold_results(self, fold: int, results: Dict[str, Any]):
        """Save results for a single fold."""
        fold_file = self.output_dir / f"fold_{fold}_results.json"
        with open(fold_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def run_all_folds(self):
        """Run all k folds sequentially."""
        console.print(Panel.fit(
            f"[bold]K-Fold Cross-Validation[/bold]\n"
            f"Model: {self.model_name}\n"
            f"Folds: {self.k_folds}\n"
            f"Output: {self.output_dir}",
            title="Experiment Configuration",
            border_style="blue"
        ))
        
        # Progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Running {self.k_folds}-fold cross-validation...", 
                total=self.k_folds
            )
            
            # Run each fold
            for fold in range(self.k_folds):
                fold_result = self.run_single_fold(fold)
                self.fold_results.append(fold_result)
                progress.update(task, advance=1)
        
        # Aggregate results
        self._aggregate_results()
        
        # Display summary
        self._display_summary()
        
        # Save aggregated results
        self._save_aggregated_results()
    
    def _aggregate_results(self):
        """Aggregate results across all folds."""
        # Filter successful folds
        successful_folds = [r for r in self.fold_results if r.get('status') == 'success']
        
        if not successful_folds:
            console.print("[red]No successful folds to aggregate![/red]")
            return
        
        # Metrics to aggregate
        metrics = ['test_acc', 'test_loss', 'val_acc', 'val_loss', 
                  'precision', 'recall', 'f1']
        
        for metric in metrics:
            values = [r.get(metric) for r in successful_folds if metric in r]
            if values:
                self.aggregated_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Add metadata
        self.aggregated_results['metadata'] = {
            'model': self.model_name,
            'k_folds': self.k_folds,
            'successful_folds': len(successful_folds),
            'failed_folds': len(self.fold_results) - len(successful_folds),
            'timestamp': datetime.now().isoformat(),
            'total_duration': sum(r.get('duration', 0) for r in self.fold_results)
        }
    
    def _display_summary(self):
        """Display a summary table of results."""
        if not self.aggregated_results:
            return
        
        # Create summary table
        table = Table(title=f"{self.k_folds}-Fold Cross-Validation Results", 
                     show_header=True, 
                     header_style="bold magenta")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Mean ± Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        
        # Add rows for each metric
        for metric, stats in self.aggregated_results.items():
            if metric != 'metadata' and isinstance(stats, dict):
                mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                table.add_row(
                    metric.replace('_', ' ').title(),
                    mean_std,
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}"
                )
        
        console.print("\n")
        console.print(table)
        
        # Print metadata
        meta = self.aggregated_results['metadata']
        console.print(f"\n[dim]Total duration: {meta['total_duration']/60:.1f} minutes[/dim]")
        console.print(f"[dim]Successful folds: {meta['successful_folds']}/{self.k_folds}[/dim]")
    
    def _save_aggregated_results(self):
        """Save aggregated results to file."""
        # Save detailed results
        results_file = self.output_dir / "kfold_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'fold_results': self.fold_results,
                'aggregated_results': self.aggregated_results
            }, f, indent=2)
        
        # Save summary for easy reading
        summary_file = self.output_dir / "kfold_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"K-Fold Cross-Validation Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Folds: {self.k_folds}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Results:\n")
            f.write("-" * 50 + "\n")
            
            for metric, stats in self.aggregated_results.items():
                if metric != 'metadata' and isinstance(stats, dict):
                    f.write(f"{metric.replace('_', ' ').title():20s}: "
                           f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})\n")
        
        console.print(f"\n[green]✓ Results saved to {self.output_dir}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run k-fold cross-validation experiments"
    )
    
    parser.add_argument(
        'model',
        type=str,
        help='Model name (e.g., swin_tiny, resnet50, efficientnet_b0)'
    )
    
    parser.add_argument(
        '--k-folds',
        type=int,
        default=7,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='experiment/base',
        help='Base configuration name (default: experiment/base)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Weights & Biases project name'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Create runner
    runner = KFoldExperimentRunner(
        model_name=args.model,
        k_folds=args.k_folds,
        base_config=args.config,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        wandb_project=args.wandb_project
    )
    
    # Run experiments
    try:
        runner.run_all_folds()
    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Experiment failed: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()