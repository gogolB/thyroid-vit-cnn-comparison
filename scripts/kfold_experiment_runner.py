#!/usr/bin/env python3
"""
K-Fold Cross-Validation Runner for Multiple Models
Directly integrates with the Unified Experiment Runner
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
import time

# Import the unified runner
from scripts.unified_experiment_runner import UnifiedExperimentRunner

console = Console()


class KFoldMultiModelRunner:
    """Manages k-fold cross-validation for multiple models."""
    
    def __init__(self, 
                 models: List[str],
                 k_folds: int = 5,
                 output_dir: Optional[Path] = None,
                 quick_test: bool = False,
                 training_config: str = 'standard'):
        """
        Initialize multi-model k-fold runner.
        
        Args:
            models: List of model names to evaluate
            k_folds: Number of folds for cross-validation
            output_dir: Directory to save results
            quick_test: Whether to run in quick test mode
            training_config: Training configuration to use
        """
        self.models = models
        self.k_folds = k_folds
        self.output_dir = output_dir or Path(f"experiments/kfold_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.quick_test = quick_test
        self.training_config = training_config
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.all_results = {}  # model -> list of fold results
        self.summary_stats = {}  # model -> aggregated statistics
        
        # Initialize unified runner
        config_dir = Path(__file__).parent.parent / "configs"
        self.runner = UnifiedExperimentRunner(config_dir)
        
    def run_single_fold(self, model_name: str, fold: int) -> Dict[str, Any]:
        """
        Run a single fold for a specific model.
        
        Args:
            model_name: Name of the model
            fold: Fold number (1-indexed for dataset compatibility)
            
        Returns:
            Dictionary containing fold results
        """
        console.print(f"  [cyan]Fold {fold}/{self.k_folds}[/cyan]")
        
        # Determine model type
        available_models = self.runner.get_available_models()
        if model_name in available_models['vit']:
            model_type = 'vit'
        elif model_name in available_models['cnn']:
            model_type = 'cnn'
        else:
            # Try to infer from name
            if any(vit in model_name for vit in ['vit', 'deit', 'swin']):
                model_type = 'vit'
            else:
                model_type = 'cnn'
        
        # Build overrides for this fold
        overrides = {
            'dataset.fold': fold,
            'wandb.mode': 'disabled',  # Disable wandb for batch runs
            'paths.checkpoint_dir': str(self.output_dir / model_name / f'fold_{fold}')
        }
        
        # Run the experiment
        try:
            result = self.runner.run_single_experiment(
                model_name=model_name,
                model_type=model_type,
                training_config=self.training_config,
                overrides=overrides,
                quick_test=self.quick_test,
                fold=fold
            )
            
            # Add fold information
            result['fold'] = fold
            
            # Save individual fold result
            fold_file = self.output_dir / model_name / f'fold_{fold}_results.json'
            fold_file.parent.mkdir(exist_ok=True)
            with open(fold_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            console.print(f"    [red]Failed: {str(e)}[/red]")
            return {
                'status': 'failed',
                'error': str(e),
                'fold': fold,
                'model_name': model_name
            }
    
    def run_model_kfold(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Run all k folds for a single model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of results for all folds
        """
        console.print(f"\n[bold cyan]Running {self.k_folds}-fold CV for {model_name}[/bold cyan]")
        
        fold_results = []
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                f"Training {model_name}...", 
                total=self.k_folds
            )
            
            for fold in range(1, self.k_folds + 1):  # 1-indexed for dataset
                fold_result = self.run_single_fold(model_name, fold)
                fold_results.append(fold_result)
                progress.update(task, advance=1)
        
        return fold_results
    
    def calculate_statistics(self, model_name: str, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics across all folds for a model.
        
        Args:
            model_name: Name of the model
            fold_results: List of results from all folds
            
        Returns:
            Dictionary of aggregated statistics
        """
        # Filter successful runs
        successful_folds = [r for r in fold_results if r.get('status') == 'success']
        
        if not successful_folds:
            return {
                'model': model_name,
                'status': 'all_failed',
                'successful_folds': 0,
                'total_folds': self.k_folds
            }
        
        # Extract metrics
        test_accs = [r['test_acc'] for r in successful_folds]
        val_accs = [r['val_acc'] for r in successful_folds]
        train_accs = [r['train_acc'] for r in successful_folds]
        
        # Find best and worst folds
        best_idx = np.argmax(test_accs)
        worst_idx = np.argmin(test_accs)
        
        stats = {
            'model': model_name,
            'successful_folds': len(successful_folds),
            'total_folds': self.k_folds,
            'test_acc': {
                'mean': np.mean(test_accs),
                'std': np.std(test_accs),
                'best': test_accs[best_idx],
                'best_fold': successful_folds[best_idx]['fold'],
                'worst': test_accs[worst_idx],
                'worst_fold': successful_folds[worst_idx]['fold'],
                'all_values': test_accs
            },
            'val_acc': {
                'mean': np.mean(val_accs),
                'std': np.std(val_accs),
                'best': max(val_accs),
                'worst': min(val_accs)
            },
            'train_acc': {
                'mean': np.mean(train_accs),
                'std': np.std(train_accs)
            }
        }
        
        # Add timing information if available
        if 'time_seconds' in successful_folds[0]:
            times = [r['time_seconds'] for r in successful_folds]
            stats['training_time'] = {
                'mean_minutes': np.mean(times) / 60,
                'total_minutes': sum(times) / 60
            }
        
        return stats
    
    def display_summary(self):
        """Display a comprehensive summary table."""
        # Create summary table
        table = Table(
            title=f"{self.k_folds}-Fold Cross-Validation Summary",
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Best Acc", justify="right", style="green")
        table.add_column("Worst Acc", justify="right", style="red")
        table.add_column("Fold", justify="center")
        table.add_column("Avg Acc ± Std", justify="right", style="yellow")
        table.add_column("Runs", justify="center")
        
        # Add rows for each model
        for model_name, stats in self.summary_stats.items():
            if stats.get('status') == 'all_failed':
                table.add_row(
                    model_name,
                    "Failed",
                    "Failed",
                    "-",
                    "Failed",
                    f"0/{stats['total_folds']}"
                )
            else:
                test_stats = stats['test_acc']
                table.add_row(
                    model_name,
                    f"{test_stats['best']:.4f}",
                    f"{test_stats['worst']:.4f}",
                    f"{test_stats['best_fold']}/{test_stats['worst_fold']}",
                    f"{test_stats['mean']:.4f} ± {test_stats['std']:.4f}",
                    f"{stats['successful_folds']}/{stats['total_folds']}"
                )
        
        console.print("\n")
        console.print(table)
        
        # Print additional insights
        if self.summary_stats:
            console.print("\n[bold]Key Insights:[/bold]")
            
            # Find overall best model
            best_model = max(
                [(k, v) for k, v in self.summary_stats.items() if v.get('status') != 'all_failed'],
                key=lambda x: x[1]['test_acc']['mean'],
                default=(None, None)
            )
            
            if best_model[0]:
                console.print(f"• Best average performance: [green]{best_model[0]}[/green] "
                            f"({best_model[1]['test_acc']['mean']:.4f} ± {best_model[1]['test_acc']['std']:.4f})")
                
                # Find most consistent model (lowest std)
                most_consistent = min(
                    [(k, v) for k, v in self.summary_stats.items() if v.get('status') != 'all_failed'],
                    key=lambda x: x[1]['test_acc']['std']
                )
                console.print(f"• Most consistent model: [blue]{most_consistent[0]}[/blue] "
                            f"(std: {most_consistent[1]['test_acc']['std']:.4f})")
    
    def save_results(self):
        """Save all results to files."""
        # Save detailed results
        results_file = self.output_dir / "kfold_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metadata': {
                    'models': self.models,
                    'k_folds': self.k_folds,
                    'timestamp': datetime.now().isoformat(),
                    'quick_test': self.quick_test
                },
                'fold_results': self.all_results,
                'summary_statistics': self.summary_stats
            }, f, indent=2)
        
        # Save summary report
        report_file = self.output_dir / "kfold_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"K-Fold Cross-Validation Report\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models: {', '.join(self.models)}\n")
            f.write(f"Folds: {self.k_folds}\n")
            f.write(f"Quick Test: {self.quick_test}\n\n")
            
            f.write("Results Summary:\n")
            f.write("-"*50 + "\n\n")
            
            for model_name, stats in self.summary_stats.items():
                f.write(f"{model_name}:\n")
                if stats.get('status') == 'all_failed':
                    f.write("  All folds failed\n\n")
                else:
                    test_stats = stats['test_acc']
                    f.write(f"  Test Accuracy: {test_stats['mean']:.4f} ± {test_stats['std']:.4f}\n")
                    f.write(f"  Best Fold: {test_stats['best_fold']} ({test_stats['best']:.4f})\n")
                    f.write(f"  Worst Fold: {test_stats['worst_fold']} ({test_stats['worst']:.4f})\n")
                    f.write(f"  Successful Runs: {stats['successful_folds']}/{stats['total_folds']}\n")
                    if 'training_time' in stats:
                        f.write(f"  Total Time: {stats['training_time']['total_minutes']:.1f} minutes\n")
                    f.write("\n")
        
        console.print(f"\n[green]✓ Results saved to {self.output_dir}[/green]")
    
    def run_all(self):
        """Run k-fold cross-validation for all models."""
        console.print(Panel.fit(
            f"[bold]Multi-Model K-Fold Cross-Validation[/bold]\n"
            f"Models: {', '.join(self.models)}\n"
            f"Folds: {self.k_folds}\n"
            f"Output: {self.output_dir}",
            title="Experiment Configuration",
            border_style="blue"
        ))
        
        # Run each model
        for model_name in self.models:
            fold_results = self.run_model_kfold(model_name)
            self.all_results[model_name] = fold_results
            
            # Calculate statistics
            stats = self.calculate_statistics(model_name, fold_results)
            self.summary_stats[model_name] = stats
            
            # Show intermediate results
            if stats.get('status') != 'all_failed':
                console.print(f"  [green]✓[/green] {model_name}: "
                            f"{stats['test_acc']['mean']:.4f} ± {stats['test_acc']['std']:.4f}")
            else:
                console.print(f"  [red]✗[/red] {model_name}: All folds failed")
        
        # Display final summary
        self.display_summary()
        
        # Save all results
        self.save_results()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run k-fold cross-validation for multiple models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'models',
        nargs='+',
        help='List of model names to evaluate (e.g., resnet18 resnet50 efficientnet_b0)'
    )
    
    parser.add_argument(
        '--k-fold',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited epochs'
    )
    
    parser.add_argument(
        '--training-config',
        type=str,
        default='standard',
        help='Training configuration to use'
    )
    
    args = parser.parse_args()
    
    # Validate k-fold value
    if args.k_fold < 2:
        console.print("[red]Error: k-fold must be at least 2[/red]")
        sys.exit(1)
    
    # Create and run the experiment
    runner = KFoldMultiModelRunner(
        models=args.models,
        k_folds=args.k_fold,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
        training_config=args.training_config
    )
    
    try:
        runner.run_all()
        console.print("\n[bold green]✓ All experiments completed successfully![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Experiment interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()