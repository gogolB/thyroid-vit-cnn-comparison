#!/usr/bin/env python3
"""
Experiment runner for CARS Thyroid Classification
Handles both CNN and ViT experiments with proper configuration
Fixed subprocess working directory and environment setup
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run thyroid classification experiments')
    
    # Model selection
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet-b0', 'efficientnet-b3', 'densenet121',
                               'inception_v3', 'mobilenetv3', 'vit-tiny', 'vit-small'],
                       help='Model architecture to train')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    # Quality-aware preprocessing
    parser.add_argument('--quality-aware', action='store_true', default=True,
                       help='Use quality-aware preprocessing')
    parser.add_argument('--no-quality-aware', dest='quality_aware', action='store_false',
                       help='Disable quality-aware preprocessing')
    
    # Other options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode (fast_dev_run)')
    parser.add_argument('--wandb-mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases logging mode')
    
    return parser.parse_args()


def get_model_config_name(model_name: str) -> tuple:
    """Get the configuration name and type for a model."""
    if 'resnet' in model_name:
        return f'cnn/{model_name}', 'cnn'
    elif 'efficientnet' in model_name:
        return f'cnn/{model_name}', 'cnn'
    elif 'densenet' in model_name:
        return f'cnn/{model_name}', 'cnn'
    elif 'inception' in model_name:
        return f'cnn/{model_name}', 'cnn'
    elif 'mobilenet' in model_name:
        return f'cnn/{model_name}', 'cnn'
    elif 'vit' in model_name:
        return f'vit/{model_name}', 'vit'
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(args):
    """Run the experiment with given arguments."""
    
    # Display experiment configuration
    console.print(Panel.fit(
        f"[bold cyan]CARS Thyroid Classification Experiment[/bold cyan]\n"
        f"[yellow]Model:[/yellow] {args.model}\n"
        f"[yellow]Quality-Aware:[/yellow] {args.quality_aware}\n"
        f"[yellow]Batch Size:[/yellow] {args.batch_size}\n"
        f"[yellow]Epochs:[/yellow] {args.epochs}\n"
        f"[yellow]Learning Rate:[/yellow] {args.lr}",
        border_style="blue"
    ))
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Get model configuration
    model_config, model_type = get_model_config_name(args.model)
    
    # Determine script to run
    if model_type == 'cnn':
        script = "src/training/train_cnn.py"
    else:
        script = "src/training/train_vit.py"
    
    # Build command (no conflicting Hydra args)
    cmd = [
        sys.executable,
        script,
        f"model={model_config}",
        f"training.batch_size={args.batch_size}",
        f"training.num_epochs={args.epochs}",
        f"training.optimizer.lr={args.lr}",
        f"seed={args.seed}",
        f"wandb.mode={args.wandb_mode}",
    ]
    
    # Add quality-aware flag
    if hasattr(args, 'quality_aware'):
        cmd.append(f"model.quality_aware={args.quality_aware}")
    
    # Add debug mode overrides
    if args.debug:
        cmd.extend([
            "trainer.fast_dev_run=true",
            "trainer.max_epochs=2",
            "trainer.limit_train_batches=2",
            "trainer.limit_val_batches=2",
        ])
    
    # Print command for debugging
    console.print("\n[dim]Running command:[/dim]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")
    console.print(f"[dim]Working dir: {project_root}[/dim]\n")
    
    # Run the experiment
    try:
        console.print("[cyan]Starting experiment...[/cyan]")
        
        # Set environment variables to help with Hydra
        env = os.environ.copy()
        env['HYDRA_FULL_ERROR'] = '1'  # For better error messages
        env['PYTHONPATH'] = str(project_root)  # Ensure Python can find modules
        
        # Run subprocess with proper working directory
        result = subprocess.run(
            cmd,
            cwd=project_root,  # Set working directory to project root
            env=env,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            console.print("\n[bold green]✅ Experiment completed successfully![/bold green]")
        else:
            console.print(f"\n[bold red]❌ Experiment failed with return code {result.returncode}[/bold red]")
            sys.exit(result.returncode)
            
    except Exception as e:
        console.print(f"\n[bold red]Error running experiment: {e}[/bold red]")
        sys.exit(1)


def main():
    """Main function."""
    args = parse_args()
    
    # Get project root and check if we have the right structure
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    
    if not config_dir.exists():
        console.print(f"[red]Error: Config directory not found at {config_dir}[/red]")
        console.print("[red]Please run this script from the project root directory[/red]")
        sys.exit(1)
    
    console.print(f"[green]✓ Config directory found: {config_dir}[/green]")
    
    # Run the experiment
    run_experiment(args)


if __name__ == "__main__":
    main()