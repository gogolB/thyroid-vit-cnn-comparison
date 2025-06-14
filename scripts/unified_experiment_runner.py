#!/usr/bin/env python3
"""
Unified Experiment Runner for CARS Thyroid Classification
Runs all CNN and Vision Transformer experiments directly without subprocesses
Supports ResNet, EfficientNet, DenseNet, Inception, ViT, DeiT, and Swin models
"""

import os
import sys
import traceback
import argparse
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# Import project modules
from src.data.dataset import create_data_loaders
from src.data.quality_preprocessing import create_quality_aware_transform
from src.utils.device import get_device, device_info
from src.training.train_cnn import ThyroidCNNModule
from src.utils.checkpoint_utils import BestCheckpointCallback

# Import ViT models if available
try:
    from src.models.vit import get_vit_model
    from src.training.train_vit import ThyroidViTModule
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

# Import Swin models if available
try:
    from src.models.vit.swin_transformer import (
        create_swin_tiny, create_swin_small, create_swin_base, 
        create_swin_large, create_swin_medical
    )
    SWIN_AVAILABLE = True
except ImportError:
    SWIN_AVAILABLE = False

console = Console()


class UnifiedExperimentRunner:
    """Unified experiment runner for all CNN and Vision Transformer models."""
    
    def __init__(self, config_dir: Path):
        """Initialize the experiment runner.
        
        Args:
            config_dir: Path to Hydra configuration directory
        """
        self.config_dir = config_dir.absolute()
        self.results = {}
        
        # Ensure Hydra is clean
        GlobalHydra.instance().clear()
        
        console.print(Panel.fit(
            "[bold cyan]Unified CARS Thyroid Classification Runner[/bold cyan]\n"
            "[dim]Direct execution - No subprocesses[/dim]",
            border_style="blue"
        ))
        
        # Detect available features
        if not VIT_AVAILABLE:
            console.print("[yellow]Warning: Vision Transformer models not available[/yellow]")
        if not SWIN_AVAILABLE and VIT_AVAILABLE:
            console.print("[yellow]Warning: Swin Transformer models not available[/yellow]")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available model configurations."""
        models = {
            'cnn': [],
            'vit': [] 
        }
        
        # Scan CNN models
        cnn_dir = self.config_dir / 'model' / 'cnn'
        if cnn_dir.exists():
            for config_file in cnn_dir.glob('*.yaml'):
                model_name = config_file.stem
                if model_name != '__init__':  # Skip __init__.yaml
                    models['cnn'].append(model_name)
        
        # Scan ViT models
        vit_dir = self.config_dir / 'model' / 'vit'
        if vit_dir.exists():
            for config_file in vit_dir.glob('*.yaml'):
                model_name = config_file.stem
                if model_name != '__init__':  # Skip __init__.yaml
                    models['vit'].append(model_name)
        
        return models
    
    def create_experiment_config(
        self,
        model_name: str,
        model_type: str = 'cnn',
        training_config: str = 'standard',
        overrides: Optional[Dict] = None
    ) -> DictConfig:
        """Create configuration for a single experiment.
        
        Args:
            model_name: Name of the model to train
            model_type: Type of model ('cnn' or 'vit')
            training_config: Training configuration to use
            overrides: Dictionary of configuration overrides
            
        Returns:
            Composed configuration
        """
        try:
            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()
            
            # Adjust training config for specific models
            if model_name.startswith('swin') and training_config == 'standard':
                # Check if swin_standard exists
                swin_config_path = self.config_dir / 'training' / 'swin_standard.yaml'
                if swin_config_path.exists():
                    training_config = 'swin_standard'
                    console.print("[cyan]Using Swin-optimized training configuration[/cyan]")
                elif model_type == 'vit':
                    # Fall back to vit_standard if it exists
                    vit_config_path = self.config_dir / 'training' / 'vit_standard.yaml'
                    if vit_config_path.exists():
                        training_config = 'vit_standard'
            elif model_type == 'vit' and training_config == 'standard':
                # Check if vit_standard exists for other ViT models
                vit_config_path = self.config_dir / 'training' / 'vit_standard.yaml'
                if vit_config_path.exists():
                    training_config = 'vit_standard'
            
            # Initialize Hydra with config directory
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                # Compose configuration
                cfg = compose(
                    config_name="config",  # Use the main config.yaml
                    overrides=[
                        f"model={model_type}/{model_name}",
                        f"training={training_config}",
                        "hydra.run.dir=outputs/${model.name}/${now:%Y-%m-%d_%H-%M-%S}",
                        "hydra.job.chdir=false"
                    ]
                )
                
            # Convert to dict for manipulation
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            
            # Apply Swin-specific adjustments
            if model_name.startswith('swin'):
                self._apply_swin_config(model_name, cfg_dict)
            
            # Apply overrides
            if overrides:
                for key, value in overrides.items():
                    keys = key.split('.')
                    target = cfg_dict
                    for k in keys[:-1]:
                        if k not in target:
                            target[k] = {}
                        target = target[k]
                    target[keys[-1]] = value
            
            # Convert back to OmegaConf and resolve any remaining interpolations
            cfg = OmegaConf.create(cfg_dict)
            
            # Ensure directories exist
            Path(cfg.paths.data_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            return cfg
            
        except Exception as e:
            console.print(f"[red]Error creating config: {e}[/red]")
            console.print("\n[red]Stack trace:[/red]")
            console.print(traceback.format_exc())
            raise
        finally:
            # Always clean up Hydra
            GlobalHydra.instance().clear()
    
    def _apply_swin_config(self, model_name: str, cfg_dict: dict):
        """Apply Swin-specific configuration adjustments."""
        # Check for Blackwell GPU optimizations
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory > 80:  # Blackwell GPU detected
                console.print("[green]Blackwell GPU detected! Applying optimizations.[/green]")
                
                # Apply Blackwell optimizations if defined in model config
                if 'blackwell_optimizations' in cfg_dict.get('model', {}):
                    blackwell_opts = cfg_dict['model']['blackwell_optimizations']
                    
                    if 'batch_size' in blackwell_opts:
                        cfg_dict['training']['batch_size'] = blackwell_opts['batch_size']
                        console.print(f"[cyan]Batch size set to {blackwell_opts['batch_size']}[/cyan]")
                    
                    if 'mixed_precision' in blackwell_opts:
                        cfg_dict['trainer']['precision'] = (
                            'bf16-mixed' if blackwell_opts['mixed_precision'] == 'bf16' else '16-mixed'
                        )
            
            # Memory warning for Swin-Large
            if model_name == 'swin_large' and gpu_memory < 40:
                console.print("[yellow]⚠️  Warning: Swin-Large requires significant GPU memory![/yellow]")
                console.print(f"[yellow]Available: {gpu_memory:.1f}GB, Recommended: 40GB+[/yellow]")
    
    def _execute_training(self, cfg: DictConfig) -> Dict:
        """Execute the training for a single experiment.
        
        Args:
            cfg: Experiment configuration
            
        Returns:
            Training results
        """
        # Get device
        device = get_device()
        console.print(f"[cyan]Using device: {device}[/cyan]")
        
        # Create data loaders
        console.print("[yellow]Creating data loaders...[/yellow]")
        train_loader, val_loader, test_loader = create_data_loaders(cfg)
        
        # Determine model type from config
        model_type = cfg.model.get('type', None)
        model_name = cfg.model.name
        
        # Create appropriate model module
        if model_type in ['vit', 'deit', 'swin'] or model_name.startswith(('vit_', 'deit_', 'swin_')):
            if not VIT_AVAILABLE:
                raise ImportError("Vision Transformer models are not available. Please check installation.")
            
            # Use Vision Transformer module
            model_module = ThyroidViTModule(cfg)
            
            # For Swin models, override the model creation if available
            if model_name.startswith('swin') and SWIN_AVAILABLE:
                params = dict(cfg.model.params) if 'params' in cfg.model else {}
                
                # Create the appropriate Swin model
                if model_name == 'swin_tiny':
                    model_module.model = create_swin_tiny(**params)
                elif model_name == 'swin_small':
                    model_module.model = create_swin_small(**params)
                elif model_name == 'swin_base':
                    model_module.model = create_swin_base(**params)
                elif model_name == 'swin_large':
                    model_module.model = create_swin_large(**params)
                elif model_name == 'swin_medical':
                    model_module.model = create_swin_medical(**params)
                    
                console.print(f"[green]Created Swin Transformer: {model_name}[/green]")
        else:
            # Use CNN module
            model_module = ThyroidCNNModule(cfg)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model_module.parameters())
        trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        console.print(f"[green]Model created: {cfg.model.name}[/green]")
        console.print(f"  Total parameters: {total_params:,}")
        console.print(f"  Trainable parameters: {trainable_params:,}")
        
        # Create callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_acc:.4f}}",
            monitor='val_acc',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if cfg.training.early_stopping.enabled:
            early_stop_callback = EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
                min_delta=cfg.training.early_stopping.min_delta,
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Progress bar
        callbacks.append(RichProgressBar())
        
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        
        # Best checkpoint callback
        callbacks.append(BestCheckpointCallback())
        
        # Create logger
        logger = None
        if cfg.wandb.mode != 'disabled':
            logger = WandbLogger(
                project=cfg.wandb.project,
                name=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=cfg.wandb.tags,
                mode=cfg.wandb.mode,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        # Create trainer
        trainer_config = dict(cfg.trainer)
        trainer_config.update({
            'callbacks': callbacks,
            'logger': logger,
            'max_epochs': cfg.training.num_epochs,
        })
        
        # Handle device-specific settings
        if device.type == 'mps':
            trainer_config['accelerator'] = 'mps'
            trainer_config['devices'] = 1
            trainer_config['precision'] = 32  # MPS doesn't support mixed precision yet
        elif device.type == 'cuda':
            trainer_config['accelerator'] = 'gpu'
            trainer_config['devices'] = 1
            
            # Apply mixed precision settings for Swin if specified
            if model_name.startswith('swin') and 'mixed_precision' in cfg.get('training', {}):
                if cfg.training.mixed_precision == 'bf16':
                    trainer_config['precision'] = 'bf16-mixed'
                elif cfg.training.mixed_precision == 'fp16':
                    trainer_config['precision'] = '16-mixed'
        else:
            trainer_config['accelerator'] = 'cpu'
            trainer_config['devices'] = 1
        
        trainer = pl.Trainer(**trainer_config)
        
        # Train
        console.print(f"\n[bold yellow]Starting training for {cfg.model.name}...[/bold yellow]")
        trainer.fit(model_module, train_loader, val_loader)
        
        # Test
        console.print(f"\n[bold yellow]Testing {cfg.model.name}...[/bold yellow]")
        test_results = trainer.test(model_module, test_loader, ckpt_path='best')
        
        # Prepare results
        results = {
            'test_acc': test_results[0]['test_acc'] if test_results else 0.0,
            'val_acc': checkpoint_callback.best_model_score.item() if hasattr(checkpoint_callback, 'best_model_score') else 0.0,
            'train_acc': trainer.callback_metrics.get('train_acc', 0.0).item() if 'train_acc' in trainer.callback_metrics else 0.0,
            'total_params': total_params
        }
        
        # Clean up
        if logger:
            wandb.finish()
        
        return results
    
    def run_single_experiment(
        self,
        model_name: str,
        model_type: str = 'cnn',
        training_config: str = 'standard',
        overrides: Optional[Dict] = None,
        quick_test: bool = False
    ) -> Dict:
        """Run a single experiment.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            training_config: Training configuration
            overrides: Configuration overrides
            quick_test: Whether to run a quick test
            
        Returns:
            Experiment results dictionary
        """
        start_time = time.time()
        
        console.print(f"\n[bold cyan]Starting experiment: {model_name}[/bold cyan]")
        console.print("[dim]━" * 80 + "[/dim]")
        
        try:
            # Create configuration
            experiment_overrides = overrides.copy() if overrides else {}
            
            if quick_test:
                experiment_overrides.update({
                    'trainer.max_epochs': 2,
                    'trainer.limit_train_batches': 10,
                    'trainer.limit_val_batches': 5,
                    'training.early_stopping.patience': 1
                })
            
            # Create config with proper error handling
            try:
                cfg = self.create_experiment_config(
                    model_name=model_name,
                    model_type=model_type,
                    training_config=training_config,
                    overrides=experiment_overrides
                )
            except Exception as e:
                console.print(f"[red]Configuration error details:[/red]")
                console.print("\n[red]Stack trace:[/red]")
                console.print(traceback.format_exc())
                raise ValueError(f"Configuration creation failed: {e}")
            
            # Run the experiment
            result = self._execute_training(cfg)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            result.update({
                'status': 'success',
                'time_seconds': elapsed_time,
                'time_formatted': f"{elapsed_time/60:.1f} min",
                'model_name': model_name,
                'model_type': model_type
            })
            
            console.print(f"\n[green]✓ {model_name} completed successfully![/green]")
            console.print(f"[green]  Test accuracy: {result['test_acc']:.4f}[/green]")
            
            return result
            
        except Exception as e:
            console.print(f"\n[red]✗ Error running {model_name}: {e}[/red]")
            console.print("\n[red]Stack trace:[/red]")
            console.print(traceback.format_exc())
            
            return {
                'status': 'error',
                'error': str(e),
                'time_seconds': time.time() - start_time,
                'model_name': model_name,
                'model_type': model_type
            }
    
    def run_model_sweep(
        self,
        models: List[str],
        model_type: str = 'cnn',
        training_config: str = 'standard',
        overrides: Optional[Dict] = None,
        quick_test: bool = False
    ) -> Dict[str, Dict]:
        """Run experiments for multiple models.
        
        Args:
            models: List of model names to test
            model_type: Type of models
            training_config: Training configuration
            overrides: Configuration overrides
            quick_test: Whether to run quick tests
            
        Returns:
            Dictionary of results by model name
        """
        console.print(Panel.fit(
            f"[bold cyan]Model Sweep: {model_type.upper()}[/bold cyan]\n"
            f"[dim]Testing models: {', '.join(models)}[/dim]",
            border_style="blue"
        ))
        
        results = {}
        
        for model_name in models:
            result = self.run_single_experiment(
                model_name=model_name,
                model_type=model_type,
                training_config=training_config,
                overrides=overrides,
                quick_test=quick_test
            )
            results[model_name] = result
        
        return results
    
    def display_results(self, results: Dict[str, Dict]):
        """Display experiment results in a formatted table.
        
        Args:
            results: Dictionary of results by model name
        """
        # Create results table
        table = Table(title="Experiment Results", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Test Acc", style="green")
        table.add_column("Val Acc", style="yellow")
        table.add_column("Train Acc", style="blue")
        table.add_column("Params", style="magenta")
        table.add_column("Time", style="white")
        
        # Add results to table
        for model_name, result in results.items():
            status = "[green]✓[/green]" if result['status'] == 'success' else "[red]✗[/red]"
            test_acc = f"{result.get('test_acc', 0):.4f}" if result.get('test_acc') else "N/A"
            val_acc = f"{result.get('val_acc', 0):.4f}" if result.get('val_acc') else "N/A"
            train_acc = f"{result.get('train_acc', 0):.4f}" if result.get('train_acc') else "N/A"
            params = f"{result.get('total_params', 0):,}" if result.get('total_params') else "N/A"
            time_str = result.get('time_formatted', f"{result['time_seconds']/60:.1f} min")
            
            table.add_row(model_name, status, test_acc, val_acc, train_acc, params, time_str)
        
        console.print("\n")
        console.print(table)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("experiments") / f"unified_results_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {results_file}[/green]")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Unified CARS Thyroid Classification Runner')
    
    # Experiment mode
    parser.add_argument('mode', choices=['single', 'sweep', 'efficientnet', 'resnet', 'vit', 'all'],
                       help='Experiment mode: single model, model sweep, or all models')
    
    # Model selection
    parser.add_argument('--model', '-m', type=str, default='resnet18',
                       help='Model name for single mode (e.g., resnet18, efficientnet_b0, vit_tiny)')
    
    parser.add_argument('--model-type', type=str, choices=['cnn', 'vit'], default='cnn',
                       help='Model type (cnn or vit)')
    
    parser.add_argument('--models', '-M', nargs='+', type=str,
                       help='List of models for sweep mode')
    
    # Training configuration
    parser.add_argument('--training-config', '-t', type=str, default='standard',
                       help='Training configuration to use')
    
    # Quick test mode
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with limited epochs/batches')
    
    # Training overrides
    parser.add_argument('--epochs', '-e', type=int,
                       help='Override number of epochs')
    
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Override batch size')
    
    parser.add_argument('--lr', type=float,
                       help='Override learning rate')
    
    # Other options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for training')
    
    parser.add_argument('--wandb-mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases logging mode')
    
    parser.add_argument('--no-quality-aware', action='store_true',
                       help='Disable quality-aware preprocessing')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Get project root and config directory
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    
    if not config_dir.exists():
        console.print(f"[red]Error: Config directory not found at {config_dir}[/red]")
        console.print("[red]Please run this script from the project root directory[/red]")
        sys.exit(1)
    
    # Initialize runner
    runner = UnifiedExperimentRunner(config_dir)
    
    # Get available models
    available_models = runner.get_available_models()
    console.print(f"\n[cyan]Available CNN models:[/cyan] {', '.join(available_models['cnn'])}")
    console.print(f"[cyan]Available ViT models:[/cyan] {', '.join(available_models['vit'])}")
    
    # Build configuration overrides
    overrides = {}
    if args.epochs:
        overrides['training.num_epochs'] = args.epochs
    if args.batch_size:
        overrides['training.batch_size'] = args.batch_size
    if args.lr:
        overrides['training.optimizer.lr'] = args.lr
    if args.device != 'auto':
        overrides['device'] = args.device
    if args.wandb_mode != 'online':
        overrides['wandb.mode'] = args.wandb_mode
    if args.no_quality_aware:
        overrides['model.quality_aware'] = False
    
    # Run experiments based on mode
    results = {}
    
    if args.mode == 'single':
        # Determine model type if not specified
        if args.model in available_models['vit']:
            model_type = 'vit'
        elif args.model in available_models['cnn']:
            model_type = 'cnn'
        else:
            model_type = args.model_type
            
        # Single model experiment
        result = runner.run_single_experiment(
            model_name=args.model,
            model_type=model_type,
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        results[args.model] = result
        
    elif args.mode == 'vit':
        # ViT model sweep
        vit_models = available_models['vit']
        if not vit_models:
            console.print("[red]No ViT models found![/red]")
            sys.exit(1)
        
        results = runner.run_model_sweep(
            models=vit_models,
            model_type='vit',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'sweep':
        # Custom model sweep
        if not args.models:
            console.print("[red]Please specify models with --models for sweep mode[/red]")
            sys.exit(1)
        
        # Determine model type from first model
        if args.models[0] in available_models['vit']:
            model_type = 'vit'
        else:
            model_type = 'cnn'
        
        results = runner.run_model_sweep(
            models=args.models,
            model_type=model_type,
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'efficientnet':
        # EfficientNet sweep
        efficientnet_models = [m for m in available_models['cnn'] if m.startswith('efficientnet')]
        if not efficientnet_models:
            console.print("[red]No EfficientNet models found![/red]")
            sys.exit(1)
        
        results = runner.run_model_sweep(
            models=efficientnet_models,
            model_type='cnn',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'resnet':
        # ResNet sweep
        resnet_models = [m for m in available_models['cnn'] if m.startswith('resnet')]
        if not resnet_models:
            console.print("[red]No ResNet models found![/red]")
            sys.exit(1)
        
        results = runner.run_model_sweep(
            models=resnet_models,
            model_type='cnn',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'all':
        # All available models
        results = runner.run_model_sweep(
            models=available_models['cnn'],
            model_type='cnn',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
    
    # Display results
    runner.display_results(results)
    
    # Show recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    # Find best model
    successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
    if successful_results:
        best_model = max(successful_results.items(), key=lambda x: x[1].get('test_acc', 0))
        console.print(f"[green]• Best performing model: {best_model[0]} ({best_model[1].get('test_acc', 0):.4f} test accuracy)[/green]")
        
        # Compare with baseline
        if best_model[1].get('test_acc', 0) > 0.853:  # Better than ResNet18 baseline
            console.print("[green]• Model outperforms ResNet18 baseline (85.3%)![/green]")
        
        # Check for Swin models
        if best_model[0].startswith('swin'):
            console.print("• Swin Transformer shows promising results!")
            console.print("• Consider ensemble with CNN models for best performance")
    
    console.print("• Consider ensemble of top performing models")
    console.print("• Test with different augmentation levels")
    console.print("• Analyze per-quality-tier performance")
    
    console.print("\n[bold green]✓ All experiments complete![/bold green]")


if __name__ == "__main__":
    main()