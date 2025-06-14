#!/usr/bin/env python3
"""
Unified Experiment Runner for CARS Thyroid Classification
Runs all CNN experiments directly without subprocesses
Supports ResNet, EfficientNet, DenseNet, and Inception models
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
from src.models.vit import get_vit_model
from src.training.train_vit import ThyroidViTModule



console = Console()


class UnifiedExperimentRunner:
    """Unified experiment runner for all CNN models."""
    
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
                models['cnn'].append(model_name)
        
        # Scan ViT models (Phase 3)
        vit_dir = self.config_dir / 'model' / 'vit'
        if vit_dir.exists():
            for config_file in vit_dir.glob('*.yaml'):
                model_name = config_file.stem
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
            model_name: Name of the model (e.g., 'efficientnet_b0', 'vit_tiny')
            model_type: Type of model ('cnn' or 'vit')
            training_config: Training configuration to use
            overrides: Additional configuration overrides
            
        Returns:
            Complete configuration object
        """
        try:
            # Ensure Hydra is clean
            GlobalHydra.instance().clear()
            
            # Initialize Hydra with config directory
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            
            # Create config with overrides
            config_overrides = [
                f"model={model_type}/{model_name}",
                f"training={training_config}"
            ]
            
            # Add any additional overrides
            if overrides:
                for key, value in overrides.items():
                    config_overrides.append(f"{key}={value}")
            
            # Compose configuration
            cfg = compose(config_name="config", overrides=config_overrides)
            
            # Convert to container and resolve manually
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            
            # Get project root
            project_root = self.config_dir.parent
            
            # Manually replace interpolations in the dictionary
            def replace_interpolations(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        obj[key] = replace_interpolations(value)
                elif isinstance(obj, list):
                    return [replace_interpolations(item) for item in obj]
                elif isinstance(obj, str):
                    # Replace common Hydra interpolations
                    if obj == "${hydra:runtime.cwd}/data":
                        return str(project_root / "data")
                    elif obj == "${hydra:runtime.cwd}/logs":
                        return str(project_root / "logs")
                    elif obj == "${hydra:runtime.cwd}/checkpoints":
                        return str(project_root / "checkpoints")
                    elif obj == "${paths.data_dir}/raw":
                        return str(project_root / "data" / "raw")
                    else:
                        return obj
                else:
                    return obj
                return obj
            
            # Apply replacements
            cfg_dict = replace_interpolations(cfg_dict)
            
            # ViT-specific configuration handling
            if model_type == 'vit':
                # Set ViT-specific model target
                cfg_dict['model']['_target_'] = 'src.models.vit.get_vit_model'
                cfg_dict['model']['model_name'] = model_name
                
                # Use AdamW optimizer for ViT
                cfg_dict['training']['optimizer']['_target_'] = 'torch.optim.AdamW'
                
                # ViT-specific training settings if not already overridden
                if 'training' not in cfg_dict:
                    cfg_dict['training'] = {}
                    
                # Use ViT-specific training config if available
                if training_config == 'standard' and (self.config_dir / 'training' / 'vit_standard.yaml').exists():
                    # Override with ViT-specific training
                    config_overrides = [
                        f"model={model_type}/{model_name}",
                        "training=vit_standard"
                    ]
                    cfg = compose(config_name="config", overrides=config_overrides)
                    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
                    cfg_dict = replace_interpolations(cfg_dict)
                
                # Layer-wise learning rate decay
                if 'layer_decay' in cfg_dict.get('training', {}):
                    cfg_dict['training']['layer_wise_lr_decay'] = True
                    
                # Ensure correct learning rate for ViT (typically lower)
                if 'optimizer' in cfg_dict.get('training', {}):
                    current_lr = cfg_dict['training']['optimizer'].get('lr', 0.001)
                    if current_lr > 0.001:  # If using CNN LR, reduce it
                        cfg_dict['training']['optimizer']['lr'] = 0.0005
            
            if model_type == 'vit' and 'deit' in model_name:
                # Special handling for DeiT models
                if cfg_dict['model'].get('pretrained', False):
                    # Use lower learning rate for pretrained models
                    cfg_dict['training']['optimizer']['lr'] = 0.0005
                    
                # Enable distillation if specified
                if cfg_dict['model'].get('distilled', False):
                    cfg_dict['training']['use_distillation'] = True
            
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
                console.print(f"[red]Configuration error details:[/red]")
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
            if 'test_acc' in result:
                console.print(f"  Test Accuracy: {result['test_acc']:.4f}")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            result = {
                'status': 'failed',
                'error': str(e),
                'time_seconds': elapsed_time,
                'model_name': model_name,
                'model_type': model_type
            }
            console.print(f"\n[red]✗ {model_name} failed: {e}[/red]")
            console.print("[red]Stack trace:[/red]")
            console.print(traceback.format_exc())
        
        finally:
            # Ensure Hydra is always cleaned up
            try:
                GlobalHydra.instance().clear()
            except:
                pass
        
        console.print("[dim]━" * 80 + "[/dim]\n")
        return result
    
    def _execute_training(self, cfg: DictConfig) -> Dict:
        """Execute the actual training process.
        
        Args:
            cfg: Configuration object
            
        Returns:
            Training results
        """
        # Determine model type from config
        model_type = 'vit' if 'vit' in cfg.model.name else 'cnn'
        
        # Print configuration
        console.print(Panel.fit(
            f"[bold cyan]{model_type.upper()} Training: {cfg.model.name}[/bold cyan]\n"
            f"[dim]Dataset: {cfg.dataset.name}[/dim]",
            border_style="blue"
        ))
        
        # Set seed for reproducibility
        pl.seed_everything(cfg.seed, workers=True)
        
        # Device selection with MPS support
        device = get_device(cfg.device)
        console.print(f"\n[cyan]Device Information:[/cyan]")
        console.print(device_info())
        
        # Create data loaders with quality-aware preprocessing
        console.print("\n[cyan]Creating data loaders...[/cyan]")
        
        # Get transforms
        quality_report_path = Path(cfg.paths.data_dir).parent / 'reports' / 'quality_report.json'
        
        if cfg.model.get('quality_aware', True) and quality_report_path.exists():
            console.print("[cyan]Using quality-aware preprocessing[/cyan]")
            quality_path = quality_report_path
        else:
            console.print("[yellow]Quality-aware preprocessing disabled or report not found[/yellow]")
            quality_path = None
        
        # Get augmentation level from config
        augmentation_level = cfg.training.get('augmentation_level', 'medium')
        
        # For ViT models, we might want stronger augmentation
        if model_type == 'vit' and augmentation_level == 'medium':
            augmentation_level = 'strong'
            console.print("[cyan]Using strong augmentation for ViT model[/cyan]")
        
        train_transform = create_quality_aware_transform(
            target_size=cfg.dataset.image_size,
            quality_report_path=quality_path,
            augmentation_level=augmentation_level,
            split='train'
        )
        
        val_transform = create_quality_aware_transform(
            target_size=cfg.dataset.image_size,
            quality_report_path=quality_path,
            augmentation_level='none',  # No augmentation for validation
            split='val'
        )
        
        # Create data loaders
        data_loaders = create_data_loaders(
            root_dir=cfg.dataset.path,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.dataset.num_workers,
            transform_train=train_transform,
            transform_val=val_transform,
            target_size=cfg.dataset.image_size,
            normalize=False,  # Normalization handled in transforms
            patient_level_split=cfg.dataset.patient_level_split
        )
        
        console.print(f"[green]✓ Data loaders created[/green]")
        console.print(f"  Train samples: {len(data_loaders['train'].dataset)}")
        console.print(f"  Val samples: {len(data_loaders['val'].dataset)}")
        console.print(f"  Test samples: {len(data_loaders['test'].dataset)}")
        
        # Create model based on type
        console.print("\n[cyan]Creating model...[/cyan]")
        
        # Determine model type
        model_name = cfg.model.name
        if model_name in ['vit_tiny', 'vit_small', 'vit_base',
                        'deit_tiny', 'deit_small', 'deit_base',
                        'swin_tiny', 'swin_small', 'swin_base', "swin_large", 'swin_medical']:
            model = ThyroidViTModule(cfg)
        else:
            model = ThyroidCNNModule(cfg)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"[green]✓ Model created[/green]")
        console.print(f"  Total parameters: {total_params:,}")
        console.print(f"  Trainable parameters: {trainable_params:,}")
        console.print(f"  Model type: {model_type.upper()}")
        
        # Setup callbacks
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
        if cfg.training.get('early_stopping', None):
            early_stop_callback = EarlyStopping(
                monitor='val_acc',
                min_delta=cfg.training.early_stopping.min_delta,
                patience=cfg.training.early_stopping.patience,
                mode='max',
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        # Rich progress bar
        callbacks.append(RichProgressBar())
        
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        
        # Best checkpoint callback
        best_checkpoint_callback = BestCheckpointCallback(checkpoint_dir=Path(cfg.paths.checkpoint_dir))
        callbacks.append(best_checkpoint_callback)
        
        # Setup logger
        wandb_logger = None
        if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                name=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=OmegaConf.to_container(cfg, resolve=True),
                save_dir=cfg.paths.log_dir,
                mode=cfg.wandb.mode
            )
        
        # Update trainer config for device
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        
        # Remove Hydra-specific keys and those we'll pass separately
        trainer_config.pop('_target_', None)
        trainer_config.pop('callbacks', None)
        trainer_config.pop('logger', None)
        
        # Remove deprecated PyTorch Lightning parameters
        deprecated_params = [
            'progress_bar_refresh_rate',  # Deprecated in PL 2.0+
            'weights_summary',  # Deprecated, use enable_model_summary
            'resume_from_checkpoint',  # Deprecated, use ckpt_path
            'truncated_bptt_steps',  # Deprecated
            'weights_save_path',  # Deprecated
        ]
        
        for param in deprecated_params:
            trainer_config.pop(param, None)
        
        # Handle device-specific settings
        if device.type == 'mps':
            trainer_config['accelerator'] = 'mps'
            trainer_config['precision'] = 32  # MPS doesn't support mixed precision yet
        elif device.type == 'cuda':
            trainer_config['accelerator'] = 'gpu'
            trainer_config['precision'] = '16-mixed' if cfg.training.get('precision', '16-mixed') == '16-mixed' else 32
        else:
            trainer_config['accelerator'] = 'cpu'
            trainer_config['precision'] = 32
            
        # Create trainer
        trainer = pl.Trainer(
            **trainer_config,
            callbacks=callbacks,
            logger=wandb_logger
        )
        
        # Train
        console.print("\n[cyan]Starting training...[/cyan]")
        trainer.fit(
            model,
            train_dataloaders=data_loaders['train'],
            val_dataloaders=data_loaders['val']
        )
        
        # Test
        console.print("\n[cyan]Running test evaluation...[/cyan]")
        test_results = trainer.test(
            model,
            dataloaders=data_loaders['test'],
            ckpt_path='best'
        )
        
        # Extract results
        result = {
            'train_acc': float(trainer.callback_metrics.get('train_acc', 0)),
            'val_acc': float(trainer.callback_metrics.get('val_acc', 0)),
            'test_acc': float(test_results[0].get('test_acc', 0)),
            'best_epoch': checkpoint_callback.best_model_score.item() if hasattr(checkpoint_callback, 'best_model_score') else 0,
            'total_epochs': trainer.current_epoch + 1,
        }
        
        # Clean up
        if wandb_logger:
            wandb.finish()
        
        return result

    
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
        
        # Use ViT-specific training config if available
        training_config = 'vit_standard' if (config_dir / 'training' / 'vit_standard.yaml').exists() else args.training_config
        
        results = runner.run_model_sweep(
            models=vit_models,
            model_type='vit',
            training_config=training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'sweep':
        # Custom model sweep
        if args.models:
            # Determine model types
            for model in args.models:
                if model in available_models['vit']:
                    model_type = 'vit'
                elif model in available_models['cnn']:
                    model_type = 'cnn'
                else:
                    console.print(f"[yellow]Warning: Unknown model {model}, skipping[/yellow]")
                    continue
                
                result = runner.run_single_experiment(
                    model_name=model,
                    model_type=model_type,
                    training_config=args.training_config,
                    overrides=overrides,
                    quick_test=args.quick_test
                )
                results[model] = result
        else:
            console.print("[red]No models specified for sweep mode![/red]")
            sys.exit(1)
        
    elif args.mode == 'efficientnet':
        # EfficientNet sweep
        efficientnet_models = [m for m in available_models['cnn'] if 'efficientnet' in m]
        if not efficientnet_models:
            console.print("[red]No EfficientNet models found![/red]")
            sys.exit(1)
        
        # Use EfficientNet-specific training config
        training_config = 'efficientnet' if args.training_config == 'standard' else args.training_config
        
        results = runner.run_model_sweep(
            models=efficientnet_models,
            model_type='cnn',
            training_config=training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'resnet':
        # ResNet sweep
        resnet_models = [m for m in available_models['cnn'] if 'resnet' in m]
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
    
    console.print("• Consider ensemble of top performing models")
    console.print("• Test with different augmentation levels")
    console.print("• Analyze per-quality-tier performance")
    
    console.print("\n[bold green]✓ All experiments complete![/bold green]")


if __name__ == "__main__":
    main()