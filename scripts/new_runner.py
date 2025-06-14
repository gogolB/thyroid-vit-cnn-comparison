#!/usr/bin/env python3
"""
New Unified Experiment Runner for CARS Thyroid Classification
Clean implementation supporting CNN, ViT, DeiT, and Swin models
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


class NewUnifiedRunner:
    """Clean unified experiment runner for all models."""
    
    def __init__(self, config_dir: Path):
        """Initialize the experiment runner."""
        self.config_dir = config_dir.absolute()
        self.results = {}
        
        # Ensure Hydra is clean
        GlobalHydra.instance().clear()
        
        console.print(Panel.fit(
            "[bold cyan]Unified CARS Thyroid Classification Runner[/bold cyan]\n"
            "[dim]Clean implementation - All models supported[/dim]",
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
                if model_name != '__init__':
                    models['cnn'].append(model_name)
        
        # Scan ViT models
        vit_dir = self.config_dir / 'model' / 'vit'
        if vit_dir.exists():
            for config_file in vit_dir.glob('*.yaml'):
                model_name = config_file.stem
                if model_name != '__init__':
                    models['vit'].append(model_name)
        
        return models
    
    def create_config(self, model_name: str, model_type: str = 'cnn', 
                     training_config: str = 'standard', overrides: Optional[Dict] = None) -> DictConfig:
        """Create configuration for an experiment."""
        try:
            # Clear Hydra
            GlobalHydra.instance().clear()
            
            # Initialize Hydra (no context manager!)
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            
            # Build overrides list
            config_overrides = [
                f"model={model_type}/{model_name}",
                f"training={training_config}",
                "hydra.run.dir=outputs/${model.name}/${now:%Y-%m-%d_%H-%M-%S}",
                "hydra.job.chdir=false"
            ]
            
            # Add custom overrides
            if overrides:
                for key, value in overrides.items():
                    config_overrides.append(f"{key}={value}")
            
            # Compose configuration
            cfg = compose(config_name="config", overrides=config_overrides)
            
            # Resolve paths manually to avoid interpolation issues
            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            project_root = self.config_dir.parent
            
            # Fix common path interpolations
            if 'paths' in cfg_dict:
                if 'data_dir' in cfg_dict['paths']:
                    cfg_dict['paths']['data_dir'] = str(project_root / "data")
                if 'log_dir' in cfg_dict['paths']:
                    cfg_dict['paths']['log_dir'] = str(project_root / "logs")
                if 'checkpoint_dir' in cfg_dict['paths']:
                    cfg_dict['paths']['checkpoint_dir'] = str(project_root / "checkpoints")
            
            if 'dataset' in cfg_dict and 'path' in cfg_dict['dataset']:
                cfg_dict['dataset']['path'] = str(project_root / "data" / "raw")
            
            # Convert back to DictConfig
            cfg = OmegaConf.create(cfg_dict)
            
            # Ensure directories exist
            Path(cfg.paths.data_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.paths.log_dir).mkdir(parents=True, exist_ok=True)
            Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            return cfg
            
        except Exception as e:
            console.print(f"[red]Error creating config: {e}[/red]")
            raise
        finally:
            GlobalHydra.instance().clear()
    
    def run_experiment(self, model_name: str, model_type: str = 'cnn',
                      training_config: str = 'standard', overrides: Optional[Dict] = None,
                      quick_test: bool = False) -> Dict:
        """Run a single experiment."""
        start_time = time.time()
        
        console.print(f"\n[bold cyan]Starting experiment: {model_name}[/bold cyan]")
        console.print("[dim]━" * 80 + "[/dim]")
        
        try:
            # Handle quick test
            if quick_test:
                if not overrides:
                    overrides = {}
                overrides.update({
                    'trainer.max_epochs': 2,
                    'trainer.limit_train_batches': 10,
                    'trainer.limit_val_batches': 5,
                    'training.early_stopping.patience': 1
                })
            
            # Create configuration
            cfg = self.create_config(model_name, model_type, training_config, overrides)
            
            # Execute training
            result = self._train_model(cfg)
            
            # Add timing and status
            elapsed_time = time.time() - start_time
            result.update({
                'status': 'success',
                'time_seconds': elapsed_time,
                'time_formatted': f"{elapsed_time/60:.1f} min",
                'model_name': model_name,
                'model_type': model_type
            })
            
            console.print(f"\n[green]✓ {model_name} completed successfully![/green]")
            
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
            console.print(traceback.format_exc())
        
        console.print("[dim]━" * 80 + "[/dim]\n")
        return result
    
    def _train_model(self, cfg: DictConfig) -> Dict:
        """Execute the actual training."""
        # Set device
        device = get_device()
        console.print(f"[cyan]Using device: {device}[/cyan]")
        
        # Set seed
        pl.seed_everything(cfg.seed, workers=True)
        
        # Create data loaders
        console.print("[yellow]Creating data loaders...[/yellow]")
        
        # Setup transforms
        quality_report_path = Path(cfg.paths.data_dir).parent / 'reports' / 'quality_report.json'
        
        if cfg.model.get('quality_aware', True) and quality_report_path.exists():
            console.print("[cyan]Using quality-aware preprocessing[/cyan]")
            quality_path = quality_report_path
        else:
            console.print("[yellow]Quality-aware preprocessing disabled or report not found[/yellow]")
            quality_path = None
        
        # Create transforms
        augmentation_level = cfg.training.get('augmentation_level', 'medium')
        
        train_transform = create_quality_aware_transform(
            target_size=cfg.dataset.image_size,
            quality_report_path=quality_path,
            augmentation_level=augmentation_level,
            split='train'
        )
        
        val_transform = create_quality_aware_transform(
            target_size=cfg.dataset.image_size,
            quality_report_path=quality_path,
            augmentation_level='none',
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
            normalize=cfg.dataset.normalize,
            patient_level_split=cfg.dataset.patient_level_split
        )
        
        console.print(f"[green]✓ Data loaders created[/green]")
        console.print(f"  Train: {len(data_loaders['train'].dataset)} samples")
        console.print(f"  Val: {len(data_loaders['val'].dataset)} samples")
        console.print(f"  Test: {len(data_loaders['test'].dataset)} samples")
        
        # Create model
        console.print("\n[cyan]Creating model...[/cyan]")
        
        # Determine model type and create appropriate module
        model_name = cfg.model.name
        model_type = cfg.model.get('type', None)
        
        # Check if it's a Vision Transformer model
        is_vit = (
            model_type in ['vit', 'deit', 'swin', 'vision_transformer'] or
            model_name.startswith(('vit_', 'deit_', 'swin_')) or
            model_name in ['vit_tiny', 'vit_small', 'vit_base',
                          'deit_tiny', 'deit_small', 'deit_base',
                          'swin_tiny', 'swin_small', 'swin_base', 'swin_large', 'swin_medical']
        )
        
        if is_vit:
            if not VIT_AVAILABLE:
                raise ImportError("Vision Transformer models not available. Please install required dependencies.")
            model = ThyroidViTModule(cfg)
            console.print(f"[green]✓ Created Vision Transformer model: {model_name}[/green]")
        else:
            model = ThyroidCNNModule(cfg)
            console.print(f"[green]✓ Created CNN model: {model_name}[/green]")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"  Total parameters: {total_params:,}")
        console.print(f"  Trainable parameters: {trainable_params:,}")
        
        # Setup callbacks
        callbacks = []
        
        # Checkpoint callback
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
            early_stop = EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
                min_delta=cfg.training.early_stopping.min_delta,
                verbose=True
            )
            callbacks.append(early_stop)
        
        # Progress bar
        callbacks.append(RichProgressBar())
        
        # LR monitor
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        
        # Best checkpoint callback
        callbacks.append(BestCheckpointCallback(checkpoint_dir=Path(cfg.paths.checkpoint_dir)))
        
        # Setup logger
        logger = None
        if cfg.get('wandb', {}).get('mode', 'disabled') != 'disabled':
            logger = WandbLogger(
                project=cfg.wandb.project,
                name=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=OmegaConf.to_container(cfg, resolve=True),
                save_dir=cfg.paths.log_dir,
                mode=cfg.wandb.mode
            )
        
        # Setup trainer
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        
        # Remove deprecated params
        for param in ['progress_bar_refresh_rate', 'weights_summary', 'gpus']:
            trainer_config.pop(param, None)
        
        # Create trainer
        trainer = pl.Trainer(
            **trainer_config,
            callbacks=callbacks,
            logger=logger
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
        
        # Gather results
        results = {
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
        
        # Add metrics
        if hasattr(trainer, 'callback_metrics'):
            metrics = trainer.callback_metrics
            if 'val_acc' in metrics:
                results['val_acc'] = float(metrics['val_acc'])
            if 'train_acc' in metrics:
                results['train_acc'] = float(metrics['train_acc'])
        
        # Add test results
        if test_results and len(test_results) > 0:
            if 'test_acc' in test_results[0]:
                results['test_acc'] = float(test_results[0]['test_acc'])
        
        # Cleanup
        if logger:
            wandb.finish()
        
        return results
    
    def run_sweep(self, models: List[str], model_type: str = 'cnn',
                  training_config: str = 'standard', overrides: Optional[Dict] = None,
                  quick_test: bool = False) -> Dict[str, Dict]:
        """Run experiments for multiple models."""
        console.print(Panel.fit(
            f"[bold cyan]Model Sweep: {model_type.upper()}[/bold cyan]\n"
            f"[dim]Testing models: {', '.join(models)}[/dim]",
            border_style="blue"
        ))
        
        results = {}
        for model_name in models:
            result = self.run_experiment(
                model_name=model_name,
                model_type=model_type,
                training_config=training_config,
                overrides=overrides,
                quick_test=quick_test
            )
            results[model_name] = result
        
        return results
    
    def display_results(self, results: Dict[str, Dict]):
        """Display results in a nice table."""
        table = Table(title="Experiment Results", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Test Acc", style="green")
        table.add_column("Val Acc", style="yellow")
        table.add_column("Time", style="white")
        
        for model_name, result in results.items():
            status = "[green]✓[/green]" if result['status'] == 'success' else "[red]✗[/red]"
            test_acc = f"{result.get('test_acc', 0):.4f}" if result.get('test_acc') else "N/A"
            val_acc = f"{result.get('val_acc', 0):.4f}" if result.get('val_acc') else "N/A"
            time_str = result.get('time_formatted', f"{result.get('time_seconds', 0)/60:.1f} min")
            
            table.add_row(model_name, status, test_acc, val_acc, time_str)
        
        console.print("\n")
        console.print(table)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='New Unified CARS Thyroid Classification Runner')
    
    parser.add_argument('mode', choices=['single', 'sweep', 'vit', 'cnn', 'all'],
                       help='Experiment mode')
    
    parser.add_argument('--model', '-m', type=str, default='resnet18',
                       help='Model name for single mode')
    
    parser.add_argument('--model-type', type=str, choices=['cnn', 'vit'], default='cnn',
                       help='Model type')
    
    parser.add_argument('--models', '-M', nargs='+', type=str,
                       help='List of models for sweep mode')
    
    parser.add_argument('--training-config', '-t', type=str, default='standard',
                       help='Training configuration')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with limited epochs')
    
    parser.add_argument('--epochs', '-e', type=int,
                       help='Override number of epochs')
    
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Override batch size')
    
    parser.add_argument('--lr', type=float,
                       help='Override learning rate')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use')
    
    parser.add_argument('--wandb-mode', type=str, default='offline',
                       choices=['online', 'offline', 'disabled'],
                       help='W&B logging mode')
    
    parser.add_argument('--no-quality-aware', action='store_true',
                       help='Disable quality-aware preprocessing')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    
    if not config_dir.exists():
        console.print(f"[red]Error: Config directory not found at {config_dir}[/red]")
        sys.exit(1)
    
    # Initialize runner
    runner = NewUnifiedRunner(config_dir)
    
    # Get available models
    available_models = runner.get_available_models()
    console.print(f"\n[cyan]Available CNN models:[/cyan] {', '.join(available_models['cnn'])}")
    console.print(f"[cyan]Available ViT models:[/cyan] {', '.join(available_models['vit'])}")
    
    # Build overrides
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
    
    # Run experiments
    results = {}
    
    if args.mode == 'single':
        # Determine model type
        if args.model in available_models['vit']:
            model_type = 'vit'
        elif args.model in available_models['cnn']:
            model_type = 'cnn'
        else:
            model_type = args.model_type
        
        result = runner.run_experiment(
            model_name=args.model,
            model_type=model_type,
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        results[args.model] = result
        
    elif args.mode == 'vit':
        # All ViT models
        vit_models = available_models['vit']
        if not vit_models:
            console.print("[red]No ViT models found![/red]")
            sys.exit(1)
        
        results = runner.run_sweep(
            models=vit_models,
            model_type='vit',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'cnn':
        # All CNN models
        cnn_models = available_models['cnn']
        if not cnn_models:
            console.print("[red]No CNN models found![/red]")
            sys.exit(1)
        
        results = runner.run_sweep(
            models=cnn_models,
            model_type='cnn',
            training_config=args.training_config,
            overrides=overrides,
            quick_test=args.quick_test
        )
        
    elif args.mode == 'sweep':
        # Custom model list
        if not args.models:
            console.print("[red]No models specified for sweep![/red]")
            sys.exit(1)
        
        for model in args.models:
            if model in available_models['vit']:
                model_type = 'vit'
            elif model in available_models['cnn']:
                model_type = 'cnn'
            else:
                console.print(f"[yellow]Warning: Unknown model {model}, skipping[/yellow]")
                continue
            
            result = runner.run_experiment(
                model_name=model,
                model_type=model_type,
                training_config=args.training_config,
                overrides=overrides,
                quick_test=args.quick_test
            )
            results[model] = result
    
    # Display results
    runner.display_results(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("experiments") / f"new_unified_results_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {results_file}[/green]")
    console.print("\n[bold green]✓ All experiments complete![/bold green]")


if __name__ == "__main__":
    main()
