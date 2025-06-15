#!/usr/bin/env python3
"""
CNN Training Script for CARS Thyroid Classification
Supports CUDA, MPS (Mac), and CPU training
Updated to include EfficientNet support
Fixed Hydra config path for subprocess execution from project root
"""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchmetrics import Accuracy, AUROC, F1Score
import timm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.panel import Panel

# Import project modules
from src.data.dataset import create_data_loaders
from src.data.quality_preprocessing import create_quality_aware_transform
from src.models.cnn.base_cnn import BaseCNN
from src.utils.training import get_device, device_info
from src.utils.training import BestCheckpointCallback
from src.training.lightning_modules import ThyroidCNNModule


console = Console()


# FIXED: Changed config_path from "../../configs" to "configs" to work with subprocess from project root
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Print configuration
    console.print(Panel.fit(
        f"[bold cyan]CNN Training: {cfg.model.name}[/bold cyan]\n"
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
        data_dir=cfg.paths.data_dir,
        config=cfg,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform  # Use same as val for test
    )
    
    console.print(f"[green]✓ Data loaders created[/green]")
    console.print(f"  Train samples: {len(data_loaders['train'].dataset)}")
    console.print(f"  Val samples: {len(data_loaders['val'].dataset)}")
    console.print(f"  Test samples: {len(data_loaders['test'].dataset)}")
    
    # Create model
    console.print("\n[cyan]Creating model...[/cyan]")
    model = ThyroidCNNModule(cfg)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[green]✓ Model created[/green]")
    console.print(f"  Total parameters: {total_params:,}")
    console.print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_acc:.4f}}",  # Changed val_acc to val_acc
        monitor='val_acc',  # Keep original metric name for monitoring
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if cfg.training.get('early_stopping', None):
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    best_checkpoint_callback = BestCheckpointCallback(checkpoint_dir=Path(cfg.paths.checkpoint_dir))
    callbacks.append(best_checkpoint_callback)
    
    # Setup logger
    logger = None
    if cfg.wandb.mode != 'disabled':
        logger = WandbLogger(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Update trainer config for device
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    
    # Remove Hydra-specific keys and those we'll pass separately
    trainer_config.pop('_target_', None)
    trainer_config.pop('callbacks', None)
    trainer_config.pop('logger', None)
    
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
    if not cfg.trainer.fast_dev_run:
        console.print("\n[cyan]Running test evaluation...[/cyan]")
        trainer.test(model, dataloaders=data_loaders['test'])
    
    # Save final results
    if logger:
        wandb.finish()
    
    console.print("\n[bold green]✓ Training complete![/bold green]")


if __name__ == "__main__":
    main()