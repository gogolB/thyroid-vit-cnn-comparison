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
from src.utils.device import get_device, device_info
from src.utils.checkpoint_utils import BestCheckpointCallback


console = Console()


class ThyroidCNNModule(pl.LightningModule):
    """PyTorch Lightning module for thyroid CNN training."""
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = self._create_model()
        
        # Loss function
        self.criterion = hydra.utils.instantiate(config.training.loss)
        
        # Metrics
        num_classes = config.dataset.get('num_classes', 2)  # Default to binary
        self.train_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass',
                                  num_classes=num_classes)
        self.val_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass',
                                num_classes=num_classes)
        self.val_auc = AUROC(task='binary' if num_classes == 2 else 'multiclass',
                             num_classes=num_classes)
        self.val_f1 = F1Score(task='binary' if num_classes == 2 else 'multiclass',
                              num_classes=num_classes)
        self.test_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass',
                                num_classes=num_classes)
        
        # For warmup
        self.warmup_epochs = config.training.get('warmup_epochs', 0)
        self.warmup_lr_scale = config.training.get('warmup_lr_scale', 0.1)
    
    def _create_model(self):
        """Create the CNN model based on configuration."""
        model_config = self.config.model
        
        # Handle ResNet models
        if 'resnet' in model_config.name.lower():
            from torchvision import models
            
            # Get the appropriate ResNet variant
            if model_config.name == 'resnet18':
                base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if model_config.pretrained else None)
            elif model_config.name == 'resnet34':
                base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if model_config.pretrained else None)
            elif model_config.name == 'resnet50':
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if model_config.pretrained else None)
            elif model_config.name == 'resnet101':
                base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if model_config.pretrained else None)
            else:
                raise ValueError(f"Unsupported ResNet variant: {model_config.name}")
            
            # Modify for single channel input
            base_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # Modify final layer for our number of classes
            num_features = base_model.fc.in_features
            num_classes = self.config.dataset.get('num_classes', 2)
            base_model.fc = torch.nn.Linear(num_features, num_classes)
            
            return base_model
            
        # Handle EfficientNet models
        elif 'efficientnet' in model_config.name.lower():
            from src.models.cnn.efficientnet import create_efficientnet_model
            
            # Create EfficientNet model using our factory function
            model = create_efficientnet_model(self.config)
            return model
            
        # Handle DenseNet models
        elif 'densenet' in model_config.name.lower():
            from src.models.cnn.densenet import create_densenet_model
            
            # Create DenseNet model using our factory function
            model = create_densenet_model(self.config)
            return model
            
        # Handle Inception models
        elif 'inception' in model_config.name.lower():
            from src.models.cnn.inception import create_inception_model
            model = create_inception_model(self.config)
            return model
            
        else:
            raise ValueError(f"Unsupported model: {model_config.name}")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        # For AUC, we need probabilities
        probs = torch.softmax(logits, dim=1)
        if self.config.dataset.get('num_classes', 2) == 2:
            # Binary classification - use probability of positive class
            auc = self.val_auc(probs[:, 1], y)
        else:
            # Multi-class classification
            auc = self.val_auc(probs, y)
            
        f1 = self.val_f1(preds, y)
        
        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/auc', auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        # Log metrics
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        # Get average test metrics
        test_acc = self.test_acc.compute()
        
        # Log final test results
        self.log('test/final_acc', test_acc)
        
        # Print results
        console.print(f"\n[bold green]Test Results:[/bold green]")
        console.print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Reset metrics
        self.test_acc.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with warmup support."""
        optimizer = hydra.utils.instantiate(
            self.config.training.optimizer,
            params=self.parameters()
        )
        
        scheduler = hydra.utils.instantiate(
            self.config.training.scheduler,
            optimizer=optimizer
        )
        
        # If warmup is enabled, we'll handle it manually in on_train_batch_start
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_batch_start(self, batch, batch_idx):
        """Handle learning rate warmup."""
        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + batch_idx / self.trainer.num_training_batches) / self.warmup_epochs
            lr_scale = self.warmup_lr_scale + (1 - self.warmup_lr_scale) * warmup_factor
            
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale if 'initial_lr' in param_group else param_group['lr']
    
    def on_train_epoch_start(self):
        """Store initial learning rates for warmup."""
        if self.current_epoch == 0 and self.warmup_epochs > 0:
            for param_group in self.optimizers().param_groups:
                param_group['initial_lr'] = param_group['lr']


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
        filename=f"{cfg.model.name}-{{epoch:02d}}-{{val_acc:.4f}}",  # Changed val/acc to val_acc
        monitor='val/acc',  # Keep original metric name for monitoring
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