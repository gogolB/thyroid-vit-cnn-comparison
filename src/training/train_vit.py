"""
PyTorch Lightning module for Vision Transformer training on thyroid classification.
Supports ViT, DeiT, and Swin Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from typing import Optional, Dict, Any
from omegaconf import DictConfig
import wandb

from src.models.vit import get_vit_model


class ThyroidViTModule(pl.LightningModule):
    """Lightning module for Vision Transformer thyroid classification."""
    
    def __init__(self, config: DictConfig):
        """Initialize the ViT module.
        
        Args:
            config: Hydra configuration object
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = self._create_model()
        
        # Loss function with label smoothing if specified
        label_smoothing = config.get('loss', {}).get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics (using underscores for compatibility)
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_acc = Accuracy(task='binary')
        
        # For attention visualization
        self.log_attention_maps = config.get('log_attention_maps', False)
        
    def _create_model(self) -> nn.Module:
        """Create the Vision Transformer model."""
        # Get model configuration
        model_config = self.config.model
        
        # Create model using factory function
        model = get_vit_model(
            model_name=model_config.name,
            img_size=self.config.dataset.image_size,
            in_chans=1,  # Grayscale
            num_classes=self.config.dataset.num_classes,
            drop_rate=model_config.get('drop_rate', 0.0),
            attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
            drop_path_rate=model_config.get('drop_path_rate', 0.1),
        )
        
        return model
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        
        # Update metrics
        self.val_acc(preds, labels)
        self.val_auc(probs, labels)
        self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True)
        
        # Log attention maps if requested (only for first batch)
        if self.log_attention_maps and batch_idx == 0 and hasattr(self.model, 'get_attention_maps'):
            self._log_attention_maps(images, labels)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        images, labels = batch
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Get optimizer config
        opt_config = self.config.training.optimizer
        
        # Create parameter groups for layer-wise learning rate decay if specified
        if self.config.training.get('layer_wise_lr_decay', False):
            param_groups = self._get_parameter_groups_with_decay()
        else:
            param_groups = self.model.parameters()
        
        # Create optimizer
        if opt_config._target_ == 'torch.optim.AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_config.lr,
                weight_decay=opt_config.get('weight_decay', 0.05),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            # Fallback to Adam
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_config.lr,
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        
        # Create scheduler if specified
        if 'scheduler' in self.config.training:
            sched_config = self.config.training.scheduler
            
            if sched_config._target_ == 'torch.optim.lr_scheduler.CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_config.get('T_max', self.config.training.num_epochs),
                    eta_min=sched_config.get('eta_min', 1e-6)
                )
            elif 'transformers' in sched_config._target_:
                # For transformers schedulers, we need to handle them differently
                # This is a placeholder - actual implementation would import transformers
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config.training.num_epochs,
                    eta_min=1e-6
                )
            else:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=30,
                    gamma=0.1
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        
        return optimizer
    
    def _get_parameter_groups_with_decay(self):
        """Get parameter groups with layer-wise learning rate decay."""
        # This is a simplified version - full implementation would decay by layer depth
        decay_rate = self.config.training.get('layer_decay', 0.75)
        base_lr = self.config.training.optimizer.lr
        
        # Group parameters by layer
        param_groups = []
        
        # Embedding parameters
        embed_params = []
        if hasattr(self.model, 'patch_embed'):
            embed_params.extend(self.model.patch_embed.parameters())
        if hasattr(self.model, 'cls_token'):
            embed_params.append(self.model.cls_token)
        if hasattr(self.model, 'pos_embed'):
            embed_params.append(self.model.pos_embed)
            
        if embed_params:
            param_groups.append({
                'params': embed_params,
                'lr': base_lr * (decay_rate ** 2)  # Lower LR for embeddings
            })
        
        # Transformer blocks (if accessible)
        if hasattr(self.model, 'blocks'):
            num_layers = len(self.model.blocks)
            for i, block in enumerate(self.model.blocks):
                layer_lr = base_lr * (decay_rate ** (num_layers - i - 1))
                param_groups.append({
                    'params': block.parameters(),
                    'lr': layer_lr
                })
        
        # Head parameters (highest learning rate)
        if hasattr(self.model, 'head'):
            param_groups.append({
                'params': self.model.head.parameters(),
                'lr': base_lr
            })
        
        # Fallback: if we can't identify layers, use all parameters
        if not param_groups:
            param_groups = [{
                'params': self.model.parameters(),
                'lr': base_lr
            }]
        
        return param_groups
    
    def _log_attention_maps(self, images, labels):
        """Log attention maps to wandb."""
        if not self.logger or not hasattr(self.logger.experiment, 'log'):
            return
            
        # Get attention maps
        self.model.eval()
        with torch.no_grad():
            _ = self.model(images[:4])  # Use first 4 images
            attention_maps = self.model.get_attention_maps()
        
        if attention_maps is None:
            return
        
        # Log attention from last layer
        if len(attention_maps) > 0:
            last_layer_attention = attention_maps[-1]  # Shape: (B, H, N, N)
            
            # Average over heads
            avg_attention = last_layer_attention.mean(dim=1)  # Shape: (B, N, N)
            
            # Create visualization
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            
            for i in range(min(4, avg_attention.shape[0])):
                attn = avg_attention[i].cpu().numpy()
                
                # Focus on CLS token attention (first row)
                cls_attn = attn[0, 1:]  # Skip CLS token itself
                
                # Reshape to 2D grid
                num_patches = int(np.sqrt(len(cls_attn)))
                cls_attn = cls_attn.reshape(num_patches, num_patches)
                
                # Plot
                im = axes[i].imshow(cls_attn, cmap='hot', interpolation='nearest')
                axes[i].set_title(f'Sample {i+1} (Label: {labels[i].item()})')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Log to wandb
            self.logger.experiment.log({
                "attention_maps": wandb.Image(fig),
                "global_step": self.global_step
            })
            
            plt.close(fig)
        
        self.model.train()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Log final test accuracy
        test_acc = self.test_acc.compute()
        self.log('final_test_acc', test_acc)
        
        # Print summary
        print(f"\nTest Accuracy: {test_acc:.4f}")