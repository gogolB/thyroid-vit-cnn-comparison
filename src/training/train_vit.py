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
from omegaconf import DictConfig, OmegaConf
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
        # Resolve the loss config to avoid DictConfig issues
        from omegaconf import OmegaConf
        if 'loss' in config:
            loss_config = OmegaConf.to_container(config.loss, resolve=True)
            label_smoothing = float(loss_config.get('label_smoothing', 0.0))
        else:
            label_smoothing = 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics (using underscores for compatibility)
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_acc = Accuracy(task='binary')
        
        # For attention visualization
        self.log_attention_maps = bool(config.get('log_attention_maps', False))
        
        # Create model
        self.model = self._create_model()
        
        # Loss function with label smoothing if specified
        # Resolve the loss config to avoid DictConfig issues
        from omegaconf import OmegaConf
        # Handle loss config properly
        if 'loss' in config and config.loss is not None:
            if OmegaConf.is_config(config.loss):
                loss_config = OmegaConf.to_container(config.loss, resolve=True)
            else:
                loss_config = config.loss if isinstance(config.loss, dict) else {}
            label_smoothing = float(loss_config.get('label_smoothing', 0.0))
        else:
            label_smoothing = 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics (using underscores for compatibility)
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_acc = Accuracy(task='binary')
    
        # For attention visualization
        log_attention = OmegaConf.to_container(config, resolve=True).get('log_attention_maps', False)
        self.log_attention_maps = bool(log_attention)
        
    def _create_model(self) -> nn.Module:
        """Create the Vision Transformer model."""
        model_config = self.config.model
        model_name = model_config.name
        
        # Extract model parameters
        model_params = OmegaConf.to_container(model_config.get('params', {}), resolve=True)
        
        # Add pretrained flag if specified
        if hasattr(model_config, 'pretrained'):
            model_params['pretrained'] = model_config.pretrained
        
        # Add pretrained config if specified
        if hasattr(model_config, 'pretrained_cfg'):
            model_params['pretrained_cfg'] = OmegaConf.to_container(
                model_config.pretrained_cfg, resolve=True
            )
        
        # Create model using the factory function
        model = get_vit_model(model_name, **model_params)
        
        # Log model information
        param_count = sum(p.numel() for p in model.parameters())
        self.log_dict({
            'model/parameters': float(param_count),
            # 'model/name': model_name,  # Can't log strings
        }, on_epoch=False, on_step=False, logger=True)
        
        return model
    
    def forward(self, x):
        """Forward pass."""
        # Special handling for Swin models - they need 224x224
        if 'swin' in self.config.model.name.lower() and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.model(x)
    
    def _handle_logits_shape(self, logits, labels, context=""):
        """Helper method to handle various logit shapes consistently"""
        if logits.dim() == 1:
            # Single prediction? This is unusual
            if len(logits) == len(labels):
                # Assume binary classification with single output
                logits = torch.stack([1 - logits, logits], dim=1)
            else:
                raise ValueError(f"{context} Unexpected 1D logits shape: {logits.shape}")
        elif logits.dim() == 3:
            # [batch, 1, num_classes] -> [batch, num_classes]
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)
            else:
                print(f"{context} Warning: Unexpected 3D logits shape {logits.shape}, using first dimension")
                logits = logits[:, 0, :]
        elif logits.dim() == 4:
            # Spatial output - need to pool correctly
            print(f"{context} Got 4D logits {logits.shape}, applying global average pooling")
            
            # Check which dimension has num_classes (should be 2 for binary)
            if logits.shape[-1] == 2:  # [B, H, W, 2]
                logits = logits.mean(dim=[1, 2])  # -> [B, 2]
            elif logits.shape[1] == 2:  # [B, 2, H, W] 
                logits = logits.mean(dim=[2, 3])  # -> [B, 2]
            else:
                # Default: assume last dim is classes
                logits = logits.mean(dim=[1, 2])
        elif logits.dim() != 2:
            raise ValueError(f"{context} Unexpected logits shape: {logits.shape}")
        
        # Verify output shape
        if logits.shape[1] != 2:
            raise ValueError(f"{context} Expected 2 classes, got {logits.shape[1]}")
        
        return logits

    def training_step(self, batch, batch_idx):
        """Training step with DeiT support"""
        images, labels = batch
        
        # Defensive label handling
        if labels.dtype != torch.long:
            labels = labels.long()
        
        if labels.dim() == 0:  # Scalar
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        
        # Forward pass
        outputs = self.model(images)
        
        # Handle DeiT distillation outputs
        if isinstance(outputs, tuple) and len(outputs) == 2:
            # DeiT returns (class_logits, distill_logits) during training
            class_logits, distill_logits = outputs
            
            # Fix shapes using helper
            class_logits = self._handle_logits_shape(class_logits, labels, "[Training-Class]")
            distill_logits = self._handle_logits_shape(distill_logits, labels, "[Training-Distill]")
            
            # Calculate losses
            class_loss = self.criterion(class_logits, labels)
            distill_loss = self.criterion(distill_logits, labels)
            
            # Combine losses
            loss = 0.5 * class_loss + 0.5 * distill_loss
            
            # Use class logits for metrics
            logits = class_logits
        else:
            # Standard output - fix shape using helper
            logits = self._handle_logits_shape(outputs, labels, "[Training]")
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
        
        # Defensive label handling
        if labels.dtype != torch.long:
            labels = labels.long()
        
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        
        # Forward pass
        logits = self(images)
        
        # Fix shape using helper
        logits = self._handle_logits_shape(logits, labels, "[Validation]")
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        # Update metrics
        self.val_acc(preds, labels)
        self.val_auc(probs, labels)
        self.val_f1(preds, labels)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True)
        
        # Log attention maps if requested
        if self.log_attention_maps and batch_idx == 0 and hasattr(self.model, 'get_attention_maps'):
            self._log_attention_maps(images, labels)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images, labels = batch
        
        # Defensive label handling
        if labels.dtype != torch.long:
            labels = labels.long()
        
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        
        # Forward pass
        logits = self(images)
        
        # Fix shape using helper
        logits = self._handle_logits_shape(logits, labels, "[Test]")
        
        # Compute loss
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
        from omegaconf import OmegaConf
        
        # Convert the entire training config to a regular Python dict
        training_config = OmegaConf.to_container(self.config.training, resolve=True)
        opt_config = training_config['optimizer']
        
        # Create parameter groups for layer-wise learning rate decay if specified
        if training_config.get('layer_wise_lr_decay', False):
            param_groups = self._get_parameter_groups_with_decay()
        else:
            param_groups = self.model.parameters()
        
        # Create optimizer - all values are now primitive types
        if opt_config['_target_'] == 'torch.optim.AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=float(opt_config['lr']),
                weight_decay=float(opt_config.get('weight_decay', 0.05)),
                betas=tuple(opt_config.get('betas', [0.9, 0.999]))
            )
        else:
            # Fallback to Adam
            optimizer = torch.optim.Adam(
                param_groups,
                lr=float(opt_config['lr']),
                weight_decay=float(opt_config.get('weight_decay', 0.0))
            )
        
        # Create scheduler if specified
        if 'scheduler' in training_config:
            sched_config = training_config['scheduler']
            
            if sched_config['_target_'] == 'torch.optim.lr_scheduler.CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(sched_config.get('T_max', training_config['num_epochs'])),
                    eta_min=float(sched_config.get('eta_min', 1e-6))
                )
            elif 'transformers' in sched_config.get('_target_', ''):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(training_config['num_epochs']),
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
        from omegaconf import OmegaConf
        
        # Resolve training config
        training_config = OmegaConf.to_container(self.config.training, resolve=True)
        
        # This is a simplified version - full implementation would decay by layer depth
        decay_rate = float(training_config.get('layer_decay', {}).get('decay_rate', 0.75))
        base_lr = float(self.config.training.optimizer.lr)
        
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