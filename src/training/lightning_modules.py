"""
Consolidated PyTorch Lightning modules for the thyroid classification project.

This file will contain:
- ThyroidCNNModule
- ThyroidViTModule
- ThyroidDistillationModule
and potentially other LightningModules as the project evolves.
"""
# Standard library imports
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

# Third-party imports
import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import numpy as np
# Add other common imports that these modules might share, like from torchmetrics, etc.
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC, F1Score, Specificity, Recall, Precision, StatScores # Specific imports for ThyroidCNNModule

# Project-specific imports (will be needed when modules are moved here)
from src.models.vit.deit_models import DistillationLoss
from src.utils.models import TeacherModelLoader, EnsembleTeacher
from src.utils.training import get_device # Though get_device is not directly used in the provided ThyroidDistillationModule, it was in the original file.
# from src.utils.schedulers import CosineWarmupScheduler # This was a try-except import, will handle if needed later
from src.models.registry import ModelRegistry # Needed for ThyroidCNNModule
# from src.utils.some_utility import some_function

# Define common base class or interfaces if applicable, or just prepare for module definitions.
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
        self.num_classes = config.dataset.get('num_classes', 2)  # Default to binary
        self.metric_task = 'binary' if self.num_classes == 2 else 'multiclass'
        
        # Determine num_classes argument for AUROC (None for binary, self.num_classes for multiclass)
        auroc_num_classes_arg = self.num_classes if self.metric_task == 'multiclass' else None

        self.train_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        
        self.val_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.val_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg)
        self.val_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes)
        self.val_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.val_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes) # Sensitivity is Recall
        self.val_ppv = Precision(task=self.metric_task, num_classes=self.num_classes) # PPV is Precision
        self.val_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)

        self.test_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.test_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg) # Added for completeness if needed, though not explicitly requested for test_step logging here
        self.test_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes) # Added for completeness
        self.test_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.test_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes)
        self.test_ppv = Precision(task=self.metric_task, num_classes=self.num_classes)
        self.test_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)
        
        # For warmup
        self.warmup_epochs = config.training.get('warmup_epochs', 0)
        self.warmup_lr_scale = config.training.get('warmup_lr_scale', 0.1)
    
    def _create_model(self):
        """Create the CNN model based on configuration."""
        if not hasattr(self, 'config') or not hasattr(self.config, 'model'):
            raise AttributeError("ThyroidCNNModule requires 'self.config' with a 'model' attribute for ModelRegistry.")

        # Create a mutable copy of the model config to add in_channels
        # Ensure it handles both DictConfig and regular dict
        if isinstance(self.config.model, DictConfig):
            model_construction_config = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
        else: # Assuming it's a dict
            model_construction_config = self.config.model.copy()

        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'channels'):
            model_construction_config.in_channels = self.config.dataset.channels
        else:
            # Default to 3 if not specified, though ResNet model itself might default
            model_construction_config.in_channels = 3
            print("Warning: Dataset channels not found in config, defaulting model in_channels to 3 for CNN.")
        
        return ModelRegistry.create_model(model_construction_config)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x) # Renamed to 'outputs' to avoid confusion
        
        # Handle potential tuple output from models like InceptionV3 (main_output, aux_output)
        if isinstance(outputs, tuple):
            logits = outputs[0] # Use the main output for loss and metrics
        else:
            logits = outputs
            
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
        outputs = self(x) # Renamed to 'outputs'
        
        # Handle potential tuple output
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = self.criterion(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        # For AUC, we need probabilities
        probs = torch.softmax(logits, dim=1)
        # Use probs[:, 1] for binary AUROC, probs for multiclass
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        
        self.val_auc.update(auc_probs_input, y)
        self.val_f1.update(preds, y)
        self.val_specificity.update(preds, y)
        self.val_sensitivity.update(preds, y)
        self.val_ppv.update(preds, y)
        self.val_stat_scores.update(preds, y)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sensitivity', self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ppv', self.val_ppv, on_step=False, on_epoch=True, prog_bar=True)
        
        # Calculate and log NPV
        stat_scores_output = self.val_stat_scores.compute() # Compute once
        tn = stat_scores_output[2] # Index for True Negatives
        fn = stat_scores_output[3] # Index for False Negatives
        npv = tn / (tn + fn + 1e-6)
        self.log('val_npv', npv, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': self.val_acc.compute()}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x) # Renamed to 'outputs'

        # Handle potential tuple output
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc.update(preds, y)
        # For AUC and F1 on test, if needed (using probs for AUC)
        probs = torch.softmax(logits, dim=1)
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        self.test_auc.update(auc_probs_input, y)
        self.test_f1.update(preds, y)

        self.test_specificity.update(preds, y)
        self.test_sensitivity.update(preds, y)
        self.test_ppv.update(preds, y)
        self.test_stat_scores.update(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True)
        self.log('test_sensitivity', self.test_sensitivity, on_step=False, on_epoch=True)
        self.log('test_ppv', self.test_ppv, on_step=False, on_epoch=True)

        # Calculate and log NPV
        stat_scores_output = self.test_stat_scores.compute() # Compute once
        tn = stat_scores_output[2] # Index for True Negatives
        fn = stat_scores_output[3] # Index for False Negatives
        npv = tn / (tn + fn + 1e-6)
        self.log('test_npv', npv, on_step=False, on_epoch=True)
        
        # Return all computed metrics for potential aggregation
        # The trainer.test() call will aggregate these if logged with on_epoch=True
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(), # Return computed value
            'test_auc': self.test_auc.compute(),
            'test_f1': self.test_f1.compute(),
            'test_specificity': self.test_specificity.compute(),
            'test_sensitivity': self.test_sensitivity.compute(),
            'test_ppv': self.test_ppv.compute(),
            'test_npv': npv
        }
    
    def on_test_epoch_end(self):
        # Get average test metrics (already computed and logged in test_step if on_epoch=True)
        # This hook can be used for additional summary or cleanup if needed.
        # For now, main logging is in test_step.
        # Example: if you want to print all final metrics:
        final_test_acc = self.test_acc.compute()
        final_test_auc = self.test_auc.compute()
        final_test_f1 = self.test_f1.compute()
        final_test_specificity = self.test_specificity.compute()
        final_test_sensitivity = self.test_sensitivity.compute()
        final_test_ppv = self.test_ppv.compute()
        
        stat_scores_output = self.test_stat_scores.compute()
        final_test_npv = stat_scores_output[2] / (stat_scores_output[2] + stat_scores_output[3] + 1e-6)

        self.log('test_final_acc', final_test_acc) # Already logged via test_acc metric object
        # self.log('test_final_auc', final_test_auc) # etc.

        print(f"\nFinal Test Results (from on_test_epoch_end):")
        print(f"  Accuracy: {final_test_acc:.4f}")
        print(f"  AUROC: {final_test_auc:.4f}")
        print(f"  F1 Score: {final_test_f1:.4f}")
        print(f"  Specificity: {final_test_specificity:.4f}")
        print(f"  Sensitivity: {final_test_sensitivity:.4f}")
        print(f"  PPV: {final_test_ppv:.4f}")
        print(f"  NPV: {final_test_npv:.4f}")
        
        # Reset metrics (they are reset automatically by PL if logged with on_epoch=True)
        # self.test_acc.reset() # Not strictly necessary here
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler with warmup support."""
        # Assuming self.config.training.optimizer_params exists and has lr, weight_decay
        if not hasattr(self.config.training, 'optimizer_params'):
            raise ValueError("Missing optimizer_params in training configuration for ThyroidCNNModule.")

        opt_params = self.config.training.optimizer_params
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_params.get('lr', 0.001), # Default lr if not specified
            weight_decay=opt_params.get('weight_decay', 0.01) # Default wd if not specified
        )
        
        if hasattr(self.config.training, 'scheduler_params') and self.config.training.scheduler_params is not None:
            sched_params = self.config.training.scheduler_params
            # Example: CosineAnnealingLR, adapt if other schedulers are used via config
            if sched_params.get('name', 'cosineannealinglr').lower() == 'cosineannealinglr':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=sched_params.get('T_max', self.config.training.get('epochs', 100)), # Default T_max to total epochs
                    eta_min=sched_params.get('eta_min', 0.0)
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': self.config.training.get('monitor_metric', 'val_loss'),
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
        
        # If warmup is enabled, we'll handle it manually in on_train_batch_start
        # If no scheduler_params, just return optimizer
        return optimizer
    
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
class ThyroidViTModule(pl.LightningModule):
    """Lightning module for Vision Transformer thyroid classification."""
    
    def __init__(self, config: DictConfig, optimizer_params: Optional[Dict] = None):
        """Initialize the ViT module.
        
        Args:
            config: Hydra configuration object
            optimizer_params: Optional dictionary of optimizer parameters to override config
        """
        from src.utils.logging import get_logger
        logger = get_logger(__name__)
        
        super().__init__()
        self.config = config
        self.optimizer_params = optimizer_params
        self.save_hyperparameters()
        
        # Load optimizer params from JSON file if not provided
        if optimizer_params is None:
            try:
                import json
                with open('configs/vit_optimizer_params.json', 'r') as f:
                    optimizer_params = json.load(f)
                    self.optimizer_params = optimizer_params
                    logger.info(f"Loaded optimizer parameters from JSON: {optimizer_params}")
            except Exception as e:
                logger.error(f"Failed to load optimizer parameters from JSON: {e}")
                raise ValueError("Optimizer parameters not provided and failed to load from JSON")
        
        # Create model
        self.model = self._create_model()
        
        # Loss function with label smoothing if specified
        # Resolve the loss config to avoid DictConfig issues
        if 'loss' in config:
            loss_config = OmegaConf.to_container(config.loss, resolve=True)
            label_smoothing = float(loss_config.get('label_smoothing', 0.0))
        else:
            label_smoothing = 0.0
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Metrics
        self.num_classes = config.dataset.get('num_classes', 2)
        self.metric_task = 'binary' if self.num_classes == 2 else 'multiclass'

        auroc_num_classes_arg = self.num_classes if self.metric_task == 'multiclass' else None

        self.train_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        
        self.val_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.val_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg)
        self.val_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes)
        self.val_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.val_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes)
        self.val_ppv = Precision(task=self.metric_task, num_classes=self.num_classes)
        self.val_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)

        self.test_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.test_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg)
        self.test_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes)
        self.test_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.test_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes)
        self.test_ppv = Precision(task=self.metric_task, num_classes=self.num_classes)
        self.test_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)
        
        # For attention visualization
        self.log_attention_maps = bool(config.get('log_attention_maps', False))
        
    def _create_model(self) -> nn.Module:
        """Create the Vision Transformer model."""
        if not hasattr(self, 'config') or not hasattr(self.config, 'model'):
            raise AttributeError("ThyroidViTModule requires 'self.config' with a 'model' attribute for ModelRegistry.")

        # Create a mutable copy of the model config to add in_channels
        if isinstance(self.config.model, DictConfig):
            model_construction_config = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
        else: # Assuming it's a dict
            model_construction_config = self.config.model.copy()

        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'channels'):
            model_construction_config.in_channels = self.config.dataset.channels
        else:
            model_construction_config.in_channels = 3 # Default to 3 if not specified
            print("Warning: Dataset channels not found in config, defaulting model in_channels to 3 for ViT.")
        
        # Ensure img_size is also passed if the model needs it (some timm models infer from name, others need it explicitly)
        if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'img_size') and not hasattr(model_construction_config, 'img_size'):
            model_construction_config.img_size = self.config.dataset.img_size

        return ModelRegistry.create_model(model_construction_config)
    
    def forward(self, x):
        """Forward pass."""
        # Special handling for Swin models - they need 224x224
        # Ensure model name is accessible via self.config.model.name
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'name') and \
           'swin' in self.config.model.name.lower() and x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return self.model(x)
    
    def _handle_logits_shape(self, logits, labels, context=""):
        """Helper method to handle various logit shapes consistently"""
        if logits.dim() == 1:
            if len(logits) == len(labels):
                logits = torch.stack([1 - logits, logits], dim=1)
            else:
                raise ValueError(f"{context} Unexpected 1D logits shape: {logits.shape}")
        elif logits.dim() == 3:
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)
            else:
                print(f"{context} Warning: Unexpected 3D logits shape {logits.shape}, using first dimension")
                logits = logits[:, 0, :]
        elif logits.dim() == 4:
            print(f"{context} Got 4D logits {logits.shape}, applying global average pooling")
            if logits.shape[-1] == 2:
                logits = logits.mean(dim=[1, 2])
            elif logits.shape[1] == 2:
                logits = logits.mean(dim=[2, 3])
            else:
                logits = logits.mean(dim=[1, 2])
        elif logits.dim() != 2:
            raise ValueError(f"{context} Unexpected logits shape: {logits.shape}")
        
        if logits.shape[1] != 2: # Assuming binary classification
            raise ValueError(f"{context} Expected 2 classes, got {logits.shape[1]}")
        
        return logits

    def training_step(self, batch, batch_idx):
        """Training step with DeiT support"""
        images, labels = batch
        
        if labels.dtype != torch.long:
            labels = labels.long()
        
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        
        outputs = self.model(images)
        
        if isinstance(outputs, tuple) and len(outputs) == 2:
            class_logits, distill_logits = outputs
            class_logits = self._handle_logits_shape(class_logits, labels, "[Training-Class]")
            distill_logits = self._handle_logits_shape(distill_logits, labels, "[Training-Distill]")
            class_loss = self.criterion(class_logits, labels)
            distill_loss = self.criterion(distill_logits, labels)
            loss = 0.5 * class_loss + 0.5 * distill_loss
            logits = class_logits
        else:
            logits = self._handle_logits_shape(outputs, labels, "[Training]")
            loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, labels = batch
        
        if labels.dtype != torch.long:
            labels = labels.long()
        
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
            
        logits = self(images)
        logits = self._handle_logits_shape(logits, labels, "[Validation]")
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        # Use probs[:, 1] for binary AUROC, probs for multiclass
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        
        self.val_acc.update(preds, labels)
        self.val_auc.update(auc_probs_input, labels)
        self.val_f1.update(preds, labels)
        self.val_specificity.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_ppv.update(preds, labels)
        self.val_stat_scores.update(preds, labels)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
        self.log('val_specificity', self.val_specificity, on_epoch=True, prog_bar=True)
        self.log('val_sensitivity', self.val_sensitivity, on_epoch=True, prog_bar=True)
        self.log('val_ppv', self.val_ppv, on_epoch=True, prog_bar=True)

        stat_scores_output = self.val_stat_scores.compute()
        tn = stat_scores_output[2]
        fn = stat_scores_output[3]
        npv = tn / (tn + fn + 1e-6)
        self.log('val_npv', npv, on_epoch=True, prog_bar=True)
        
        if self.log_attention_maps and batch_idx == 0 and hasattr(self.model, 'get_attention_maps'):
            self._log_attention_maps(images, labels)
            
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        images, labels = batch
        
        if labels.dtype != torch.long:
            labels = labels.long()
            
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
            
        logits = self(images)
        logits = self._handle_logits_shape(logits, labels, "[Test]")
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        
        self.test_acc.update(preds, labels)
        probs = F.softmax(logits, dim=1)
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        self.test_auc.update(auc_probs_input, labels)
        self.test_f1.update(preds, labels)
        self.test_specificity.update(preds, labels)
        self.test_sensitivity.update(preds, labels)
        self.test_ppv.update(preds, labels)
        self.test_stat_scores.update(preds, labels)
        
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auc', self.test_auc, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_epoch=True)
        self.log('test_sensitivity', self.test_sensitivity, on_epoch=True)
        self.log('test_ppv', self.test_ppv, on_epoch=True)

        stat_scores_output = self.test_stat_scores.compute()
        tn = stat_scores_output[2]
        fn = stat_scores_output[3]
        npv = tn / (tn + fn + 1e-6)
        self.log('test_npv', npv, on_epoch=True)
        
        return {
            'test_loss': loss,
            'test_acc': self.test_acc.compute(),
            'test_auc': self.test_auc.compute(),
            'test_f1': self.test_f1.compute(),
            'test_specificity': self.test_specificity.compute(),
            'test_sensitivity': self.test_sensitivity.compute(),
            'test_ppv': self.test_ppv.compute(),
            'test_npv': npv
        }
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Prefer directly passed optimizer_params, fallback to config
        if self.optimizer_params is not None:
            opt_params = self.optimizer_params
        else:
            if not hasattr(self.config.training, 'optimizer_params'):
                raise ValueError("Missing optimizer_params in training configuration for ThyroidViTModule.")
            opt_params_config = self.config.training.optimizer_params
            # Convert DictConfig to dict if necessary
            opt_params = OmegaConf.to_container(opt_params_config, resolve=True) if isinstance(opt_params_config, DictConfig) else opt_params_config


        base_lr = float(opt_params.get('lr', 0.001))
        weight_decay = float(opt_params.get('weight_decay', 0.05))
        betas = tuple(opt_params.get('betas', (0.9, 0.999)))

        if self.config.training.get('layer_wise_lr_decay', False):
            param_groups = self._get_parameter_groups_with_decay() # This method uses self.config.training.optimizer.lr, needs update
        else:
            param_groups = self.model.parameters()
            
        # Assuming AdamW as a common default, adjust if other optimizers are primary
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr, # lr might be overridden by layer-wise decay if param_groups is structured
            weight_decay=weight_decay,
            betas=betas
        )
            
        if hasattr(self.config.training, 'scheduler_params') and self.config.training.scheduler_params is not None:
            sched_params_config = self.config.training.scheduler_params
            sched_params = OmegaConf.to_container(sched_params_config, resolve=True) if isinstance(sched_params_config, DictConfig) else sched_params_config
            
            scheduler_name = sched_params.get('name', 'cosineannealinglr').lower()

            if scheduler_name == 'cosineannealinglr':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(sched_params.get('T_max', self.config.training.get('epochs', 100))),
                    eta_min=float(sched_params.get('eta_min', 1e-6))
                )
            # Add other scheduler types here if needed
            else:
                # Default or fallback scheduler if name not recognized
                print(f"Warning: Scheduler '{scheduler_name}' not recognized or configured. No scheduler will be used.")
                return optimizer # Return only optimizer if scheduler config is problematic

            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}
        
        return optimizer
    
    def _get_parameter_groups_with_decay(self):
        """Get parameter groups with layer-wise learning rate decay."""
        training_config = OmegaConf.to_container(self.config.training, resolve=True)
        decay_rate = float(self.config.training.get('layer_decay', {}).get('decay_rate', 0.75))
        # Ensure optimizer_params is used here
        base_lr = float(self.config.training.optimizer_params.get('lr', 0.001))
        param_groups = []
        embed_params = []
        
        if hasattr(self.model, 'patch_embed'):
            embed_params.extend(self.model.patch_embed.parameters())
        if hasattr(self.model, 'cls_token'):
            embed_params.append(self.model.cls_token)
        if hasattr(self.model, 'pos_embed'):
            embed_params.append(self.model.pos_embed)
            
        if embed_params:
            param_groups.append({'params': embed_params, 'lr': base_lr * (decay_rate ** 2)})
            
        if hasattr(self.model, 'blocks'):
            num_layers = len(self.model.blocks)
            for i, block in enumerate(self.model.blocks):
                layer_lr = base_lr * (decay_rate ** (num_layers - i - 1))
                param_groups.append({'params': block.parameters(), 'lr': layer_lr})
                
        if hasattr(self.model, 'head'):
            param_groups.append({'params': self.model.head.parameters(), 'lr': base_lr})
            
        if not param_groups:
            param_groups = [{'params': self.model.parameters(), 'lr': base_lr}]
            
        return param_groups
    
    def _log_attention_maps(self, images, labels):
        """Log attention maps to wandb."""
        if not self.logger or not hasattr(self.logger.experiment, 'log'):
            return
            
        self.model.eval()
        with torch.no_grad():
            _ = self.model(images[:4])
            if not hasattr(self.model, 'get_attention_maps'):
                print("Warning: Model does not have get_attention_maps method. Cannot log attention.")
                self.model.train()
                return
            attention_maps = self.model.get_attention_maps()
        
        if attention_maps is None:
            self.model.train()
            return
        
        if len(attention_maps) > 0:
            last_layer_attention = attention_maps[-1]
            avg_attention = last_layer_attention.mean(dim=1)
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            
            for i in range(min(4, avg_attention.shape[0])):
                attn = avg_attention[i].cpu().numpy()
                cls_attn = attn[0, 1:]
                
                if len(cls_attn) == 0 or int(np.sqrt(len(cls_attn)))**2 != len(cls_attn):
                    print(f"Warning: Cannot reshape cls_attn of length {len(cls_attn)} to a square grid for sample {i}.")
                    continue

                num_patches = int(np.sqrt(len(cls_attn)))
                cls_attn = cls_attn.reshape(num_patches, num_patches)
                
                im = axes[i].imshow(cls_attn, cmap='hot', interpolation='nearest')
                axes[i].set_title(f'Sample {i+1} (Label: {labels[i].item()})')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            self.logger.experiment.log({"attention_maps": wandb.Image(fig), "global_step": self.global_step})
            plt.close(fig)
            
        self.model.train()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        # Similar to ThyroidCNNModule, main logging is in test_step.
        # This hook can be used for additional summary.
        final_test_acc = self.test_acc.compute()
        final_test_auc = self.test_auc.compute()
        final_test_f1 = self.test_f1.compute()
        final_test_specificity = self.test_specificity.compute()
        final_test_sensitivity = self.test_sensitivity.compute()
        final_test_ppv = self.test_ppv.compute()
        
        stat_scores_output = self.test_stat_scores.compute()
        final_test_npv = stat_scores_output[2] / (stat_scores_output[2] + stat_scores_output[3] + 1e-6)

        # self.log('final_test_acc', final_test_acc) # Already logged via metric object

        print(f"\nFinal Test Results ViT (from on_test_epoch_end):")
        print(f"  Accuracy: {final_test_acc:.4f}")
        print(f"  AUROC: {final_test_auc:.4f}")
        print(f"  F1 Score: {final_test_f1:.4f}")
        print(f"  Specificity: {final_test_specificity:.4f}")
        print(f"  Sensitivity: {final_test_sensitivity:.4f}")
        print(f"  PPV: {final_test_ppv:.4f}")
        print(f"  NPV: {final_test_npv:.4f}")
# Standard library imports
# (already present, no change here)

# Third-party imports
# (already present, no change here)

# Project-specific imports
# (already present or added, no change here)


class ThyroidDistillationModule(pl.LightningModule):
    """
    Lightning module for knowledge distillation training.
    Trains a student model (typically DeiT) using knowledge from a pre-trained teacher.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the distillation module.
        
        Args:
            config: Hydra configuration object with distillation parameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Verify distillation is enabled
        if not config.distillation.get('enabled', False):
            raise ValueError("Distillation must be enabled in config")
        
        # Create student model
        self.student = self._create_student_model()
        
        # Load teacher model(s)
        self.teacher, self.teacher_metrics = self._load_teacher_model()
        
        # Freeze teacher weights
        if config.distillation.get('freeze_teacher', True):
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # Create loss functions
        self.criterion = self._create_loss_functions()
        
        # Setup metrics
        self._setup_metrics()
        
        # Progressive distillation schedule
        self.progressive_schedule = None
        if config.distillation.get('progressive_distillation', False):
            # Ensure progressive_schedule is a dict-like structure
            schedule_data = config.distillation.get('progressive_schedule', {})
            if isinstance(schedule_data, DictConfig):
                 self.progressive_schedule = {int(k): v for k, v in OmegaConf.to_container(schedule_data, resolve=True).items()}
            elif isinstance(schedule_data, dict):
                 self.progressive_schedule = {int(k): v for k, v in schedule_data.items()}
            else:
                self.progressive_schedule = {} # Default to empty if format is unexpected
                print(f"Warning: progressive_schedule format unexpected: {type(schedule_data)}")

        # Log numeric configuration values only
        self.log_dict({
            'config/alpha': config.distillation.alpha,
            'config/temperature': config.distillation.temperature,
        }, on_epoch=False, on_step=False)
        
        student_model_name_for_log = "N/A"
        if hasattr(config, 'student_model') and config.student_model is not None:
            student_model_name_for_log = config.student_model.get('name', 'N/A')
        
        print(f"Distillation Config - Student: {student_model_name_for_log}, "
              f"Teacher: {config.distillation.teacher_model_type}")
    
    def _create_student_model(self) -> nn.Module:
        """Create the student model using ModelRegistry."""
        if not hasattr(self, 'config') or not hasattr(self.config, 'student_model') or self.config.student_model is None:
            raise AttributeError("ThyroidDistillationModule requires 'self.config' with a 'student_model' attribute (DictConfig) for ModelRegistry.")
        
        student_model_config = self.config.student_model
        
        model = ModelRegistry.create_model(student_model_config)
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log_dict({
            'model/student_parameters': float(param_count),
            'model/student_trainable': float(trainable_count),
        }, on_epoch=False, on_step=False)
        
        student_model_name = student_model_config.get('name', 'UnknownStudentModel')
        print(f"Student model: {student_model_name}")
        print(f"Parameters: {param_count:,} (trainable: {trainable_count:,})")
        
        return model
    
    def _load_teacher_model(self) -> Tuple[nn.Module, Dict[str, float]]:
        """Load the pre-trained teacher model(s)."""
        print("\nLoading teacher model(s)...")
        
        current_device = get_device() # Default device
        if self.trainer and hasattr(self.trainer.strategy, 'root_device'):
            current_device = self.trainer.strategy.root_device
        elif hasattr(self, 'device') and self.device is not None: # Fallback to self.device if trainer not fully setup
            current_device = self.device

        teacher, metrics = TeacherModelLoader.create_teacher_from_config(
            self.config, 
            device=current_device,
            verbose=True
        )
        
        teacher.eval()
        
        if isinstance(teacher, EnsembleTeacher):
            total_params = sum(sum(p.numel() for p in t.parameters()) for t in teacher.teachers)
            self.log('model/teacher_ensemble_size', float(len(teacher.teachers)), on_epoch=False, on_step=False)
        else:
            total_params = sum(p.numel() for p in teacher.parameters())
        
        self.log('model/teacher_parameters', float(total_params), on_epoch=False, on_step=False)
        
        for key, value in metrics.items():
            self.log(f'teacher/{key}', float(value), on_epoch=False, on_step=False)
        
        return teacher, metrics
    
    def _create_loss_functions(self) -> DistillationLoss:
        """Create the distillation loss function."""
        dist_cfg = self.config.distillation
        
        label_smoothing = 0.0
        loss_config_source = None
        if hasattr(self.config, 'loss') and self.config.loss is not None:
            loss_config_source = self.config.loss
        elif hasattr(self.config, 'training') and hasattr(self.config.training, 'loss') and self.config.training.loss is not None:
            loss_config_source = self.config.training.loss
        
        if loss_config_source:
            if isinstance(loss_config_source, DictConfig): # OmegaConf DictConfig
                 label_smoothing = OmegaConf.to_container(loss_config_source).get('label_smoothing', 0.0)
            elif isinstance(loss_config_source, dict): # Standard dict
                 label_smoothing = loss_config_source.get('label_smoothing', 0.0)
            elif hasattr(loss_config_source, 'label_smoothing'): # Attribute access
                 label_smoothing = loss_config_source.label_smoothing
                 
        base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        criterion = DistillationLoss(
            base_criterion=base_criterion,
            teacher_model=None,
            distillation_type=dist_cfg.get('distillation_type', 'soft'),
            alpha=dist_cfg.get('alpha', 0.5),
            tau=dist_cfg.get('temperature', 4.0)
        )
        return criterion
    
    def _setup_metrics(self):
        """Setup metrics for tracking performance."""
        self.num_classes = self.config.dataset.get('num_classes', 2)
        self.metric_task = 'binary' if self.num_classes == 2 else 'multiclass'
        
        auroc_num_classes_arg = self.num_classes if self.metric_task == 'multiclass' else None

        self.train_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        
        self.val_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.val_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg)
        self.val_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes)
        self.val_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.val_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes)
        self.val_ppv = Precision(task=self.metric_task, num_classes=self.num_classes)
        self.val_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)
        
        self.test_acc = Accuracy(task=self.metric_task, num_classes=self.num_classes)
        self.test_auc = AUROC(task=self.metric_task, num_classes=auroc_num_classes_arg)
        self.test_f1 = F1Score(task=self.metric_task, num_classes=self.num_classes)
        self.test_specificity = Specificity(task=self.metric_task, num_classes=self.num_classes)
        self.test_sensitivity = Recall(task=self.metric_task, num_classes=self.num_classes)
        self.test_ppv = Precision(task=self.metric_task, num_classes=self.num_classes)
        self.test_stat_scores = StatScores(task=self.metric_task, num_classes=self.num_classes, average='macro' if self.metric_task == 'multiclass' else None)

        self.teacher_agreement = Accuracy(task=self.metric_task, num_classes=self.num_classes)
    
    def get_current_alpha(self) -> float:
        """Get current distillation weight based on progressive schedule."""
        if self.progressive_schedule is None or not self.progressive_schedule:
            return self.config.distillation.alpha
        
        current_epoch = self.current_epoch
        applicable_alpha = self.config.distillation.alpha 
        
        # Iterate through sorted schedule epochs to find the latest applicable alpha
        # self.progressive_schedule keys are already integers due to __init__ processing
        for epoch_threshold in sorted(self.progressive_schedule.keys()):
            if current_epoch >= epoch_threshold:
                applicable_alpha = self.progressive_schedule[epoch_threshold]
            else:
                # Since epochs are sorted, no need to check further
                break 
        return applicable_alpha
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.student(x)
    
    def get_teacher_outputs(self, x: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
        return teacher_outputs
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        teacher_outputs = self.get_teacher_outputs(images)
        student_outputs = self.student(images)
        
        if isinstance(student_outputs, tuple) and len(student_outputs) == 2:
            outputs_cls, outputs_dist = student_outputs
        else:
            outputs_cls = outputs_dist = student_outputs
        
        class_loss = self.criterion.base_criterion(outputs_cls, labels)
        
        if self.config.distillation.distillation_type == 'soft':
            T = self.config.distillation.temperature
            dist_loss = F.kl_div(
                F.log_softmax(outputs_dist / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=False # For recent PyTorch versions, ensure log_target behavior
            ) * (T ** 2)
        else: # hard distillation
            teacher_labels = teacher_outputs.argmax(dim=1)
            dist_loss = self.criterion.base_criterion(outputs_dist, teacher_labels)
        
        alpha = self.get_current_alpha()
        total_loss = (1 - alpha) * class_loss + alpha * dist_loss
        
        preds = outputs_cls.argmax(dim=1)
        acc = self.train_acc(preds, labels)
        teacher_preds = teacher_outputs.argmax(dim=1)
        agreement = self.teacher_agreement(preds, teacher_preds)
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=False, on_epoch=True)
        self.log('distill_loss', dist_loss, on_step=False, on_epoch=True)
        self.log('alpha', alpha, on_step=False, on_epoch=True)
        self.log('teacher_agreement', agreement, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.student(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        loss = self.criterion.base_criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        
        # For AUROC, probs are needed.
        # Ensure output is suitable for softmax (e.g. not already softmaxed)
        if outputs.ndim == 2 and outputs.shape[1] > 1: # Standard logits
            probs = F.softmax(outputs, dim=1)
        else: # Handle cases like single logit output for binary, or already probabilities
            probs = outputs # Or apply sigmoid if single logit for binary
            if probs.ndim == 1 or probs.shape[1] == 1: # Binary case, ensure two columns for AUROC if needed by metric
                 probs = torch.stack([1-probs.squeeze(), probs.squeeze()], dim=1) if self.val_auc.task == 'binary' and probs.ndim ==1 else probs


        self.val_acc.update(preds, labels)
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        self.val_auc.update(auc_probs_input, labels)
        self.val_f1.update(preds, labels)
        self.val_specificity.update(preds, labels)
        self.val_sensitivity.update(preds, labels)
        self.val_ppv.update(preds, labels)
        self.val_stat_scores.update(preds, labels)
        
        teacher_outputs = self.get_teacher_outputs(images)
        teacher_preds = teacher_outputs.argmax(dim=1)
        val_agreement = (preds == teacher_preds).float().mean() # This is a manual calculation, not using torchmetrics object
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_specificity', self.val_specificity, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sensitivity', self.val_sensitivity, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ppv', self.val_ppv, on_step=False, on_epoch=True, prog_bar=True)
        
        stat_scores_output = self.val_stat_scores.compute()
        tn = stat_scores_output[2]
        fn = stat_scores_output[3]
        npv = tn / (tn + fn + 1e-6)
        self.log('val_npv', npv, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val_teacher_agreement', val_agreement, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.student(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        preds = outputs.argmax(dim=1)
        
        self.test_acc.update(preds, labels)
        # For other metrics on test set
        probs = F.softmax(outputs, dim=1)
        auc_probs_input = probs[:, 1] if self.metric_task == 'binary' and probs.shape[1] == self.num_classes else probs
        self.test_auc.update(auc_probs_input, labels)
        self.test_f1.update(preds, labels)
        self.test_specificity.update(preds, labels)
        self.test_sensitivity.update(preds, labels)
        self.test_ppv.update(preds, labels)
        self.test_stat_scores.update(preds, labels)

        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_auc', self.test_auc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        self.log('test_specificity', self.test_specificity, on_step=False, on_epoch=True)
        self.log('test_sensitivity', self.test_sensitivity, on_step=False, on_epoch=True)
        self.log('test_ppv', self.test_ppv, on_step=False, on_epoch=True)

        stat_scores_output = self.test_stat_scores.compute()
        tn = stat_scores_output[2]
        fn = stat_scores_output[3]
        npv = tn / (tn + fn + 1e-6)
        self.log('test_npv', npv, on_step=False, on_epoch=True)

        return {
            'test_acc': self.test_acc.compute(), # Return computed value
            'test_auc': self.test_auc.compute(),
            'test_f1': self.test_f1.compute(),
            'test_specificity': self.test_specificity.compute(),
            'test_sensitivity': self.test_sensitivity.compute(),
            'test_ppv': self.test_ppv.compute(),
            'test_npv': npv
        }
    
    def configure_optimizers(self):
        if not hasattr(self.config.training, 'optimizer_params'):
            raise ValueError("Missing optimizer_params in training configuration for ThyroidDistillationModule.")
        
        opt_params_config = self.config.training.optimizer_params
        opt_params = OmegaConf.to_container(opt_params_config, resolve=True) if isinstance(opt_params_config, DictConfig) else opt_params_config

        base_lr = float(opt_params.get('lr', 0.001))
        weight_decay = float(opt_params.get('weight_decay', 0.05))
        betas = tuple(opt_params.get('betas', (0.9, 0.999)))

        if hasattr(self.student, 'get_parameter_groups'): # Assumes get_parameter_groups uses base_lr internally or lr_scale
            param_groups = self.student.get_parameter_groups(
                weight_decay=weight_decay, # Pass WD here
                # layer_decay is often handled inside get_parameter_groups based on its own config
            )
            # Ensure base_lr is applied if get_parameter_groups doesn't set it
            for group in param_groups:
                if 'lr' not in group: # If model's get_parameter_groups doesn't set lr
                    group['lr'] = base_lr * group.get('lr_scale', 1.0) # Apply base_lr with scaling
        else:
            param_groups = [{'params': self.student.parameters(), 'lr': base_lr}]

        # Defaulting to AdamW, as it's common for ViTs/DeiTs
        optimizer = torch.optim.AdamW(
            param_groups, # param_groups should have lr set per group
            # lr=base_lr, # Not needed if lr is in param_groups
            betas=betas,
            weight_decay=weight_decay # AdamW handles WD correctly
        )
        
        if hasattr(self.config.training, 'scheduler_params') and self.config.training.scheduler_params is not None:
            sched_params_config = self.config.training.scheduler_params
            sched_params = OmegaConf.to_container(sched_params_config, resolve=True) if isinstance(sched_params_config, DictConfig) else sched_params_config
            scheduler_name = sched_params.get('name', 'cosineannealinglr').lower()
            
            if scheduler_name == 'cosineannealinglr':
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=int(sched_params.get('T_max', self.config.training.get('epochs', 100))),
                    eta_min=float(sched_params.get('eta_min', 1e-6))
                )
            elif scheduler_name == 'cosine_warmup':
                 try:
                    from src.utils.schedulers import CosineWarmupScheduler
                    scheduler = CosineWarmupScheduler(
                        optimizer,
                        warmup_epochs=int(sched_params.get('warmup_epochs', 5)),
                        max_epochs=int(self.config.training.get('epochs', 100)),
                        min_lr=float(sched_params.get('min_lr', 1e-6))
                    )
                 except ImportError:
                    print("CosineWarmupScheduler not found, falling back to CosineAnnealingLR.")
                    scheduler = CosineAnnealingLR(optimizer, T_max=int(self.config.training.get('epochs', 100)), eta_min=1e-6)
            else:
                print(f"Warning: Scheduler '{scheduler_name}' not recognized. No scheduler will be used.")
                return optimizer

            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
            }
        
        return optimizer
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if hasattr(self.config, 'distillation') and self.config.distillation is not None:
            checkpoint['distillation_config'] = OmegaConf.to_container(
                self.config.distillation, resolve=True
            )
        if hasattr(self, 'teacher_metrics'):
             checkpoint['teacher_metrics'] = self.teacher_metrics
        checkpoint['current_alpha'] = self.get_current_alpha()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'teacher_metrics' in checkpoint:
            self.teacher_metrics = checkpoint['teacher_metrics']