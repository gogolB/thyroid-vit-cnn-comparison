"""
PyTorch Lightning module for knowledge distillation training of Vision Transformers.
Supports distilling knowledge from CNN or ViT teachers into DeiT students.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from typing import Optional, Dict, Any, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path

from src.models.vit import get_vit_model
from src.models.vit.deit_models import DistillationLoss
from src.utils.teacher_loader import TeacherModelLoader, EnsembleTeacher
from src.utils.device import get_device


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
        
        # Create student model (must support distillation)
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
            self.progressive_schedule = config.distillation.get('progressive_schedule', {
                0: 0.9,
                100: 0.7,
                200: 0.5
            })
        
        # Log numeric configuration values only
        # String values are saved via save_hyperparameters() for W&B
        self.log_dict({
            'config/alpha': config.distillation.alpha,
            'config/temperature': config.distillation.temperature,
        }, on_epoch=False, on_step=False)
        
        # Print configuration info (will be captured by W&B logs)
        print(f"Distillation Config - Student: {config.model.name}, "
              f"Teacher: {config.distillation.teacher_model_type}")
    
    def _create_student_model(self) -> nn.Module:
        """Create the student model (DeiT with distillation support)."""
        model_config = self.config.model
        model_name = model_config.name
        
        # Ensure model supports distillation
        if 'deit' not in model_name.lower():
            print(f"Warning: Model {model_name} may not support distillation tokens")
        
        # Extract model parameters
        model_params = OmegaConf.to_container(model_config.get('params', {}), resolve=True)
        
        # Ensure distillation is enabled for DeiT
        if 'deit' in model_name.lower():
            model_params['distilled'] = True
        
        # Add pretrained flag
        if hasattr(model_config, 'pretrained'):
            model_params['pretrained'] = model_config.pretrained
        
        # Add pretrained config
        if hasattr(model_config, 'pretrained_cfg'):
            model_params['pretrained_cfg'] = OmegaConf.to_container(
                model_config.pretrained_cfg, resolve=True
            )
        
        # Create model
        model = get_vit_model(model_name, **model_params)
        
        # Log model information
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log_dict({
            'model/student_parameters': float(param_count),
            'model/student_trainable': float(trainable_count),
        }, on_epoch=False, on_step=False)
        
        print(f"Student model: {model_name}")
        print(f"Parameters: {param_count:,} (trainable: {trainable_count:,})")
        
        return model
    
    def _load_teacher_model(self) -> Tuple[nn.Module, Dict[str, float]]:
        """Load the pre-trained teacher model(s)."""
        print("\nLoading teacher model(s)...")
        
        # Use the teacher loader utility
        teacher, metrics = TeacherModelLoader.create_teacher_from_config(
            self.config,
            device=self.device if hasattr(self, 'device') else None,
            verbose=True
        )
        
        # Set to eval mode
        teacher.eval()
        
        # Log teacher information
        if isinstance(teacher, EnsembleTeacher):
            total_params = sum(
                sum(p.numel() for p in t.parameters()) 
                for t in teacher.teachers
            )
            self.log('model/teacher_ensemble_size', float(len(teacher.teachers)))
        else:
            total_params = sum(p.numel() for p in teacher.parameters())
        
        self.log('model/teacher_parameters', float(total_params))
        
        # Log teacher performance metrics
        for key, value in metrics.items():
            self.log(f'teacher/{key}', float(value))
        
        return teacher, metrics
    
    def _create_loss_functions(self) -> DistillationLoss:
        """Create the distillation loss function."""
        dist_cfg = self.config.distillation
        
        # Base classification loss
        # Check if loss config exists
        label_smoothing = 0.0
        if hasattr(self.config, 'loss') and self.config.loss is not None:
            if isinstance(self.config.loss, dict):
                label_smoothing = self.config.loss.get('label_smoothing', 0.0)
            elif hasattr(self.config.loss, 'label_smoothing'):
                label_smoothing = self.config.loss.label_smoothing
        elif hasattr(self.config, 'training') and hasattr(self.config.training, 'loss'):
            if isinstance(self.config.training.loss, dict):
                label_smoothing = self.config.training.loss.get('label_smoothing', 0.0)
            elif hasattr(self.config.training.loss, 'label_smoothing'):
                label_smoothing = self.config.training.loss.label_smoothing
                
        base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Distillation loss
        criterion = DistillationLoss(
            base_criterion=base_criterion,
            teacher_model=None,  # We'll pass teacher outputs manually
            distillation_type=dist_cfg.get('distillation_type', 'soft'),
            alpha=dist_cfg.get('alpha', 0.5),
            tau=dist_cfg.get('temperature', 4.0)
        )
        
        return criterion
    
    def _setup_metrics(self):
        """Setup metrics for tracking performance."""
        self.train_acc = Accuracy(task='binary', num_classes=2)
        self.val_acc = Accuracy(task='binary', num_classes=2)
        self.val_auc = AUROC(task='binary', num_classes=2)
        self.val_f1 = F1Score(task='binary', num_classes=2)
        self.test_acc = Accuracy(task='binary', num_classes=2)
        
        # Distillation-specific metrics
        self.teacher_agreement = Accuracy(task='multiclass', num_classes=2)
    
    def get_current_alpha(self) -> float:
        """Get current distillation weight based on progressive schedule."""
        if self.progressive_schedule is None:
            return self.config.distillation.alpha
        
        current_epoch = self.current_epoch
        
        # Find the appropriate alpha value
        epochs = sorted(self.progressive_schedule.keys())
        alpha = self.config.distillation.alpha
        
        for epoch in epochs:
            if current_epoch >= int(epoch):
                alpha = self.progressive_schedule[epoch]
        
        return alpha
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through student model."""
        return self.student(x)
    
    def get_teacher_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Get teacher predictions for distillation."""
        self.teacher.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
        return teacher_outputs
    
    def training_step(self, batch, batch_idx):
        """Training step with knowledge distillation."""
        images, labels = batch
        
        # Get teacher predictions
        teacher_outputs = self.get_teacher_outputs(images)
        
        # Student forward pass
        student_outputs = self.student(images)
        
        # Handle DeiT dual outputs
        if isinstance(student_outputs, tuple) and len(student_outputs) == 2:
            # DeiT returns (class_logits, distill_logits)
            outputs_cls, outputs_dist = student_outputs
        else:
            # Single output - use for both classification and distillation
            outputs_cls = student_outputs
            outputs_dist = student_outputs
        
        # Calculate losses
        class_loss = self.criterion.base_criterion(outputs_cls, labels)
        
        # Distillation loss
        if self.config.distillation.distillation_type == 'soft':
            # Soft distillation with KL divergence
            T = self.config.distillation.temperature
            dist_loss = F.kl_div(
                F.log_softmax(outputs_dist / T, dim=1),
                F.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean'
            ) * (T ** 2)
        else:
            # Hard distillation with teacher's predictions as labels
            teacher_labels = teacher_outputs.argmax(dim=1)
            dist_loss = self.criterion.base_criterion(outputs_dist, teacher_labels)
        
        # Combine losses with current alpha
        alpha = self.get_current_alpha()
        total_loss = (1 - alpha) * class_loss + alpha * dist_loss
        
        # Calculate metrics
        preds = outputs_cls.argmax(dim=1)
        acc = self.train_acc(preds, labels)
        
        # Teacher agreement
        teacher_preds = teacher_outputs.argmax(dim=1)
        agreement = self.teacher_agreement(preds, teacher_preds)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('class_loss', class_loss, on_step=False, on_epoch=True)
        self.log('distill_loss', dist_loss, on_step=False, on_epoch=True)
        self.log('alpha', alpha, on_step=False, on_epoch=True)
        self.log('teacher_agreement', agreement, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - evaluate student only."""
        images, labels = batch
        
        # Student predictions (eval mode)
        outputs = self.student(images)
        
        # Handle dual outputs
        if isinstance(outputs, tuple):
            # Average predictions in eval mode
            outputs = outputs[0]  # Use class token output
        
        loss = self.criterion.base_criterion(outputs, labels)
        
        # Calculate metrics
        preds = outputs.argmax(dim=1)
        acc = self.val_acc(preds, labels)
        
        # For AUC
        probs = F.softmax(outputs, dim=1)
        if probs.shape[1] == 2:
            auc = self.val_auc(probs[:, 1], labels)
        else:
            auc = self.val_auc(probs, labels)
        
        f1 = self.val_f1(preds, labels)
        
        # Teacher agreement for validation
        teacher_outputs = self.get_teacher_outputs(images)
        teacher_preds = teacher_outputs.argmax(dim=1)
        val_agreement = (preds == teacher_preds).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=False, on_epoch=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True)
        self.log('val_teacher_agreement', val_agreement, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step - evaluate student model."""
        images, labels = batch
        
        # Student predictions
        outputs = self.student(images)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Calculate test accuracy
        preds = outputs.argmax(dim=1)
        acc = self.test_acc(preds, labels)
        
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return acc
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler for distillation."""
        # Get optimizer config from training configuration
        training_cfg = self.config.training
        opt_cfg = training_cfg.optimizer
        
        # Layer-wise learning rate decay for ViT
        base_lr = opt_cfg.get('lr', opt_cfg.get('learning_rate', 0.001))
        if hasattr(self.student, 'get_parameter_groups'):
            param_groups = self.student.get_parameter_groups(
                weight_decay=opt_cfg.get('weight_decay', 0.05),
                layer_decay=opt_cfg.get('layer_decay', 0.75)
            )
            # Apply the base learning rate to each group
            for group in param_groups:
                if 'lr_scale' in group:
                    group['lr'] = base_lr * group['lr_scale']
                else:
                    group['lr'] = base_lr
        else:
            param_groups = self.student.parameters()
        
        # Create optimizer
        if opt_cfg.get('_target_', '').endswith('AdamW') or opt_cfg.get('name', '').lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_cfg.get('lr', opt_cfg.get('learning_rate', 0.001)),
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 0.05)
            )
        elif opt_cfg.get('_target_', '').endswith('Adam') or opt_cfg.get('name', '').lower() == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_cfg.get('lr', opt_cfg.get('learning_rate', 0.001)),
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.get('_target_', opt_cfg.get('name', 'unknown'))}")
        
        # Create scheduler if specified
        scheduler_cfg = training_cfg.scheduler
        
        if scheduler_cfg.get('_target_', '').endswith('CosineAnnealingLR') or scheduler_cfg.get('name', '') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get('T_max', training_cfg.num_epochs),
                eta_min=scheduler_cfg.get('eta_min', scheduler_cfg.get('min_lr', 1e-6))
            )
        elif scheduler_cfg.get('name', '') == 'cosine_warmup':
            try:
                from src.utils.schedulers import CosineWarmupScheduler
                scheduler = CosineWarmupScheduler(
                    optimizer,
                    warmup_epochs=scheduler_cfg.get('warmup_epochs', 5),
                    max_epochs=training_cfg.num_epochs,
                    min_lr=scheduler_cfg.get('min_lr', 1e-6)
                )
            except ImportError:
                # Fallback to cosine annealing without warmup
                from torch.optim.lr_scheduler import CosineAnnealingLR
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=training_cfg.num_epochs,
                    eta_min=scheduler_cfg.get('min_lr', 1e-6)
                )
        else:
            # Default to cosine annealing
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=training_cfg.num_epochs,
                eta_min=1e-6
            )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save distillation-specific information to checkpoint."""
        checkpoint['distillation_config'] = OmegaConf.to_container(
            self.config.distillation, resolve=True
        )
        checkpoint['teacher_metrics'] = self.teacher_metrics
        checkpoint['current_alpha'] = self.get_current_alpha()
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load distillation information from checkpoint."""
        if 'teacher_metrics' in checkpoint:
            self.teacher_metrics = checkpoint['teacher_metrics']