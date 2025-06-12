"""
ResNet18 implementation for thyroid tissue classification.
Optimized for medical imaging with quality-aware features.
"""

import sys
from pathlib import Path

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # Go up to project root
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Tuple, List
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

from rich.console import Console
console = Console()


class MedicalResNet18(nn.Module):
    """
    ResNet18 adapted for medical imaging with single-channel input.
    Includes quality-aware features and improved initialization.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        hidden_dim: int = 256,
        use_attention: bool = True,
        quality_aware: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.quality_aware = quality_aware
        self.use_attention = use_attention
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Adapt first conv layer for single channel
        if in_channels != 3:
            # Save pretrained weights
            pretrained_weight = self.backbone.conv1.weight.data
            
            # Create new conv layer
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize with pretrained weights (average across RGB channels)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Remove original FC layer
        self.backbone.fc = nn.Identity()
        
        # Add attention module if requested
        if self.use_attention:
            self.attention = SpatialAttention(self.feature_dim)
        
        # Quality-aware feature extraction
        if self.quality_aware:
            self.quality_encoder = QualityEncoder(hidden_dim=64)
            classifier_input_dim = self.feature_dim + 64
        else:
            classifier_input_dim = self.feature_dim
        
        # Improved classifier head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _initialize_classifier(self):
        """Initialize classifier layers with better defaults for medical imaging."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features with optional attention."""
        # Get features before global pooling
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x, attention_map = self.attention(x)
        else:
            attention_map = None
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        
        return features, attention_map
    
    def forward(self, x: torch.Tensor, quality_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional quality scores.
        
        Args:
            x: Input images [B, C, H, W]
            quality_scores: Optional quality scores [B, 3] for (dark, contrast, artifacts)
            
        Returns:
            Dictionary with logits and optional attention maps
        """
        # Extract features
        features, attention_map = self.extract_features(x)
        
        # Add quality features if available
        if self.quality_aware and quality_scores is not None:
            quality_features = self.quality_encoder(quality_scores)
            features = torch.cat([features, quality_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        output = {'logits': logits}
        if attention_map is not None:
            output['attention'] = attention_map
        
        return output


class SpatialAttention(nn.Module):
    """Spatial attention module for highlighting important regions."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial attention."""
        attention = torch.sigmoid(self.conv(x))
        attended = x * attention
        return attended, attention


class QualityEncoder(nn.Module):
    """Encode quality scores into features."""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, quality_scores: torch.Tensor) -> torch.Tensor:
        """Encode quality scores [B, 3] -> [B, hidden_dim]"""
        return self.encoder(quality_scores)


class ResNet18Lightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for ResNet18 with quality-aware training.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = 'cosine',  # 'cosine' or 'plateau'
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2,
        hidden_dim: int = 256,
        use_attention: bool = True,
        quality_aware: bool = True,
        quality_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = MedicalResNet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            quality_aware=quality_aware
        )
        
        # Loss function with optional class weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Quality-aware loss weights
        self.quality_weights = quality_weights or {
            'high_quality': 1.0,
            'extreme_dark': 0.8,
            'low_contrast': 0.9,
            'artifacts': 0.9,
            'multiple_issues': 0.7
        }
        
        # Metrics
        self.train_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_auc = AUROC(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        
        # For tracking predictions
        self.validation_step_outputs = []
    
    def forward(self, x: torch.Tensor, quality_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return self.model(x, quality_scores)
    
    def compute_quality_scores(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute quality scores for a batch of images.
        Returns tensor of shape [B, 3] with scores for (dark, contrast, artifacts).
        """
        B = batch.shape[0]
        quality_scores = torch.zeros(B, 3, device=batch.device)
        
        # Simple heuristic - in practice, you'd use the quality preprocessor
        for i in range(B):
            img = batch[i, 0]  # Get single channel
            
            # Dark score (lower mean = higher score)
            mean_val = img.mean()
            quality_scores[i, 0] = torch.sigmoid((150 - mean_val) / 50)
            
            # Contrast score (lower std = higher score)
            std_val = img.std()
            quality_scores[i, 1] = torch.sigmoid((80 - std_val) / 30)
            
            # Artifact score (higher max/mean ratio = higher score)
            if mean_val > 0:
                ratio = img.max() / mean_val
                quality_scores[i, 2] = torch.sigmoid((ratio - 30) / 10)
        
        return quality_scores
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        
        # Compute quality scores if quality-aware
        quality_scores = self.compute_quality_scores(images) if self.hparams.quality_aware else None
        
        # Forward pass
        outputs = self(images, quality_scores)
        logits = outputs['logits']
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Apply quality-based weighting if available
        if self.hparams.quality_aware and quality_scores is not None:
            # Simple weighting based on overall quality
            quality_weight = 1.0 - 0.3 * quality_scores.mean(dim=1)  # Lower weight for lower quality
            loss = (loss * quality_weight).mean()
        
        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        images, labels = batch
        
        # Compute quality scores if quality-aware
        quality_scores = self.compute_quality_scores(images) if self.hparams.quality_aware else None
        
        # Forward pass
        outputs = self(images, quality_scores)
        logits = outputs['logits']
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Metrics
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        self.val_acc(preds, labels)
        self.val_auc(probs[:, 1] if self.hparams.num_classes == 2 else probs, labels)
        self.val_f1(preds, labels)
        
        # Logging
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val/auc', self.val_auc, on_epoch=True)
        self.log('val/f1', self.val_f1, on_epoch=True)
        
        # Store for epoch end analysis
        self.validation_step_outputs.append({
            'preds': preds,
            'labels': labels,
            'quality_scores': quality_scores
        })
    
    def on_validation_epoch_end(self) -> None:
        """Analyze performance by quality tier."""
        if not self.validation_step_outputs or not self.hparams.quality_aware:
            self.validation_step_outputs.clear()
            return
        
        # Aggregate predictions
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_quality = torch.cat([x['quality_scores'] for x in self.validation_step_outputs if x['quality_scores'] is not None])
        
        if all_quality.numel() > 0:
            # Analyze by quality tier
            quality_sum = all_quality.sum(dim=1)
            
            # High quality: no issues
            high_quality_mask = quality_sum < 0.5
            if high_quality_mask.any():
                hq_acc = (all_preds[high_quality_mask] == all_labels[high_quality_mask]).float().mean()
                self.log('val/acc_high_quality', hq_acc)
            
            # Low quality: any issues
            low_quality_mask = quality_sum >= 0.5
            if low_quality_mask.any():
                lq_acc = (all_preds[low_quality_mask] == all_labels[low_quality_mask]).float().mean()
                self.log('val/acc_low_quality', lq_acc)
        
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name and param.requires_grad:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hparams.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=self.hparams.weight_decay)
        
        # Scheduler
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        elif self.hparams.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=self.hparams.learning_rate * 0.01
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/acc',
                    'frequency': 1
                }
            }
        else:
            return optimizer


# Model factory function
def create_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> ResNet18Lightning:
    """Create ResNet18 model with Lightning wrapper."""
    return ResNet18Lightning(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    console.print("[bold cyan]Testing ResNet18 Implementation[/bold cyan]")
    
    # Create model
    model = create_resnet18(
        num_classes=2,
        pretrained=True,
        quality_aware=True,
        use_attention=True
    )
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 256, 256)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    outputs = model(dummy_input)
    console.print(f"✓ Output shape: {outputs['logits'].shape}")
    console.print(f"✓ Attention available: {'attention' in outputs}")
    
    # Test training step
    loss = model.training_step((dummy_input, dummy_labels), 0)
    console.print(f"✓ Training loss: {loss.item():.4f}")
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"\n[bold]Model Summary:[/bold]")
    console.print(f"Total parameters: {total_params:,}")
    console.print(f"Trainable parameters: {trainable_params:,}")
    console.print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")