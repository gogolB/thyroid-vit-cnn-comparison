"""
Base CNN class for thyroid classification models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from rich.console import Console

console = Console()


class BaseCNN(nn.Module, ABC):
    """Abstract base class for CNN models."""
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
        hidden_dim: int = 512,
        pool_type: str = 'avg',
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        
        # Will be set by subclasses
        self.backbone = None
        self.classifier = None
        self.features_dim = None
    
    @abstractmethod
    def _create_backbone(self) -> nn.Module:
        """Create the backbone network."""
        pass
    
    @abstractmethod
    def _get_features_dim(self) -> int:
        """Get the dimension of features from backbone."""
        pass
    
    def _create_classifier(self) -> nn.Module:
        """Create the classifier head."""
        layers = []
        
        # Global pooling
        if self.pool_type == 'avg':
            layers.append(nn.AdaptiveAvgPool2d(1))
        else:
            layers.append(nn.AdaptiveMaxPool2d(1))
        
        layers.append(nn.Flatten())
        
        # Optional hidden layer
        if self.hidden_dim > 0:
            layers.extend([
                nn.Linear(self.features_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.num_classes)
            ])
        else:
            layers.append(nn.Linear(self.features_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        console.print("[yellow]Backbone frozen[/yellow]")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        console.print("[green]Backbone unfrozen[/green]")
    
    def get_param_groups(self, lr: float = 0.001, backbone_lr_scale: float = 0.1) -> list:
        """
        Get parameter groups with different learning rates.
        
        Args:
            lr: Base learning rate
            backbone_lr_scale: Scale factor for backbone learning rate
            
        Returns:
            List of parameter groups
        """
        backbone_params = list(self.backbone.parameters())
        classifier_params = list(self.classifier.parameters())
        
        param_groups = [
            {'params': backbone_params, 'lr': lr * backbone_lr_scale},
            {'params': classifier_params, 'lr': lr}
        ]
        
        return param_groups
    
    @property
    def num_parameters(self) -> Dict[str, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        classifier_total = sum(p.numel() for p in self.classifier.parameters())
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'backbone_total': backbone_total,
            'backbone_trainable': backbone_trainable,
            'classifier_total': classifier_total,
            'classifier_trainable': classifier_trainable
        }
    
    def print_summary(self):
        """Print model summary."""
        params = self.num_parameters
        
        console.print("\n[bold cyan]Model Summary[/bold cyan]")
        console.print(f"Architecture: {self.__class__.__name__}")
        console.print(f"Input channels: {self.in_channels}")
        console.print(f"Number of classes: {self.num_classes}")
        console.print(f"Dropout rate: {self.dropout_rate}")
        console.print(f"Hidden dimension: {self.hidden_dim}")
        console.print(f"Pool type: {self.pool_type}")
        
        console.print("\n[bold]Parameters:[/bold]")
        console.print(f"Total: {params['total']:,}")
        console.print(f"Trainable: {params['trainable']:,}")
        console.print(f"  Backbone: {params['backbone_trainable']:,} / {params['backbone_total']:,}")
        console.print(f"  Classifier: {params['classifier_trainable']:,} / {params['classifier_total']:,}")