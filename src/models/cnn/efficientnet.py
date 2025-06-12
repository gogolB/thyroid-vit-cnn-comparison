"""
EfficientNet models for CARS thyroid classification.
Uses timm library for pre-trained EfficientNet implementations.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any
from omegaconf import DictConfig


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 model for binary classification."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        in_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Create EfficientNet-B0 model using timm
        self.model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_connect_rate
        )
        
        # Store model info
        self.num_features = self.model.num_features
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_classifier(self):
        """Get the classifier layer."""
        return self.model.classifier
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


class EfficientNetB1(nn.Module):
    """EfficientNet-B1 model for binary classification."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        in_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Create EfficientNet-B1 model using timm
        self.model = timm.create_model(
            'efficientnet_b1',
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_connect_rate
        )
        
        self.num_features = self.model.num_features
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_classifier(self):
        """Get the classifier layer."""
        return self.model.classifier
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


class EfficientNetB2(nn.Module):
    """EfficientNet-B2 model for binary classification."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        drop_connect_rate: float = 0.2,
        in_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Create EfficientNet-B2 model using timm
        self.model = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_connect_rate
        )
        
        self.num_features = self.model.num_features
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_classifier(self):
        """Get the classifier layer."""
        return self.model.classifier
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


class EfficientNetB3(nn.Module):
    """EfficientNet-B3 model for binary classification."""
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        drop_connect_rate: float = 0.3,
        in_channels: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Create EfficientNet-B3 model using timm
        self.model = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_connect_rate
        )
        
        self.num_features = self.model.num_features
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_classifier(self):
        """Get the classifier layer."""
        return self.model.classifier
    
    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def create_efficientnet_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create EfficientNet models based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        EfficientNet model instance
    """
    model_map = {
        'efficientnet_b0': EfficientNetB0,
        'efficientnet_b1': EfficientNetB1,
        'efficientnet_b2': EfficientNetB2,
        'efficientnet_b3': EfficientNetB3,
    }
    
    model_name = cfg.model.name
    if model_name not in model_map:
        raise ValueError(f"Unknown EfficientNet model: {model_name}")
    
    # Extract model parameters from config
    model_params = {
        'num_classes': cfg.dataset.num_classes,
        'pretrained': cfg.model.pretrained,
        'dropout_rate': cfg.model.dropout_rate,
        'in_channels': 1,  # CARS images are single channel
    }
    
    # Add drop_connect_rate if specified in config
    if hasattr(cfg.model, 'drop_connect_rate'):
        model_params['drop_connect_rate'] = cfg.model.drop_connect_rate
    
    # Create and return model
    model = model_map[model_name](**model_params)
    
    # Freeze backbone if specified
    if cfg.model.freeze_backbone:
        model.freeze_backbone()
    
    return model


# Convenience functions for direct model creation
def efficientnet_b0(num_classes: int = 2, pretrained: bool = True, **kwargs) -> EfficientNetB0:
    """Create EfficientNet-B0 model."""
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained, **kwargs)


def efficientnet_b1(num_classes: int = 2, pretrained: bool = True, **kwargs) -> EfficientNetB1:
    """Create EfficientNet-B1 model."""
    return EfficientNetB1(num_classes=num_classes, pretrained=pretrained, **kwargs)


def efficientnet_b2(num_classes: int = 2, pretrained: bool = True, **kwargs) -> EfficientNetB2:
    """Create EfficientNet-B2 model."""
    return EfficientNetB2(num_classes=num_classes, pretrained=pretrained, **kwargs)


def efficientnet_b3(num_classes: int = 2, pretrained: bool = True, **kwargs) -> EfficientNetB3:
    """Create EfficientNet-B3 model."""
    return EfficientNetB3(num_classes=num_classes, pretrained=pretrained, **kwargs)