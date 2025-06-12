"""
DenseNet models for CARS thyroid classification.
Uses torchvision's DenseNet with modifications for single-channel input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
from omegaconf import DictConfig


class ChannelAttention(nn.Module):
    """Channel attention module for DenseNet."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling path
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Max pooling path
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class DenseNet121(nn.Module):
    """
    DenseNet121 model adapted for CARS thyroid classification.
    Includes modifications for single-channel input and medical imaging.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
        quality_aware: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.quality_aware = quality_aware
        
        # Load pretrained DenseNet121
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Get the number of features before classifier
        num_features = self.backbone.classifier.in_features
        
        # Modify first convolution for single channel input
        if in_channels != 3:
            # Save the pretrained weights
            pretrained_conv1 = self.backbone.features.conv0.weight.data
            
            # Create new conv layer
            self.backbone.features.conv0 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # Initialize with averaged pretrained weights
            if pretrained:
                self.backbone.features.conv0.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add channel attention if requested
        if self.use_attention:
            self.channel_attention = ChannelAttention(num_features)
        
        # Quality-aware feature encoding
        if self.quality_aware:
            self.quality_encoder = nn.Sequential(
                nn.Linear(3, 32),  # 3 quality metrics
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
            classifier_input_dim = num_features + 64
        else:
            classifier_input_dim = num_features
        
        # Enhanced classifier head for medical imaging
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),  # Slightly less dropout in second layer
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier with better defaults
        self._initialize_classifier()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
    
    def _initialize_classifier(self):
        """Initialize classifier layers with Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def extract_quality_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quality-related features from input images."""
        batch_size = x.size(0)
        
        # Calculate quality metrics per image
        quality_features = []
        
        for i in range(batch_size):
            img = x[i, 0]  # Get single channel
            
            # Mean intensity (normalized)
            mean_intensity = img.mean().item()
            
            # Standard deviation (contrast measure)
            std_intensity = img.std().item()
            
            # Entropy (information content)
            # Simplified entropy calculation
            hist = torch.histc(img, bins=256, min=0, max=1)
            hist = hist / hist.sum()
            entropy = -(hist * torch.log2(hist + 1e-10)).sum().item()
            
            quality_features.append([mean_intensity, std_intensity, entropy])
        
        return torch.tensor(quality_features, device=x.device, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional quality-aware processing."""
        
        # Extract features from backbone
        features = self.backbone.features(x)
        
        # Apply ReLU and adaptive average pooling
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Apply channel attention if enabled
        if self.use_attention:
            # Reshape for attention module
            b, c = features.shape
            features_2d = features.view(b, c, 1, 1)
            features_2d = self.channel_attention(features_2d)
            features = features_2d.view(b, c)
        
        # Add quality-aware features if enabled
        if self.quality_aware:
            quality_feats = self.extract_quality_features(x)
            quality_encoded = self.quality_encoder(quality_feats)
            features = torch.cat([features, quality_encoded], dim=1)
        
        # Final classification
        output = self.classifier(features)
        
        return output
    
    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before classification (useful for visualization)."""
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        if self.use_attention:
            b, c = features.shape
            features_2d = features.view(b, c, 1, 1)
            features_2d = self.channel_attention(features_2d)
            features = features_2d.view(b, c)
        
        return features


def create_densenet_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create DenseNet models based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        DenseNet model instance
    """
    model_map = {
        'densenet121': DenseNet121,
        # Future: can add DenseNet169, DenseNet201, etc.
    }
    
    model_name = cfg.model.name
    if model_name not in model_map:
        raise ValueError(f"Unknown DenseNet model: {model_name}")
    
    # Extract model parameters from config
    model_params = {
        'num_classes': cfg.dataset.num_classes,
        'pretrained': cfg.model.pretrained,
        'dropout_rate': cfg.model.dropout_rate,
        'in_channels': cfg.model.get('in_channels', 1),
        'use_attention': cfg.model.get('use_attention', True),
        'quality_aware': cfg.model.get('quality_aware', True),
        'freeze_backbone': cfg.model.get('freeze_backbone', False)
    }
    
    # Create and return model
    return model_map[model_name](**model_params)


# Convenience function for direct model creation
def densenet121(num_classes: int = 2, pretrained: bool = True, **kwargs) -> DenseNet121:
    """Create DenseNet121 model."""
    return DenseNet121(num_classes=num_classes, pretrained=pretrained, **kwargs)