"""
DenseNet models for CARS thyroid classification.
Refactored to use ModelRegistry and CNNBase.
"""

import torch.nn as nn
import timm

from src.models.registry import ModelRegistry
from src.models.base import CNNBase
from omegaconf import DictConfig # Keep if config is used, which it is in __init__

@ModelRegistry.register(['densenet121', 'densenet161', 'densenet169', 'densenet201'], 'cnn')
class DenseNet(CNNBase):
    """Unified DenseNet implementation for all variants."""
    
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.variant = config.name  # e.g., 'densenet121'
        # The original __init__ took DictConfig, so keeping it for consistency
        # self.config is available from CNNBase if needed
        self._build_model()

    def _build_model(self):
        """
        Builds the DenseNet model based on the variant specified in the config,
        using the timm library.
        Example: self.variant could be 'densenet121', 'densenet169', etc.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2) # Default to 2 classes
        in_chans = self.config.get('in_channels', self.config.get('channels', 1)) # Get in_channels from model or dataset config
        
        model_name = self.variant # e.g., 'densenet121'

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_chans
                # timm handles classifier replacement based on num_classes.
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create DenseNet model {model_name} using timm: {e}")

    def forward(self, x):
        # Placeholder for forward pass, actual implementation depends on _build_model
        if hasattr(self, 'model') and self.model is not None:
            return self.model(x)
        raise NotImplementedError("DenseNet.forward() not implemented yet or model not built.")

# Ensure no other DenseNet-specific model classes or factory functions remain.
# The old DenseNet121, ChannelAttention, create_densenet_model, and densenet121 functions
# have been removed.