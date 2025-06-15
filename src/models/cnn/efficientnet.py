"""
EfficientNet models for CARS thyroid classification.
Uses timm library for pre-trained EfficientNet implementations.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from src.models.registry import ModelRegistry
from src.models.base import CNNBase

@ModelRegistry.register(['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'], 'cnn')
class EfficientNet(CNNBase):
    """Unified EfficientNet implementation."""
    
    VARIANT_CONFIG = {
        'efficientnet_b0': {'width': 1.0, 'depth': 1.0, 'dropout': 0.2},
        'efficientnet_b1': {'width': 1.0, 'depth': 1.1, 'dropout': 0.2},
        'efficientnet_b2': {'width': 1.1, 'depth': 1.2, 'dropout': 0.3},
        'efficientnet_b3': {'width': 1.2, 'depth': 1.4, 'dropout': 0.3},
    }

    def __init__(self, config: DictConfig): # Changed from 'config' to 'config: DictConfig' for consistency
        super().__init__(config)
        self.variant = config.name # e.g., 'efficientnet_b0'
        # Ensure config.name is one of the keys in VARIANT_CONFIG or handle default
        if self.variant not in self.VARIANT_CONFIG:
            raise ValueError(f"Unsupported EfficientNet variant: {self.variant}. Supported variants are: {list(self.VARIANT_CONFIG.keys())}")
        self.variant_params = self.VARIANT_CONFIG[self.variant] # Use direct access after check
        self._build_model()

    def _build_model(self):
        """
        Builds the EfficientNet model based on the variant specified in the config,
        using the timm library.
        Example: self.variant could be 'efficientnet_b0', 'efficientnet_b1', etc.
        self.variant_params comes from the VARIANT_CONFIG dictionary.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2) # Default to 2 classes
        
        # Get dropout from variant_params, with a fallback default if not specified for a variant
        # The default dropout in timm for efficientnet_b0 is 0.2, b1 is 0.2, b2 is 0.3, b3 is 0.3.
        # We use the values from VARIANT_CONFIG if present.
        default_dropout = 0.2 # A general fallback
        dropout_rate = self.variant_params.get('dropout', default_dropout)
        in_chans = self.config.get('in_channels', self.config.get('channels', 1)) # Get in_channels from model or dataset config

        model_name = self.variant # e.g., 'efficientnet_b0'

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout_rate,
                in_chans=in_chans
                # timm handles classifier replacement based on num_classes.
                # Other parameters from self.variant_params (like width, depth) are typically
                # part of the model_name string itself for timm (e.g., 'tf_efficientnet_b0_ap').
                # If specific width/depth multipliers are needed beyond standard variants,
                # timm's create_model might have other ways, or a custom model definition is needed.
                # For now, we rely on standard timm variants.
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create EfficientNet model {model_name} using timm: {e}")

    # Ensure no other EfficientNet-specific model classes or factory functions remain.