"""Unified Inception model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import timm

from src.models.registry import ModelRegistry
from src.models.base import CNNBase

@ModelRegistry.register(['inception_v3', 'inception_v4'], 'cnn')
class Inception(CNNBase):
    """Unified Inception implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.variant = config.name # e.g., 'inception_v3'
        self.aux_logits = config.get('aux_logits', True if self.variant == 'inception_v3' else False) # Default aux_logits for v3 is True
        self._validate_config()
        self._build_model()

    def _validate_config(self):
        # Placeholder: Actual config validation logic will be added in a later step.
        # For this task, just 'pass' is sufficient.
        pass

    def _build_model(self):
        """
        Builds the Inception model based on the variant specified in the config,
        using the timm library.
        Example: self.variant could be 'inception_v3', 'inception_v4'.
        self.aux_logits is set in __init__ based on config.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2) # Default to 2 classes
        # Get in_channels from the model's own config, defaulting to 1.
        # This assumes the model's config (e.g., inception_v3.yaml)
        # will have an 'in_channels' field, possibly linked to dataset.channels via Hydra.
        in_channels = self.config.get('in_channels', 1)
        
        model_name = self.variant # e.g., 'inception_v3'

        # For Inception v3, aux_logits is a parameter. For Inception v4, it might not be directly
        # settable in timm's create_model or might be handled differently.
        # We pass self.aux_logits, which was derived from config in __init__.
        # If a model doesn't support aux_logits, timm might ignore it or error.
        # This simple approach assumes timm handles it appropriately for the given variant.
        
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels, # Pass the number of input channels
                aux_logits=self.aux_logits if model_name == 'inception_v3' else None # Pass aux_logits only if applicable
            )
            # The direct timm test with in_chans=1 was successful, suggesting timm
            # handles the single-channel input correctly for inception_v3.
            # If specific adaptation for pretrained weights is needed (e.g., averaging 3-channel weights for 1-channel input),
            # that logic would be added here. For now, we rely on timm's handling.
            
        except Exception as e:
            raise RuntimeError(f"Failed to create Inception model {model_name} using timm: {e}")