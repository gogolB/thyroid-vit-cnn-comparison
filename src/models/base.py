"""
Abstract base classes for all models.
Enforces consistent interfaces.
"""
import torch
import torch.nn as nn
# import pytorch_lightning as pl # Not using pl.LightningModule as base for these architecture classes

class ModelBase(nn.Module):
    """
    Base class for all model architectures in this project.
    Handles common initialization and expects a _build_model method.
    """
    def __init__(self, config):
        """
        Initializes the ModelBase.

        Args:
            config: A configuration object containing model parameters.
                    This config is passed down from the ModelRegistry.
        """
        super().__init__()
        self.config = config
        self.model = None  # To be populated by the _build_model method in subclasses

    def _build_model(self):
        """
        Abstract method to be implemented by subclasses to construct the actual model.
        The constructed torch.nn.Module should be assigned to self.model.
        """
        raise NotImplementedError("Subclasses must implement _build_model.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        
        Raises:
            RuntimeError: If the model has not been built yet (i.e., _build_model was not called or did not set self.model).
        """
        if self.model is None:
            raise RuntimeError(
                "The model has not been built yet. "
                "Ensure _build_model() is called in the subclass's __init__ and assigns to self.model."
            )
        return self.model(x)

class CNNBase(ModelBase):
    """
    Base class for CNN-based model architectures.
    Inherits from ModelBase.
    """
    def __init__(self, config):
        """
        Initializes the CNNBase.

        Args:
            config: A configuration object for the CNN model.
        """
        super().__init__(config)
        # Add any CNN-specific common initializations here if needed in the future.
        # For now, it primarily serves as a type for CNN models and inherits ModelBase functionality.
        # The _build_model method will be implemented by specific CNN architectures like ResNet, EfficientNet.

# Optional: Define a ViTBase if there's common logic for ViT models beyond ModelBase.
# For now, ViT models (VisionTransformer, DeiT, SwinTransformer) can inherit directly from ModelBase.
# class ViTBase(ModelBase):
#     """
#     Base class for ViT-based model architectures.
#     Inherits from ModelBase.
#     """
#     def __init__(self, config):
#         super().__init__(config)
#         # Add any ViT-specific common initializations here.