"""
ResNet implementations for thyroid tissue classification.
Refactored to use ModelRegistry and CNNBase.
"""

import sys
from pathlib import Path
from typing import Tuple

# Add project root to path for imports (Retained from original)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # src/models/cnn -> src/models -> src -> .
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torchvision.models as tv_models # Standard for ResNet implementations

from src.models.registry import ModelRegistry
from src.models.base import CNNBase

# Helper classes (Retained from original, as they are not ResNet model classes)
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

# New Unified ResNet class
@ModelRegistry.register(['resnet18', 'resnet34', 'resnet50', 'resnet101'], 'cnn')
class ResNet(CNNBase):
    """Unified ResNet implementation for all variants."""
    
    def __init__(self, config):
        super().__init__(config)
        self.variant = config.name  # resnet18, resnet34, etc.
        self._build_model()

    def _build_model(self):
        """
        Builds the ResNet model based on the variant specified in the config.
        Example: self.variant could be 'resnet18', 'resnet34', 'resnet50', 'resnet101'.
        """
        pretrained = self.config.get('pretrained', True)
        num_classes = self.config.get('num_classes', 2) # Default to 2 classes if not specified

        if self.variant == 'resnet18':
            self.model = tv_models.resnet18(pretrained=pretrained)
        elif self.variant == 'resnet34':
            self.model = tv_models.resnet34(pretrained=pretrained)
        elif self.variant == 'resnet50':
            self.model = tv_models.resnet50(pretrained=pretrained)
        elif self.variant == 'resnet101':
            self.model = tv_models.resnet101(pretrained=pretrained)
        # Add other variants like resnet152 if they were supported or needed
        # elif self.variant == 'resnet152':
        #     self.model = tv_models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet variant: {self.variant}. Supported variants are 'resnet18', 'resnet34', 'resnet50', 'resnet101'.")

        # Modify the first conv layer if in_channels is specified (e.g., for grayscale)
        # This should ideally be done BEFORE loading pretrained weights if pretrained=True and in_channels != 3
        # For simplicity here, if pretrained=True and in_channels=1, it might not work as expected
        # unless the pretrained weights are for a 1-channel model or handled carefully.
        # For this project, if pretrained=True, it's typically for 3-channel ImageNet.
        # If in_channels=1, pretrained should ideally be False or a custom 1-channel checkpoint.
        
        in_channels = self.config.get('in_channels', 3) # Default to 3 if not in config

        if in_channels != 3 and hasattr(self.model, 'conv1') and isinstance(self.model.conv1, nn.Conv2d):
            original_conv1 = self.model.conv1
            if original_conv1.in_channels == 3: # Only replace if original was 3-channel
                print(f"Adapting ResNet variant {self.variant} for {in_channels} input channels (original conv1 had 3).")
                self.model.conv1 = nn.Conv2d(
                    in_channels,
                    original_conv1.out_channels,
                    kernel_size=original_conv1.kernel_size,
                    stride=original_conv1.stride,
                    padding=original_conv1.padding,
                    bias=original_conv1.bias is not None # Preserve bias setting
                )
                # Note: If pretrained=True, the weights for conv1 are now for the wrong shape.
                # This simple replacement is fine if pretrained=False or if fine-tuning heavily.
                if pretrained and in_channels != 3:
                    print(f"Warning: ResNet variant {self.variant} loaded with pretrained=True, "
                          f"but input channels changed to {in_channels}. First layer weights are not adapted from pretrained.")
            elif original_conv1.in_channels != in_channels:
                 print(f"Warning: ResNet variant {self.variant} conv1 has {original_conv1.in_channels} channels, "
                       f"config requests {in_channels}. No change made if original conv1 not 3-channel.")


        # Modify the final fully connected layer to match num_classes
        if hasattr(self.model, 'fc') and isinstance(self.model.fc, nn.Linear):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise AttributeError(f"Could not find attribute 'fc' of type nn.Linear on model {self.variant} for num_classes adaptation.")
        
        # If quality_aware is a parameter in the config and you have a QualityEncoder
        # you might integrate it here, e.g.:
        # if self.config.get('quality_aware', False):
        #     if hasattr(self, 'quality_encoder') and self.quality_encoder is not None:
        #         # This assumes quality_encoder is initialized if needed, e.g. in __init__
        #         # And that the ResNet base class or this class handles how it's combined.
        #         # For now, this is a conceptual placeholder.
        #         print(f"Model {self.variant} is quality_aware, QualityEncoder would be integrated here.")
        #     else:
        #         print(f"Warning: Model {self.variant} configured as quality_aware, but no QualityEncoder found/initialized.")