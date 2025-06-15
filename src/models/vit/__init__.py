import torch.nn as nn
from src.models.registry import ModelRegistry

# Attempt to import model classes. If these files or classes don't exist,
# this will raise an ImportError, which is acceptable for now if they are
# to be created/refactored later. For this task, proceed assuming they might exist.
try:
    from .vision_transformer import VisionTransformer
    ModelRegistry.register(['vit_tiny', 'vit_small', 'vit_base'], 'vit')(VisionTransformer)
except ImportError:
    print("Warning: VisionTransformer not found or could not be imported for registration.")

try:
    from .deit import DeiT
    ModelRegistry.register(['deit_tiny', 'deit_small', 'deit_base'], 'vit')(DeiT)
except ImportError:
    print("Warning: DeiT not found or could not be imported for registration.")

try:
    from .swin import SwinTransformer
    ModelRegistry.register(['swin_tiny', 'swin_small', 'swin_base'], 'vit')(SwinTransformer)
except ImportError:
    print("Warning: SwinTransformer not found or could not be imported for registration.")

# Add any other exports or __all__ definition if appropriate for an __init__.py,
# but the primary goal is the registration.
# For example, if these models were previously exported:
# __all__ = ['VisionTransformer', 'DeiT', 'SwinTransformer'] # Adjust based on actual successful imports

# If the file previously had other ways of defining/exporting models,
# those should be removed or adapted to this registry pattern.