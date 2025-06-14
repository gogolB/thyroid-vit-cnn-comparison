# Update src/models/vit/__init__.py

from .vision_transformer_base import VisionTransformerBase
from .vit_models import create_vit_tiny, create_vit_small, create_vit_base
from .deit_models import create_deit_tiny, create_deit_small, create_deit_base
from .swin_transformer import (
    create_swin_tiny, create_swin_small, create_swin_base, 
    create_swin_large, create_swin_medical
)
import torch.nn as nn

# Model registry
VIT_MODEL_REGISTRY = {
    # Standard ViT models
    'vit_tiny': create_vit_tiny,
    'vit_small': create_vit_small,
    'vit_base': create_vit_base,
    
    # DeiT models
    'deit_tiny': create_deit_tiny,
    'deit_small': create_deit_small,
    'deit_base': create_deit_base,
    
    # Swin models
    'swin_tiny': create_swin_tiny,
    'swin_small': create_swin_small,
    'swin_base': create_swin_base,
    'swin_large': create_swin_large,
    'swin_medical': create_swin_medical,
}

def get_vit_model(model_name: str, **kwargs) -> nn.Module:
    """
    Get a Vision Transformer model by name.
    
    Args:
        model_name: Name of the model
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Vision Transformer model instance
    """
    if model_name not in VIT_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(VIT_MODEL_REGISTRY.keys())}"
        )
    
    return VIT_MODEL_REGISTRY[model_name](**kwargs)

# Export the main function and model names
__all__ = [
    'VisionTransformerBase',
    'get_vit_model',
    'VIT_MODEL_REGISTRY',
    # Individual model creators
    'create_vit_tiny', 'create_vit_small', 'create_vit_base',
    'create_deit_tiny', 'create_deit_small', 'create_deit_base',
    'create_swin_tiny', 'create_swin_small', 'create_swin_base',
    'create_swin_large', 'create_swin_medical',
]