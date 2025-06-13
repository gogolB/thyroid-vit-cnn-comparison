"""
Vision Transformer models for thyroid classification.
Phase 3 implementation supporting ViT, DeiT, and Swin transformers.
"""

from typing import Dict, Any, Optional
import torch.nn as nn

# Model registry for easy access
VIT_MODEL_REGISTRY = {
    # ViT models (from scratch)
    'vit_tiny': {
        'config': 'configs/model/vit/vit_tiny.yaml',
        'expected_params': 5.7e6,
        'description': 'Vision Transformer Tiny (from scratch)'
    },
    'vit_small': {
        'config': 'configs/model/vit/vit_small.yaml',
        'expected_params': 22e6,
        'description': 'Vision Transformer Small (from scratch)'
    },
    'vit_base': {
        'config': 'configs/model/vit/vit_base.yaml',
        'expected_params': 86e6,
        'description': 'Vision Transformer Base (from scratch)'
    },
    
    # DeiT models (pretrained available)
    'deit_tiny': {
        'config': 'configs/model/vit/deit_tiny.yaml',
        'expected_params': 5.7e6,
        'description': 'Data-Efficient Image Transformer Tiny'
    },
    'deit_small': {
        'config': 'configs/model/vit/deit_small.yaml',
        'expected_params': 22e6,
        'description': 'Data-Efficient Image Transformer Small'
    },
    'deit_base': {
        'config': 'configs/model/vit/deit_base.yaml',
        'expected_params': 86e6,
        'description': 'Data-Efficient Image Transformer Base'
    },
    
    # Swin models (Week 3)
    'swin_tiny': {
        'config': 'configs/model/vit/swin_tiny.yaml',
        'expected_params': 28e6,
        'description': 'Swin Transformer Tiny'
    },
    'swin_small': {
        'config': 'configs/model/vit/swin_small.yaml',
        'expected_params': 50e6,
        'description': 'Swin Transformer Small'
    },
}


def get_vit_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create ViT models.
    
    Args:
        model_name: Name of the model from VIT_MODEL_REGISTRY
        **kwargs: Additional arguments to pass to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name not found
    """
    if model_name not in VIT_MODEL_REGISTRY:
        raise ValueError(
            f"Model {model_name} not found. "
            f"Available models: {list(VIT_MODEL_REGISTRY.keys())}"
        )
    
    # Import model classes (will be implemented in subsequent files)
    if model_name.startswith('vit_'):
        from .vit_models import create_vit_model
        return create_vit_model(model_name, **kwargs)
    elif model_name.startswith('deit_'):
        from .deit_models import create_deit_model
        return create_deit_model(model_name, **kwargs)
    elif model_name.startswith('swin_'):
        from .swin_transformer import create_swin_model
        return create_swin_model(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type for {model_name}")


def list_available_models() -> Dict[str, str]:
    """
    Get a list of available ViT models with descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        name: info['description'] 
        for name, info in VIT_MODEL_REGISTRY.items()
    }


# For backward compatibility and easy imports
__all__ = [
    'VIT_MODEL_REGISTRY',
    'get_vit_model',
    'list_available_models',
]
