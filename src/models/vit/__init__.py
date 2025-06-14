
"""
Vision Transformer models package.
"""

# Import ViT and DeiT models
try:
    from .vit_models import create_vit_tiny, create_vit_small, create_vit_base
    VIT_MODELS_AVAILABLE = True
except ImportError:
    VIT_MODELS_AVAILABLE = False

try:
    from .deit_models import create_deit_tiny, create_deit_small, create_deit_base
    DEIT_MODELS_AVAILABLE = True
except ImportError:
    DEIT_MODELS_AVAILABLE = False

# Import Swin models
try:
    from .swin_transformer import (
        create_swin_tiny, create_swin_small, create_swin_base,
        create_swin_large, create_swin_medical
    )
    SWIN_MODELS_AVAILABLE = True
except ImportError:
    SWIN_MODELS_AVAILABLE = False

# Model registry
VIT_MODEL_REGISTRY = {}

if VIT_MODELS_AVAILABLE:
    VIT_MODEL_REGISTRY.update({
        'vit_tiny': create_vit_tiny,
        'vit_small': create_vit_small,
        'vit_base': create_vit_base,
    })

if DEIT_MODELS_AVAILABLE:
    VIT_MODEL_REGISTRY.update({
        'deit_tiny': create_deit_tiny,
        'deit_small': create_deit_small,
        'deit_base': create_deit_base,
    })

if SWIN_MODELS_AVAILABLE:
    VIT_MODEL_REGISTRY.update({
        'swin_tiny': create_swin_tiny,
        'swin_small': create_swin_small,
        'swin_base': create_swin_base,
        'swin_large': create_swin_large,
        'swin_medical': create_swin_medical,
    })


def get_vit_model(model_name: str, **kwargs):
    """Get a Vision Transformer model by name.
    
    Args:
        model_name: Name of the model (e.g., 'vit_tiny', 'deit_small', 'swin_base')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model name is not recognized
    """
    if model_name not in VIT_MODEL_REGISTRY:
        available = list(VIT_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown Vision Transformer model: {model_name}. "
            f"Available models: {', '.join(available)}"
        )
    
    return VIT_MODEL_REGISTRY[model_name](**kwargs)


__all__ = [
    'get_vit_model',
    'VIT_MODEL_REGISTRY',
    'VIT_MODELS_AVAILABLE',
    'DEIT_MODELS_AVAILABLE', 
    'SWIN_MODELS_AVAILABLE',
]