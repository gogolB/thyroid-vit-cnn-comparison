"""
Vision Transformer model variants
"""

import torch
import torch.nn as nn
from typing import Optional, List

# Import all necessary components from vision_transformer_base
from .vision_transformer_base import (
    VisionTransformerBase, 
    Block, 
    PatchEmbed,
    Attention,
    Mlp,
    DropPath
)


class VisionTransformer(VisionTransformerBase):
    """
    Vision Transformer implementation
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 2,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        representation_size: Optional[int] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer = None,  # Will use default from base
        norm_layer = None,   # Will use default from base
        act_layer = None,    # Will use default from base
        **kwargs
    ):
        # Set defaults before passing to parent
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        if act_layer is None:
            act_layer = nn.GELU
            
        # Call parent init - this creates patch_embed, cls_token, pos_embed, etc.
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs
        )
        
        # After parent init, we need to create the blocks
        # The base class expects self.blocks to be set by subclasses
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Create transformer blocks
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        # Representation layer (optional)
        if representation_size and representation_size > 0:
            self.representation_size = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.representation_size = None
            self.pre_logits = nn.Identity()


class ViTTiny(VisionTransformer):
    """Vision Transformer Tiny variant (~5.7M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            **kwargs
        )


class ViTSmall(VisionTransformer):
    """Vision Transformer Small variant (~22M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            **kwargs
        )


class ViTBase(VisionTransformer):
    """Vision Transformer Base variant (~86M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            **kwargs
        )


# Factory functions for easy instantiation
def create_vit_tiny(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    drop_path_rate: float = 0.1,
    **kwargs
) -> ViTTiny:
    """
    Create Vision Transformer Tiny model
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_chans: Number of input channels
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate
        **kwargs: Additional arguments
        
    Returns:
        ViTTiny model instance
    """
    return ViTTiny(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        **kwargs
    )


def create_vit_small(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    drop_path_rate: float = 0.1,
    **kwargs
) -> ViTSmall:
    """
    Create Vision Transformer Small model
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_chans: Number of input channels
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate
        **kwargs: Additional arguments
        
    Returns:
        ViTSmall model instance
    """
    return ViTSmall(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        **kwargs
    )


def create_vit_base(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    drop_path_rate: float = 0.1,
    **kwargs
) -> ViTBase:
    """
    Create Vision Transformer Base model
    
    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_chans: Number of input channels
        num_classes: Number of output classes
        drop_path_rate: Stochastic depth rate
        **kwargs: Additional arguments
        
    Returns:
        ViTBase model instance
    """
    return ViTBase(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        **kwargs
    )


def create_vit_model(model_name: str, **kwargs) -> VisionTransformer:
    """
    Create Vision Transformer model by name
    
    Args:
        model_name: Name of the model ('vit_tiny', 'vit_small', 'vit_base')
        **kwargs: Model arguments
        
    Returns:
        VisionTransformer model instance
    """
    model_map = {
        'vit_tiny': create_vit_tiny,
        'vit_small': create_vit_small,
        'vit_base': create_vit_base,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown ViT model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)


# Model parameter counts for reference
VIT_PARAMS = {
    'vit_tiny': '5.7M',
    'vit_small': '22M', 
    'vit_base': '86M',
}


if __name__ == "__main__":
    # Test model creation and parameter counts
    import numpy as np
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    # Test each variant
    for name, expected in VIT_PARAMS.items():
        model = create_vit_model(name)
        params = count_parameters(model)
        print(f"{name}: {params:,} parameters (expected ~{expected})")
        
        # Test forward pass
        x = torch.randn(2, 1, 256, 256)
        output = model(x)
        assert output.shape == (2, 2), f"Unexpected output shape: {output.shape}"
        print(f"  ✓ Forward pass successful")