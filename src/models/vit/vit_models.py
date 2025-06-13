"""
Vision Transformer model variants
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .vision_transformer_base import VisionTransformerBase, Block


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
        embed_layer = PatchEmbed,
        norm_layer = nn.LayerNorm,
        act_layer = nn.GELU,
        **kwargs
    ):
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
            **kwargs
        )
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
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
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        if self.class_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Pooling
        if self.pool_type == 'cls' and self.class_token:
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)
            
        x = self.pre_logits(x)
        return x


def create_vit_tiny(**kwargs) -> VisionTransformer:
    """Create ViT-Tiny model"""
    model_kwargs = dict(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        **kwargs
    )
    return VisionTransformer(**model_kwargs)


def create_vit_small(**kwargs) -> VisionTransformer:
    """Create ViT-Small model"""
    model_kwargs = dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        **kwargs
    )
    return VisionTransformer(**model_kwargs)


def create_vit_base(**kwargs) -> VisionTransformer:
    """Create ViT-Base model"""
    model_kwargs = dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        **kwargs
    )
    return VisionTransformer(**model_kwargs)


def create_vit_model(model_name: str, **kwargs) -> VisionTransformer:
    """Factory function for ViT models"""
    model_map = {
        'vit_tiny': create_vit_tiny,
        'vit_small': create_vit_small,
        'vit_base': create_vit_base,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
        
    return model_map[model_name](**kwargs)

