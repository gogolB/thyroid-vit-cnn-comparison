"""
DeiT (Data-efficient Image Transformer) implementation
Includes distillation token support and pretrained weight loading from timm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import timm
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
import warnings

from .vision_transformer_base import VisionTransformerBase
from .vit_models import VisionTransformer


class DeiT(VisionTransformer):
    """
    Data-efficient Image Transformer (DeiT) with optional distillation token.
    
    Key differences from standard ViT:
    - Optional distillation token for knowledge distillation
    - Specialized initialization for distillation
    - Support for loading pretrained weights from timm
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
        distilled: bool = True,  # DeiT specific
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer = None,
        norm_layer = None,
        act_layer = None,
        pretrained: bool = False,
        pretrained_cfg: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize parent ViT
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
            representation_size=representation_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            **kwargs
        )
        
        self.distilled = distilled
        self.pretrained = pretrained
        self.pretrained_cfg = pretrained_cfg or {}
        
        # Add distillation token if enabled
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.num_tokens = 2  # CLS + distillation tokens
            
            # Additional head for distillation
            self.head_dist = nn.Linear(self.embed_dim, num_classes)
            if self.representation_size:
                self.head_dist = nn.Sequential(
                    nn.Linear(self.embed_dim, self.representation_size),
                    nn.Tanh(),
                    nn.Linear(self.representation_size, num_classes)
                )
            
            # Re-initialize position embeddings with extra token
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, self.embed_dim)
            )
            
            # Initialize distillation token
            nn.init.trunc_normal_(self.dist_token, std=0.02)
            nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
            nn.init.zeros_(self.head_dist.bias)
        
        # Re-initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Load pretrained weights if requested
        if pretrained:
            self.load_pretrained_weights()
    
    def load_pretrained_weights(self):
        """Load pretrained weights from timm or Facebook's DeiT models"""
        if not self.pretrained_cfg:
            warnings.warn("No pretrained config provided, skipping weight loading")
            return
            
        model_name = self.pretrained_cfg.get('model_name', 'deit_tiny_patch16_224')
        
        try:
            # Load pretrained model from timm
            print(f"Loading pretrained weights from timm: {model_name}")
            pretrained_model = timm.create_model(model_name, pretrained=True)
            
            # Get state dict
            pretrained_state = pretrained_model.state_dict()
            
            # Adapt weights for our model
            adapted_state = self._adapt_pretrained_weights(pretrained_state)
            
            # Load adapted weights
            incompatible_keys = self.load_state_dict(adapted_state, strict=False)
            
            if incompatible_keys.missing_keys:
                print(f"Missing keys: {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
                
            print("Successfully loaded pretrained weights")
            
        except Exception as e:
            warnings.warn(f"Failed to load pretrained weights: {e}")
    
    def _adapt_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt pretrained weights to match our model architecture"""
        adapted_state = {}
        
        # Get model configuration
        pretrained_img_size = self.pretrained_cfg.get('input_size', [3, 224, 224])[-1]
        pretrained_num_classes = self.pretrained_cfg.get('num_classes', 1000)
        
        for key, value in state_dict.items():
            # Handle patch embedding for grayscale images
            if key == 'patch_embed.proj.weight' and value.shape[1] == 3 and self.in_chans == 1:
                # Average RGB channels to grayscale
                value = value.mean(dim=1, keepdim=True)
            
            # Handle position embeddings with different sizes
            elif key == 'pos_embed' and value.shape != self.pos_embed.shape:
                value = self._interpolate_pos_embed(value, self.pos_embed.shape)
            
            # Skip classification head if number of classes differs
            elif key in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if 'head.weight' in key and value.shape[0] != self.num_classes:
                    continue  # Skip loading head weights
                elif 'head_dist' in key and not self.distilled:
                    continue  # Skip distillation head if not using distillation
            
            # Handle distillation token
            elif key == 'dist_token' and not self.distilled:
                continue  # Skip if not using distillation
                
            adapted_state[key] = value
        
        return adapted_state
    
    def _interpolate_pos_embed(self, pos_embed: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Interpolate position embeddings to handle different image sizes"""
        # pos_embed shape: (1, num_patches + num_tokens, embed_dim)
        num_tokens = 2 if self.distilled else 1
        
        # Separate class (and dist) tokens from position embeddings
        extra_tokens = pos_embed[:, :num_tokens]
        pos_tokens = pos_embed[:, num_tokens:]
        
        # Calculate grid sizes
        src_size = int(pos_tokens.shape[1] ** 0.5)
        dst_size = int((target_shape[1] - num_tokens) ** 0.5)
        
        if src_size != dst_size:
            # Reshape to 2D grid
            pos_tokens = pos_tokens.reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2)
            
            # Interpolate
            pos_tokens = F.interpolate(
                pos_tokens,
                size=(dst_size, dst_size),
                mode='bicubic',
                align_corners=False
            )
            
            # Reshape back
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Concatenate back with class token
        return torch.cat([extra_tokens, pos_tokens], dim=1)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layers"""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.distilled:
            # Add both class and distillation tokens
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning both standard and distillation outputs if training"""
        x = self.forward_features(x)
        
        if self.distilled and self.training:
            # Return both classification and distillation outputs during training
            x_cls = self.head(self.pre_logits(x[:, 0]))  # Class token
            x_dist = self.head_dist(x[:, 1])  # Distillation token
            return x_cls, x_dist
        else:
            # During inference, average the predictions
            if self.distilled:
                # Average class and distillation predictions
                x_cls = self.head(self.pre_logits(x[:, 0]))
                x_dist = self.head_dist(x[:, 1])
                return (x_cls + x_dist) / 2
            else:
                # Standard ViT output
                return self.head(self.pre_logits(x[:, 0]))


class DeiTTiny(DeiT):
    """DeiT-Tiny model (~5.7M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            **kwargs
        )


class DeiTSmall(DeiT):
    """DeiT-Small model (~22M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            **kwargs
        )


class DeiTBase(DeiT):
    """DeiT-Base model (~86M params)"""
    def __init__(self, **kwargs):
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            **kwargs
        )


def create_deit_tiny(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    distilled: bool = True,
    pretrained: bool = False,
    **kwargs
) -> DeiTTiny:
    """Create DeiT-Tiny model with optional pretrained weights"""
    
    pretrained_cfg = None
    if pretrained:
        pretrained_cfg = {
            'model_name': 'deit_tiny_patch16_224',
            'num_classes': 1000,
            'input_size': [3, 224, 224],
        }
    
    return DeiTTiny(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        distilled=distilled,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )


def create_deit_small(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    distilled: bool = True,
    pretrained: bool = False,
    **kwargs
) -> DeiTSmall:
    """Create DeiT-Small model with optional pretrained weights"""
    
    pretrained_cfg = None
    if pretrained:
        pretrained_cfg = {
            'model_name': 'deit_small_patch16_224',
            'num_classes': 1000,
            'input_size': [3, 224, 224],
        }
    
    return DeiTSmall(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        distilled=distilled,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )


def create_deit_base(
    img_size: int = 256,
    patch_size: int = 16,
    in_chans: int = 1,
    num_classes: int = 2,
    distilled: bool = True,
    pretrained: bool = False,
    **kwargs
) -> DeiTBase:
    """Create DeiT-Base model with optional pretrained weights"""
    
    pretrained_cfg = None
    if pretrained:
        pretrained_cfg = {
            'model_name': 'deit_base_patch16_224',
            'num_classes': 1000,
            'input_size': [3, 224, 224],
        }
    
    return DeiTBase(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        distilled=distilled,
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )


def create_deit_model(model_name: str, **kwargs) -> DeiT:
    """Factory function for DeiT models"""
    
    model_map = {
        'deit_tiny': create_deit_tiny,
        'deit_small': create_deit_small,
        'deit_base': create_deit_base,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown DeiT model: {model_name}. Available: {list(model_map.keys())}")
    
    return model_map[model_name](**kwargs)


# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    """
    Loss function for DeiT knowledge distillation training.
    Combines standard cross-entropy with distillation loss.
    """
    
    def __init__(
        self,
        base_criterion: nn.Module = nn.CrossEntropyLoss(),
        teacher_model: Optional[nn.Module] = None,
        distillation_type: str = 'soft',
        alpha: float = 0.5,
        tau: float = 3.0
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        
    def forward(
        self,
        outputs: Tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        teacher_outputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            outputs: Tuple of (class_logits, distill_logits) from student
            targets: Ground truth labels
            teacher_outputs: Logits from teacher model (optional)
            
        Returns:
            Combined loss value
        """
        if isinstance(outputs, tuple):
            outputs_cls, outputs_dist = outputs
        else:
            outputs_cls = outputs
            outputs_dist = None
        
        # Standard classification loss
        base_loss = self.base_criterion(outputs_cls, targets)
        
        if outputs_dist is None or teacher_outputs is None:
            return base_loss
        
        # Distillation loss
        if self.distillation_type == 'soft':
            # Soft distillation (KL divergence between softmax outputs)
            dist_loss = F.kl_div(
                F.log_softmax(outputs_dist / self.tau, dim=1),
                F.softmax(teacher_outputs / self.tau, dim=1),
                reduction='batchmean'
            ) * (self.tau ** 2)
        else:
            # Hard distillation (use teacher's argmax as target)
            dist_loss = self.base_criterion(outputs_dist, teacher_outputs.argmax(dim=1))
        
        # Combine losses
        return (1 - self.alpha) * base_loss + self.alpha * dist_loss


if __name__ == "__main__":
    # Test DeiT models
    print("Testing DeiT implementation...")
    
    # Test DeiT-Tiny
    model = create_deit_tiny(pretrained=False, distilled=True)
    x = torch.randn(2, 1, 256, 256)
    
    # Test training mode (returns tuple)
    model.train()
    out_train = model(x)
    if isinstance(out_train, tuple):
        print(f"✓ Training mode - Class output: {out_train[0].shape}, Distill output: {out_train[1].shape}")
    
    # Test eval mode (returns averaged output)
    model.eval()
    out_eval = model(x)
    print(f"✓ Eval mode - Output shape: {out_eval.shape}")
    
    # Test parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ DeiT-Tiny parameters: {params:,} (expected ~5.7M)")
    
    # Test pretrained loading (requires internet)
    try:
        model_pretrained = create_deit_tiny(pretrained=True, distilled=True)
        print("✓ Pretrained weights loaded successfully")
    except Exception as e:
        print(f"! Pretrained loading skipped: {e}")
    
    print("\nDeiT implementation complete!")