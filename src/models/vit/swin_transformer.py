"""
Swin Transformer implementation for thyroid microscopy classification.
Hierarchical vision transformer with shifted windows and medical imaging adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import warnings

# Import base class from existing implementation
from src.models.vit.vision_transformer_base import VisionTransformerBase


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: Input features of shape (B, H, W, C)
        window_size: Window size
        
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
        
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MedicalWindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with medical imaging adaptations.
    
    Features:
    - Relative position bias
    - Contrast-adaptive scaling for low-contrast regions
    - Quality-guided attention weighting
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        contrast_adaptive: bool = True,
        quality_guided: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Medical adaptations
        self.contrast_adaptive = contrast_adaptive
        self.quality_guided = quality_guided
        
        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Medical adaptations
        if self.contrast_adaptive:
            self.contrast_scale = nn.Parameter(torch.ones(num_heads))
            
        if self.quality_guided:
            self.quality_gate = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (num_windows*B, N, C)
            mask: (0/-inf) mask of shape (num_windows, Wh*Ww, Wh*Ww) or None
            quality_scores: Optional quality scores for patches
            
        Returns:
            Output features of shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Contrast-adaptive scaling
        if self.contrast_adaptive and hasattr(self, 'contrast_scale'):
            attn = attn * self.contrast_scale.view(1, -1, 1, 1)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Quality-guided gating
        if self.quality_guided and quality_scores is not None:
            quality_gate = self.quality_gate(x)
            x = x * quality_gate
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted window attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        medical_adaptations: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be between 0 and window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = MedicalWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            contrast_adaptive=medical_adaptations,
            quality_guided=medical_adaptations
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        H: int, 
        W: int,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._generate_attention_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask, quality_scores=quality_scores)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    
    def _generate_attention_mask(self, Hp: int, Wp: int, device: torch.device) -> torch.Tensor:
        """Generate attention mask for shifted window attention."""
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                   slice(-self.window_size, -self.shift_size),
                   slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
                
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask


class PatchMerging(nn.Module):
    """
    Patch Merging Layer with quality-aware merging for medical images.
    """
    
    def __init__(
        self, 
        input_resolution: Tuple[int, int],
        dim: int, 
        norm_layer: nn.Module = nn.LayerNorm,
        quality_aware: bool = True
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.quality_aware = quality_aware
        
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
        if self.quality_aware:
            self.quality_weight = nn.Sequential(
                nn.Linear(4 * dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, 4),
                nn.Softmax(dim=-1)
            )
    
    def forward(
        self, 
        x: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, H*W, C)
            quality_scores: Optional quality scores
            
        Returns:
            Output features of shape (B, H/2*W/2, 2*C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)
        
        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        # Quality-aware merging
        if self.quality_aware and quality_scores is not None:
            weights = self.quality_weight(x)  # B H/2*W/2 4
            x_components = x.view(B, -1, 4, C)
            x = (x_components * weights.unsqueeze(-1)).sum(dim=2)
            x = self.norm(x.view(B, -1, C))
        else:
            x = self.norm(x)
            x = self.reduction(x)
        
        return x


class SwinTransformerStage(nn.Module):
    """
    A Swin Transformer Stage consisting of multiple blocks.
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: Union[float, List[float]] = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        medical_adaptations: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                medical_adaptations=medical_adaptations
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
                quality_aware=medical_adaptations
            )
        else:
            self.downsample = None
    
    def forward(
        self, 
        x: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the stage."""
        H, W = self.input_resolution
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, quality_scores)
            else:
                x = blk(x, H, W, quality_scores)
        
        if self.downsample is not None:
            x = self.downsample(x, quality_scores)
            
        return x, quality_scores


class SwinTransformer(VisionTransformerBase):
    """
    Swin Transformer for thyroid microscopy classification.
    
    A hierarchical vision transformer using shifted windows with medical imaging adaptations.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 1,
        num_classes: int = 2,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        medical_adaptations: bool = True,
        quality_dim: int = 4,
        **kwargs
    ):
        # Initialize parent class with compatible parameters
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=sum(depths),  # Total depth
            num_heads=num_heads[0],  # Use first stage heads for compatibility
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            quality_dim=quality_dim,
            **kwargs
        )
        
        # Swin-specific attributes
        self.depths = depths
        self.num_layers = len(depths)
        self.window_size = window_size
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.use_checkpoint = use_checkpoint
        self.medical_adaptations = medical_adaptations
        
        # Split image into non-overlapping patches
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        if self.patch_norm:
            self.patch_norm_layer = norm_layer(embed_dim)
        else:
            self.patch_norm_layer = nn.Identity()
        
        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerStage(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    img_size // patch_size // (2 ** i_layer),
                    img_size // patch_size // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                medical_adaptations=medical_adaptations
            )
            self.layers.append(layer)
        
        # Final normalization and head
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        # Medical imaging specific heads
        if self.medical_adaptations:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.num_features, self.num_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(self.num_features // 2, num_classes)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through Swin stages."""
        # Patch embedding
        x = self.patch_embed(x)  # B C H W
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.patch_norm_layer(x)
        
        # Add absolute position embedding if enabled
        if self.ape:
            x = x + self.absolute_pos_embed
        
        # Apply dropout
        x = self.pos_drop(x)
        
        # Extract quality scores if available
        quality_scores = None
        if hasattr(self, 'quality_encoder') and self.quality_encoder is not None:
            quality_scores = self.compute_quality_scores(x)
        
        # Process through stages
        for layer in self.layers:
            x, quality_scores = layer(x, quality_scores)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with optional feature/uncertainty return.
        
        Args:
            x: Input images
            return_features: Return intermediate features
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            predictions or tuple of (predictions, features, uncertainty)
        """
        features = self.forward_features(x)
        logits = self.head(features)
        
        outputs = [logits]
        
        if return_features:
            outputs.append(features)
            
        if return_uncertainty and self.medical_adaptations:
            uncertainty = self.uncertainty_head(features)
            outputs.append(uncertainty)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights with adaptation for medical imaging."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Handle grayscale conversion
        if 'patch_embed.proj.weight' in state_dict:
            weight = state_dict['patch_embed.proj.weight']
            if weight.shape[1] == 3 and self.in_chans == 1:
                # Average RGB channels
                weight = weight.mean(dim=1, keepdim=True)
                state_dict['patch_embed.proj.weight'] = weight
        
        # Load with relaxed strictness for medical adaptations
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if not strict and (missing_keys or unexpected_keys):
            warnings.warn(f"Loaded pretrained weights with missing keys: {missing_keys}, "
                         f"unexpected keys: {unexpected_keys}")
        
        return self


# Factory functions for standard configurations
import timm
import warnings
from typing import Dict, Any, Optional

def load_pretrained_swin_from_timm(
    model_name: str,
    pretrained_cfg: Dict[str, Any],
    in_chans: int = 1,
    num_classes: int = 2,
    **kwargs
) -> nn.Module:
    """Load a pretrained Swin model from timm and adapt it."""
    
    # Get timm model name from pretrained_cfg or use default mapping
    timm_model_map = {
        'swin_tiny': 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224', 
        'swin_base': 'swin_base_patch4_window7_224',
        'swin_large': 'swin_large_patch4_window12_384',
        'swin_medical': 'swin_small_patch4_window7_224',  # Use small as base for medical
    }
    
    # Use URL from config if it's a valid timm model name
    timm_model_name = pretrained_cfg.get('url', timm_model_map.get(model_name, model_name))
    
    # Remove 'microsoft/' prefix if present
    if timm_model_name.startswith('microsoft/'):
        timm_model_name = timm_model_name.replace('microsoft/', '')
    
    print(f"Loading pretrained weights from timm: {timm_model_name}")
    
    try:
        # Create the pretrained model
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3  # Load with RGB weights initially
        )
        
        # Adapt for grayscale input if needed
        if in_chans == 1:
            # Get the patch embedding layer
            patch_embed = model.patch_embed
            old_proj = patch_embed.proj
            
            # Create new conv layer for grayscale
            new_proj = nn.Conv2d(
                1, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding
            )
            
            # Average RGB weights to grayscale
            with torch.no_grad():
                new_proj.weight.data = old_proj.weight.data.mean(dim=1, keepdim=True)
                if old_proj.bias is not None:
                    new_proj.bias.data = old_proj.bias.data
            
            # Replace the projection layer
            patch_embed.proj = new_proj
            model.patch_embed = patch_embed
            
            print(f"Adapted model for grayscale input (in_chans={in_chans})")
        
        # Update any additional parameters that might be in kwargs
        if 'drop_rate' in kwargs:
            model.drop_rate = kwargs['drop_rate']
        if 'drop_path_rate' in kwargs:
            model.drop_path_rate = kwargs['drop_path_rate']
            
        print(f"Successfully loaded pretrained {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
        
    except Exception as e:
        print(f"Failed to load pretrained weights: {e}")
        print("Falling back to random initialization")
        return None


# Factory functions for standard configurations
def create_swin_tiny(pretrained: bool = False, pretrained_cfg: Optional[Dict] = None, **kwargs) -> SwinTransformer:
    """Create Swin-Tiny model."""
    
    # Try to load pretrained from timm if requested
    if pretrained and pretrained_cfg is not None:
        timm_model = load_pretrained_swin_from_timm('swin_tiny', pretrained_cfg, **kwargs)
        if timm_model is not None:
            return timm_model
    
    # Fall back to custom implementation
    defaults = {
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
    }
    
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    
    model = SwinTransformer(**kwargs)
    
    if pretrained and pretrained_cfg is None:
        warnings.warn("Pretrained weights requested but no pretrained_cfg provided")
    
    return model


def create_swin_small(pretrained: bool = False, pretrained_cfg: Optional[Dict] = None, **kwargs) -> SwinTransformer:
    """Create Swin-Small model."""
    
    # Try to load pretrained from timm if requested
    if pretrained and pretrained_cfg is not None:
        timm_model = load_pretrained_swin_from_timm('swin_small', pretrained_cfg, **kwargs)
        if timm_model is not None:
            return timm_model
    
    # Fall back to custom implementation
    defaults = {
        'embed_dim': 96,
        'depths': [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
    }
    
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    
    model = SwinTransformer(**kwargs)
    
    if pretrained and pretrained_cfg is None:
        warnings.warn("Pretrained weights requested but no pretrained_cfg provided")
    
    return model


def create_swin_base(pretrained: bool = False, pretrained_cfg: Optional[Dict] = None, **kwargs) -> SwinTransformer:
    """Create Swin-Base model."""
    
    # Try to load pretrained from timm if requested
    if pretrained and pretrained_cfg is not None:
        timm_model = load_pretrained_swin_from_timm('swin_base', pretrained_cfg, **kwargs)
        if timm_model is not None:
            return timm_model
    
    # Fall back to custom implementation
    defaults = {
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        'window_size': 7,
    }
    
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    
    model = SwinTransformer(**kwargs)
    
    if pretrained and pretrained_cfg is None:
        warnings.warn("Pretrained weights requested but no pretrained_cfg provided")
    
    return model


def create_swin_large(pretrained: bool = False, pretrained_cfg: Optional[Dict] = None, **kwargs) -> SwinTransformer:
    """Create Swin-Large model (Blackwell GPU exclusive)."""
    
    # Try to load pretrained from timm if requested
    if pretrained and pretrained_cfg is not None:
        timm_model = load_pretrained_swin_from_timm('swin_large', pretrained_cfg, **kwargs)
        if timm_model is not None:
            return timm_model
    
    # Fall back to custom implementation
    defaults = {
        'embed_dim': 192,
        'depths': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        'window_size': 7,
    }
    
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    
    model = SwinTransformer(**kwargs)
    
    if pretrained and pretrained_cfg is None:
        warnings.warn("Pretrained weights requested but no pretrained_cfg provided")
    
    return model


def create_swin_medical(
    variant: str = 'small',
    window_sizes: List[int] = [5, 7, 9, 7],
    pretrained: bool = False,
    pretrained_cfg: Optional[Dict] = None,
    **kwargs
) -> SwinTransformer:
    """
    Create Swin model with medical imaging optimizations.
    Uses Swin-Small as base when loading pretrained weights.
    """
    
    # Try to load pretrained from timm if requested
    if pretrained and pretrained_cfg is not None:
        # For medical variant, use swin_small as base
        timm_model = load_pretrained_swin_from_timm('swin_medical', pretrained_cfg, **kwargs)
        if timm_model is not None:
            print("Note: Using Swin-Small pretrained weights as base for medical variant")
            return timm_model
    
    # Fall back to custom implementation
    configs = {
        'tiny': {'embed_dim': 96, 'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24]},
        'small': {'embed_dim': 96, 'depths': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
        'base': {'embed_dim': 128, 'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]},
    }
    
    if variant not in configs:
        raise ValueError(f"Variant {variant} not recognized")
    
    config = configs[variant]
    
    for key, value in config.items():
        kwargs.setdefault(key, value)
    
    kwargs.setdefault('window_size', window_sizes[0])
    kwargs.setdefault('medical_adaptations', True)
    
    model = SwinTransformer(**kwargs)
    
    if pretrained and pretrained_cfg is None:
        warnings.warn("Pretrained weights requested but no pretrained_cfg provided")
    
    return model


if __name__ == "__main__":
    # Test the implementation
    model = create_swin_tiny(in_chans=1, num_classes=2)
    x = torch.randn(2, 1, 256, 256)
    
    # Test forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with features and uncertainty
    output, features, uncertainty = model(x, return_features=True, return_uncertainty=True)
    print(f"\nWith features and uncertainty:")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")