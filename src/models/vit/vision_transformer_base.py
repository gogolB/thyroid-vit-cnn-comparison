"""
Vision Transformer Base Class
Provides common functionality for all transformer variants
Compatible with existing CNN infrastructure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
from collections import OrderedDict


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding with quality-aware features
    Supports both Conv2d and Linear projection
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
        strict_img_size: bool = True,
        projection_type: str = 'conv',  # 'conv' or 'linear'
        quality_aware: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten
        self.projection_type = projection_type
        self.strict_img_size = strict_img_size
        self.quality_aware = quality_aware

        if projection_type == 'conv':
            self.proj = nn.Conv2d(
                in_chans, embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias
            )
        else:  # linear
            self.proj = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                         p1=patch_size, p2=patch_size),
                nn.Linear(patch_size * patch_size * in_chans, embed_dim, bias=bias)
            )
            
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Quality-aware patch scoring
        if quality_aware:
            self.quality_score = nn.Sequential(
                nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, C, H, W = x.shape
        if self.strict_img_size:
            assert H == self.img_size and W == self.img_size, \
                f"Input size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        # Calculate quality scores if enabled
        quality_scores = None
        if self.quality_aware and hasattr(self, 'quality_score'):
            quality_scores = self.quality_score(x)  # B, 1, H, W
            # Downsample to patch resolution
            quality_scores = F.avg_pool2d(quality_scores, self.patch_size)
            quality_scores = quality_scores.flatten(1)  # B, num_patches
        
        # Project patches
        if self.projection_type == 'conv':
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        else:
            x = self.proj(x)  # Already B, num_patches, embed_dim
            
        x = self.norm(x)
        return x, quality_scores


class Attention(nn.Module):
    """
    Multi-head self-attention with attention map storage
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        store_attention: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by num_heads {num_heads}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.store_attention = store_attention
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Storage for attention maps
        self.attention_maps = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Store attention maps if requested
        if self.store_attention and not self.training:
            self.attention_maps = attn.detach().cpu()
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Mlp(nn.Module):
    """Feed-forward network"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with attention and MLP
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        store_attention: bool = True
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            store_attention=store_attention
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerBase(pl.LightningModule):
    """
    Base class for Vision Transformer variants
    Compatible with existing training infrastructure
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
        embed_layer: nn.Module = PatchEmbed,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        class_token: bool = True,
        pos_embed_type: str = 'learnable',  # 'learnable' or 'sinusoidal'
        pool_type: str = 'cls',  # 'cls' or 'gap'
        quality_aware: bool = True,
        projection_type: str = 'conv',
        store_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Basic parameters
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.class_token = class_token
        self.pool_type = pool_type
        self.store_attention = store_attention
        
        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            projection_type=projection_type,
            quality_aware=quality_aware
        )
        
        # Class token
        if class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        num_positions = self.num_patches + (1 if class_token else 0)
        if pos_embed_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        elif pos_embed_type == 'sinusoidal':
            self.register_buffer('pos_embed', self._create_sinusoidal_embedding(num_positions, embed_dim))
        else:
            raise ValueError(f"Unknown position embedding type: {pos_embed_type}")
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks (to be implemented by subclasses)
        self.blocks = None  # Will be defined in subclasses
        
        # Normalization
        self.norm = norm_layer(embed_dim)
        
        # Representation layer (optional)
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
            
        # Classification head
        self.head = nn.Linear(self.num_features, num_classes)
        
        # Metrics (matching CNN implementation)
        self.train_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_auc = AUROC(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='binary' if num_classes == 2 else 'multiclass', num_classes=num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Storage for attention maps
        self._attention_storage = []
        
    def _create_sinusoidal_embedding(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal position embeddings"""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_embedding = torch.zeros(1, num_positions, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position * div_term)
        pos_embedding[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding
        
    def _init_weights(self):
        """Initialize weights using truncated normal distribution"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        # Initialize patch embedding projection
        if hasattr(self.patch_embed, 'proj'):
            if isinstance(self.patch_embed.proj, nn.Conv2d):
                fan_in = self.patch_embed.proj.in_channels * self.patch_embed.proj.kernel_size[0] ** 2
                nn.init.trunc_normal_(self.patch_embed.proj.weight, std=math.sqrt(2.0 / fan_in))
                
        # Initialize position embedding
        if hasattr(self, 'pos_embed') and self.hparams.pos_embed_type == 'learnable':
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
        # Initialize class token
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from input"""
        # Patch embedding with optional quality scores
        x, quality_scores = self.patch_embed(x)
        
        # Add class token
        if self.class_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Clear attention storage
        if self.store_attention and not self.training:
            self._attention_storage = []
        
        # Apply transformer blocks
        if self.blocks is not None:
            x = self.blocks(x)
        
        # Store attention maps from each block
        if self.store_attention and not self.training:
            for block in self.blocks:
                if hasattr(block.attn, 'attention_maps') and block.attn.attention_maps is not None:
                    self._attention_storage.append(block.attn.attention_maps)
        
        # Final normalization
        x = self.norm(x)
        
        # Pooling
        if self.pool_type == 'cls' and self.class_token:
            x = x[:, 0]
        else:  # Global average pooling
            x = x[:, 1:].mean(dim=1) if self.class_token else x.mean(dim=1)
            
        # Pre-logits
        x = self.pre_logits(x)
        
        return x, quality_scores
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features, _ = self.forward_features(x)
        x = self.head(features)
        return x
    
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get stored attention maps"""
        if not self._attention_storage:
            return None
        return torch.stack(self._attention_storage)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        features, _ = self.forward_features(x)
        return features
    
    def training_step(self, batch, batch_idx):
        """Training step - compatible with CNN implementation"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        
        # Log metrics (using underscore format)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - compatible with CNN implementation"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        
        # For AUC, we need probabilities
        probs = torch.softmax(logits, dim=1)
        if self.num_classes == 2:
            auc = self.val_auc(probs[:, 1], y)
        else:
            auc = self.val_auc(probs, y)
            
        f1 = self.val_f1(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        """Test step - compatible with CNN implementation"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        """Configure optimizer with layer-wise learning rate decay"""
        # This will be overridden in subclasses or training module
        # Default implementation for compatibility
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.get('learning_rate', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.05)
        )
        return optimizer
    
    def get_parameter_groups(self, weight_decay: float = 0.05, layer_decay: float = 0.75):
        """
        Get parameter groups for layer-wise learning rate decay
        Compatible with ViT training best practices
        """
        param_groups = []
        
        # No decay for certain parameters
        no_decay = ['bias', 'norm', 'cls_token', 'pos_embed']
        
        # Build layer groups
        if hasattr(self, 'blocks') and self.blocks is not None:
            num_layers = len(self.blocks)
            layer_scales = {}
            
            for i in range(num_layers):
                layer_scales[f'blocks.{i}'] = layer_decay ** (num_layers - i - 1)
            
            # Group parameters
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Determine weight decay
                if any(nd in name for nd in no_decay):
                    wd = 0.
                else:
                    wd = weight_decay
                    
                # Determine layer scale
                scale = 1.0
                for layer_name, layer_scale in layer_scales.items():
                    if layer_name in name:
                        scale = layer_scale
                        break
                
                # Add parameter group
                param_groups.append({
                    'params': [param],
                    'weight_decay': wd,
                    'lr_scale': scale,
                    'name': name
                })
        else:
            # Fallback for models without blocks
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if any(nd in name for nd in no_decay):
                    param_groups.append({
                        'params': [param],
                        'weight_decay': 0.,
                        'name': name
                    })
                else:
                    param_groups.append({
                        'params': [param],
                        'weight_decay': weight_decay,
                        'name': name
                    })
        
        return param_groups