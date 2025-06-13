"""
Attention visualization utilities for Vision Transformers
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import torch.nn as nn
import cv2


def visualize_attention_maps(
    attention_maps: torch.Tensor,
    original_image: np.ndarray,
    patch_size: int = 16,
    layer_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize attention maps from Vision Transformer
    
    Args:
        attention_maps: Tensor of shape (layers, batch, heads, seq_len, seq_len)
        original_image: Original input image
        patch_size: Size of each patch
        layer_indices: Which layers to visualize (default: last layer)
        save_path: Path to save visualization
        
    Returns:
        Matplotlib figure
    """
    if layer_indices is None:
        layer_indices = [-1]  # Last layer by default
        
    num_layers = len(layer_indices)
    fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))
    
    if num_layers == 0:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention maps
    for idx, layer_idx in enumerate(layer_indices):
        attn = attention_maps[layer_idx, 0]  # First sample in batch
        
        # Average over heads
        attn = attn.mean(dim=0)
        
        # Focus on CLS token attention
        cls_attn = attn[0, 1:]  # Skip CLS token itself
        
        # Reshape to image dimensions
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # Interpolate to original image size
        cls_attn = F.interpolate(
            cls_attn.unsqueeze(0).unsqueeze(0),
            size=original_image.shape[:2],
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Visualize
        im = axes[idx + 1].imshow(cls_attn, cmap='hot', alpha=0.8)
        axes[idx + 1].imshow(original_image, cmap='gray', alpha=0.2)
        axes[idx + 1].set_title(f'Layer {layer_idx} Attention')
        axes[idx + 1].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig


def get_patch_importance_scores(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Calculate importance scores for each patch
    
    Args:
        model: Vision Transformer model
        image: Input image tensor
        target_class: Target class for importance
        patch_size: Size of each patch
        
    Returns:
        Patch importance scores
    """
    model.eval()
    image.requires_grad = True
    
    # Forward pass
    output = model(image)
    
    # Backward pass for target class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients
    gradients = image.grad.data.abs()
    
    # Average over channels
    gradients = gradients.mean(dim=1)
    
    # Convert to patch importance
    B, H, W = gradients.shape
    ph = H // patch_size
    pw = W // patch_size
    
    patch_importance = gradients.reshape(B, ph, patch_size, pw, patch_size)
    patch_importance = patch_importance.mean(dim=(2, 4))
    
    return patch_importance


def create_attention_rollout(
    attention_maps: torch.Tensor,
    head_fusion: str = "mean"
) -> torch.Tensor:
    """
    Create attention rollout visualization
    
    Args:
        attention_maps: Raw attention maps from model
        head_fusion: How to fuse attention heads ('mean', 'max', 'min')
        
    Returns:
        Rolled out attention maps
    """
    # Implementation of attention rollout
    # (Details to be implemented based on paper)
    pass

