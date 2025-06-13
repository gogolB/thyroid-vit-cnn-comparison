"""
PyTest configuration and fixtures for Vision Transformer testing
Provides synthetic data generation and common test utilities
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Tuple, Dict, List
from omegaconf import DictConfig, OmegaConf


# Set random seeds for reproducibility
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests"""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Get the appropriate device for testing"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def synthetic_batch(batch_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic batch of thyroid images and labels"""
    # Generate synthetic grayscale images (batch_size, 1, 256, 256)
    images = torch.randn(batch_size, 1, 256, 256)
    
    # Add some structure to make it more realistic
    # Create circular patterns resembling thyroid nodules
    for i in range(batch_size):
        center_x = np.random.randint(64, 192)
        center_y = np.random.randint(64, 192)
        radius = np.random.randint(20, 50)
        
        y, x = np.ogrid[:256, :256]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Add pattern to image
        if i % 2 == 0:  # Normal
            images[i, 0, mask] += torch.randn(mask.sum()) * 0.3
        else:  # Cancerous (more irregular)
            images[i, 0, mask] += torch.randn(mask.sum()) * 0.5
            # Add some noise for irregularity
            noise_mask = np.random.random(mask.shape) > 0.7
            images[i, 0, mask & noise_mask] += torch.randn((mask & noise_mask).sum()) * 0.3
    
    # Normalize to [-1, 1] range
    images = torch.clamp(images, -1, 1)
    
    # Generate labels (0: normal, 1: cancerous)
    labels = torch.tensor([i % 2 for i in range(batch_size)])
    
    return images, labels


@pytest.fixture
def synthetic_dataset_generator():
    """Factory fixture to generate synthetic datasets of various sizes"""
    def _generate(num_samples: int = 100, img_size: int = 256):
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create synthetic image
            img = torch.randn(1, img_size, img_size)
            
            # Add patterns based on class
            if i % 2 == 0:  # Normal
                # Add smooth gaussian-like pattern
                center = img_size // 2
                y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
                gaussian = torch.exp(-((x - center)**2 + (y - center)**2) / (2 * 50**2))
                img += gaussian.unsqueeze(0) * 0.5
            else:  # Cancerous
                # Add irregular patterns
                num_spots = np.random.randint(3, 7)
                for _ in range(num_spots):
                    cx = np.random.randint(img_size // 4, 3 * img_size // 4)
                    cy = np.random.randint(img_size // 4, 3 * img_size // 4)
                    radius = np.random.randint(10, 30)
                    y, x = torch.meshgrid(torch.arange(img_size), torch.arange(img_size))
                    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
                    img[0, mask] += torch.randn(mask.sum()) * 0.7
            
            images.append(img)
            labels.append(i % 2)
        
        return torch.stack(images), torch.tensor(labels)
    
    return _generate


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_vit_config():
    """Sample configuration for ViT models"""
    config = {
        'model': {
            'name': 'vit_tiny',
            'type': 'vision_transformer',
            'params': {
                'img_size': 256,
                'patch_size': 16,
                'in_chans': 1,
                'num_classes': 2,
                'embed_dim': 192,
                'depth': 12,
                'num_heads': 3,
                'mlp_ratio': 4.0,
                'qkv_bias': True,
                'drop_rate': 0.0,
                'attn_drop_rate': 0.0,
                'drop_path_rate': 0.1,
                'class_token': True,
                'pos_embed_type': 'learnable',
                'pool_type': 'cls',
                'quality_aware': True,
                'store_attention': True
            }
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.05,
            'warmup_epochs': 10,
            'layer_decay': 0.75
        },
        'dataset': {
            'name': 'thyroid_synthetic',
            'num_classes': 2,
            'image_size': 256
        }
    }
    return OmegaConf.create(config)


@pytest.fixture
def sample_quality_scores():
    """Generate sample quality scores for testing quality-aware features"""
    batch_size = 4
    num_patches = 256  # 16x16 patches for 256x256 image
    
    # Generate random quality scores between 0.5 and 1.0
    quality_scores = torch.rand(batch_size, num_patches) * 0.5 + 0.5
    
    return quality_scores


@pytest.fixture
def mock_attention_maps():
    """Generate mock attention maps for testing visualization"""
    num_layers = 12
    batch_size = 2
    num_heads = 3
    seq_len = 197  # 196 patches + 1 cls token
    
    # Generate random attention maps
    attention_maps = torch.rand(num_layers, batch_size, num_heads, seq_len, seq_len)
    
    # Make them more realistic by applying softmax
    attention_maps = torch.softmax(attention_maps, dim=-1)
    
    return attention_maps


@pytest.fixture
def pretrained_weights_mock():
    """Mock pretrained weights for testing weight loading"""
    # Create a state dict that mimics timm pretrained weights
    state_dict = {
        'patch_embed.proj.weight': torch.randn(192, 1, 16, 16),
        'patch_embed.proj.bias': torch.randn(192),
        'pos_embed': torch.randn(1, 197, 192),
        'cls_token': torch.randn(1, 1, 192),
        'blocks.0.norm1.weight': torch.randn(192),
        'blocks.0.norm1.bias': torch.randn(192),
        'blocks.0.attn.qkv.weight': torch.randn(192*3, 192),
        'blocks.0.attn.qkv.bias': torch.randn(192*3),
        'blocks.0.attn.proj.weight': torch.randn(192, 192),
        'blocks.0.attn.proj.bias': torch.randn(192),
        'blocks.0.norm2.weight': torch.randn(192),
        'blocks.0.norm2.bias': torch.randn(192),
        'blocks.0.mlp.fc1.weight': torch.randn(768, 192),
        'blocks.0.mlp.fc1.bias': torch.randn(768),
        'blocks.0.mlp.fc2.weight': torch.randn(192, 768),
        'blocks.0.mlp.fc2.bias': torch.randn(192),
        'norm.weight': torch.randn(192),
        'norm.bias': torch.randn(192),
        'head.weight': torch.randn(1000, 192),  # ImageNet classes
        'head.bias': torch.randn(1000),
    }
    return state_dict

