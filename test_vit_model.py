#!/usr/bin/env python3
"""
Test script for Vision Transformer models
Verifies implementation and parameter counts
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent if __file__ != '<stdin>' else Path.cwd()
sys.path.insert(0, str(project_root))

from src.models.vit.vit_models import (
    create_vit_tiny, create_vit_small, create_vit_base,
    create_vit_model, VIT_PARAMS
)
from src.models.vit import get_vit_model


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model_creation():
    """Test model instantiation"""
    print("=" * 60)
    print("Testing Vision Transformer Model Creation")
    print("=" * 60)
    
    # Test parameters
    batch_size = 2
    img_size = 256
    patch_size = 16
    in_chans = 1
    num_classes = 2
    
    models_to_test = [
        ('vit_tiny', create_vit_tiny, 5.7e6),
        ('vit_small', create_vit_small, 22e6),
        ('vit_base', create_vit_base, 86e6),
    ]
    
    for model_name, create_fn, expected_params in models_to_test:
        print(f"\n{'='*40}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*40}")
        
        # Create model using factory function
        model = create_fn(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=0.1
        )
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Expected parameters: ~{expected_params:,.0f}")
        
        # Check if within reasonable range (±20%)
        param_ratio = total_params / expected_params
        if 0.8 <= param_ratio <= 1.2:
            print("✓ Parameter count within expected range")
        else:
            print(f"⚠ Parameter count differs by {abs(1-param_ratio)*100:.1f}%")
        
        # Test forward pass
        x = torch.randn(batch_size, in_chans, img_size, img_size)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        print(f"\nForward pass test:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch_size}, {num_classes})")
        
        assert output.shape == (batch_size, num_classes), \
            f"Unexpected output shape: {output.shape}"
        print("  ✓ Forward pass successful")
        
        # Test attention maps
        attn_maps = model.get_attention_maps()
        if attn_maps is not None and len(attn_maps) > 0:
            print(f"\nAttention maps:")
            print(f"  Number of layers: {len(attn_maps)}")
            print(f"  Shape per layer: {attn_maps[0].shape}")
            print("  ✓ Attention maps stored correctly")
        else:
            print("\n  ℹ No attention maps (model in training mode)")
        
        # Test different batch sizes
        print(f"\nTesting variable batch sizes:")
        for bs in [1, 4, 8]:
            x_test = torch.randn(bs, in_chans, img_size, img_size)
            out_test = model(x_test)
            assert out_test.shape == (bs, num_classes)
            print(f"  ✓ Batch size {bs}: OK")


def test_registry_integration():
    """Test integration with model registry"""
    print("\n" + "="*60)
    print("Testing Model Registry Integration")
    print("="*60)
    
    # Test using registry function
    for model_name in ['vit_tiny', 'vit_small', 'vit_base']:
        print(f"\nCreating {model_name} via registry...")
        try:
            model = get_vit_model(model_name, img_size=256, num_classes=2)
            print(f"  ✓ Successfully created {model_name}")
            
            # Quick forward pass test
            x = torch.randn(1, 1, 256, 256)
            output = model(x)
            assert output.shape == (1, 2)
            print(f"  ✓ Forward pass OK")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def test_different_configurations():
    """Test models with different configurations"""
    print("\n" + "="*60)
    print("Testing Different Configurations")
    print("="*60)
    
    configs = [
        {"patch_size": 32, "desc": "32x32 patches"},
        {"pool_type": "gap", "desc": "Global Average Pooling"},
        {"class_token": False, "desc": "No CLS token"},
        {"pos_embed_type": "sinusoidal", "desc": "Sinusoidal pos embedding"},
    ]
    
    for config in configs:
        desc = config.pop("desc")
        print(f"\nTesting: {desc}")
        
        try:
            model = create_vit_tiny(img_size=256, num_classes=2, **config)
            x = torch.randn(2, 1, 256, 256)
            output = model(x)
            assert output.shape == (2, 2)
            print(f"  ✓ Configuration works correctly")
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    """Run all tests"""
    print("Vision Transformer Model Tests")
    print("Project: Thyroid Classification Phase 3")
    print()
    
    # Run tests
    test_model_creation()
    test_registry_integration()
    test_different_configurations()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()