"""
Comprehensive tests for Vision Transformer base implementation
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vit.vision_transformer_base import (
    VisionTransformerBase, 
    PatchEmbed, 
    Attention, 
    Block,
    DropPath,
    Mlp
)


class TestPatchEmbed:
    """Test patch embedding module"""
    
    def test_conv_projection(self, synthetic_batch):
        """Test Conv2d projection"""
        images, _ = synthetic_batch
        
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            projection_type='conv'
        )
        
        patches, quality_scores = patch_embed(images)
        
        assert patches.shape == (4, 256, 768)  # (batch, num_patches, embed_dim)
        assert quality_scores is None or quality_scores.shape == (4, 256)
    
    def test_linear_projection(self, synthetic_batch):
        """Test Linear projection"""
        images, _ = synthetic_batch
        
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            projection_type='linear'
        )
        
        patches, quality_scores = patch_embed(images)
        
        assert patches.shape == (4, 256, 768)
    
    def test_quality_aware_scoring(self, synthetic_batch):
        """Test quality-aware patch scoring"""
        images, _ = synthetic_batch
        
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            quality_aware=True
        )
        
        patches, quality_scores = patch_embed(images)
        
        assert quality_scores is not None
        assert quality_scores.shape == (4, 256)
        assert torch.all(quality_scores >= 0) and torch.all(quality_scores <= 1)
    
    def test_different_patch_sizes(self):
        """Test different patch sizes"""
        image = torch.randn(1, 1, 256, 256)
        
        for patch_size in [8, 16, 32]:
            patch_embed = PatchEmbed(
                img_size=256,
                patch_size=patch_size,
                in_chans=1,
                embed_dim=768
            )
            
            patches, _ = patch_embed(image)
            expected_patches = (256 // patch_size) ** 2
            assert patches.shape == (1, expected_patches, 768)


class TestAttention:
    """Test attention mechanism"""
    
    def test_attention_forward(self):
        """Test attention forward pass"""
        batch_size = 2
        seq_len = 197
        dim = 768
        
        attn = Attention(dim=dim, num_heads=12, qkv_bias=True)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_attention_map_storage(self):
        """Test attention map storage in eval mode"""
        attn = Attention(dim=768, num_heads=12, store_attention=True)
        x = torch.randn(2, 197, 768)
        
        # Should not store in training mode
        attn.train()
        _ = attn(x)
        assert attn.attention_maps is None
        
        # Should store in eval mode
        attn.eval()
        _ = attn(x)
        assert attn.attention_maps is not None
        assert attn.attention_maps.shape == (2, 12, 197, 197)
    
    def test_different_head_configurations(self):
        """Test different number of heads"""
        x = torch.randn(2, 197, 768)
        
        for num_heads in [1, 3, 6, 12]:
            if 768 % num_heads == 0:  # Only valid configurations
                attn = Attention(dim=768, num_heads=num_heads)
                output = attn(x)
                assert output.shape == x.shape


class TestBlock:
    """Test transformer block"""
    
    def test_block_forward(self):
        """Test block forward pass"""
        block = Block(
            dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            drop_path=0.1
        )
        
        x = torch.randn(2, 197, 768)
        output = block(x)
        assert output.shape == x.shape
    
    def test_drop_path(self):
        """Test stochastic depth"""
        drop_path = DropPath(drop_prob=0.5)
        x = torch.randn(10, 197, 768)
        
        # Should be identity in eval mode
        drop_path.eval()
        output_eval = drop_path(x)
        assert torch.allclose(output_eval, x)
        
        # Should drop paths in training mode
        drop_path.train()
        output_train = drop_path(x)
        assert output_train.shape == x.shape


class TestVisionTransformerBase:
    """Test Vision Transformer base class"""
    
    def create_simple_vit(self, **kwargs):
        """Helper to create a simple ViT with blocks"""
        class SimpleViT(VisionTransformerBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                
                # Add transformer blocks
                dpr = [x.item() for x in torch.linspace(0, self.hparams.drop_path_rate, self.hparams.depth)]
                self.blocks = nn.Sequential(*[
                    Block(
                        dim=self.embed_dim,
                        num_heads=self.hparams.num_heads,
                        mlp_ratio=self.hparams.mlp_ratio,
                        qkv_bias=self.hparams.qkv_bias,
                        drop=self.hparams.drop_rate,
                        attn_drop=self.hparams.attn_drop_rate,
                        drop_path=dpr[i],
                        store_attention=self.hparams.store_attention
                    )
                    for i in range(self.hparams.depth)
                ])
        
        default_kwargs = {
            'img_size': 256,
            'patch_size': 16,
            'in_chans': 1,
            'num_classes': 2,
            'embed_dim': 192,
            'depth': 2,  # Small for testing
            'num_heads': 3,
            'mlp_ratio': 4.0,
            'qkv_bias': True,
            'drop_path_rate': 0.1
        }
        default_kwargs.update(kwargs)
        return SimpleViT(**default_kwargs)
    
    def test_forward_pass(self, synthetic_batch):
        """Test complete forward pass"""
        images, labels = synthetic_batch
        model = self.create_simple_vit()
        
        output = model(images)
        assert output.shape == (4, 2)  # batch_size, num_classes
    
    def test_feature_extraction(self, synthetic_batch):
        """Test feature extraction"""
        images, _ = synthetic_batch
        model = self.create_simple_vit()
        
        features = model.extract_features(images)
        assert features.shape == (4, 192)  # batch_size, embed_dim
    
    def test_training_step(self, synthetic_batch):
        """Test training step"""
        model = self.create_simple_vit()
        
        loss = model.training_step(synthetic_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
    
    def test_validation_step(self, synthetic_batch):
        """Test validation step"""
        model = self.create_simple_vit()
        
        result = model.validation_step(synthetic_batch, 0)
        assert 'val_loss' in result
        assert 'val_acc' in result
    
    def test_position_embeddings(self):
        """Test different position embedding types"""
        # Learnable
        model_learn = self.create_simple_vit(pos_embed_type='learnable')
        assert hasattr(model_learn, 'pos_embed')
        assert isinstance(model_learn.pos_embed, nn.Parameter)
        
        # Sinusoidal
        model_sin = self.create_simple_vit(pos_embed_type='sinusoidal')
        assert hasattr(model_sin, 'pos_embed')
        assert not isinstance(model_sin.pos_embed, nn.Parameter)
    
    def test_pooling_strategies(self, synthetic_batch):
        """Test different pooling strategies"""
        images, _ = synthetic_batch
        
        # CLS token pooling
        model_cls = self.create_simple_vit(class_token=True, pool_type='cls')
        out_cls = model_cls(images)
        assert out_cls.shape == (4, 2)
        
        # Global average pooling
        model_gap = self.create_simple_vit(class_token=False, pool_type='gap')
        out_gap = model_gap(images)
        assert out_gap.shape == (4, 2)
    
    def test_attention_visualization(self, synthetic_batch):
        """Test attention map extraction"""
        images, _ = synthetic_batch
        model = self.create_simple_vit(store_attention=True)
        
        # Should store attention in eval mode
        model.eval()
        with torch.no_grad():
            _ = model(images)
            attention_maps = model.get_attention_maps()
        
        assert attention_maps is not None
        assert attention_maps.shape[0] == 2  # depth
        assert attention_maps.shape[1] == 4  # batch_size
    
    def test_parameter_groups(self):
        """Test parameter groups for layer-wise LR decay"""
        model = self.create_simple_vit(depth=4)
        
        param_groups = model.get_parameter_groups(
            weight_decay=0.05,
            layer_decay=0.75
        )
        
        # Check that we have parameter groups
        assert len(param_groups) > 0
        
        # Check layer decay is applied
        layer_scales = {}
        for group in param_groups:
            if 'lr_scale' in group and 'name' in group:
                if 'blocks' in group['name']:
                    layer_scales[group['name']] = group['lr_scale']
        
        # Later blocks should have lower lr_scale
        assert len(layer_scales) > 0


class TestIntegration:
    """Integration tests"""
    
    def test_different_image_sizes(self):
        """Test with different image sizes"""
        for img_size in [224, 256, 384]:
            model = TestVisionTransformerBase().create_simple_vit(
                img_size=img_size,
                patch_size=16
            )
            
            x = torch.randn(2, 1, img_size, img_size)
            output = model(x)
            assert output.shape == (2, 2)
    
    def test_gradient_flow(self, synthetic_batch):
        """Test gradient flow through model"""
        images, labels = synthetic_batch
        model = TestVisionTransformerBase().create_simple_vit()
        
        # Forward pass
        output = model(images)
        loss = torch.nn.functional.cross_entropy(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Skip quality score parameters - they're not used in loss
                if 'quality_score' in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
    def test_batch_sizes(self, batch_size):
        """Test with different batch sizes"""
        model = TestVisionTransformerBase().create_simple_vit()
        
        x = torch.randn(batch_size, 1, 256, 256)
        output = model(x)
        assert output.shape == (batch_size, 2)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

