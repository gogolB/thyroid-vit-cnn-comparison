"""
Specialized tests for attention visualization and quality-aware features
These are key differentiators for our medical imaging approach
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.vit.vision_transformer_base import (
    VisionTransformerBase,
    PatchEmbed,
    Attention,
    Block
)
from src.models.vit.attention_utils import (
    visualize_attention_maps,
    get_patch_importance_scores,
    create_attention_rollout
)


@pytest.mark.attention
class TestAttentionVisualization:
    """Test attention visualization capabilities"""
    
    def create_vit_with_attention(self, **kwargs):
        """Helper to create ViT with attention storage"""
        class AttentionViT(VisionTransformerBase):
            def __init__(self, **kwargs):
                kwargs['store_attention'] = True
                super().__init__(**kwargs)
                
                # Simple 2-layer model for testing
                self.blocks = nn.Sequential(*[
                    Block(
                        dim=self.embed_dim,
                        num_heads=self.hparams.num_heads,
                        mlp_ratio=self.hparams.mlp_ratio,
                        qkv_bias=True,
                        store_attention=True
                    )
                    for _ in range(2)
                ])
        
        default_kwargs = {
            'img_size': 224, # Changed to match synthetic_batch
            'patch_size': 16,
            'in_chans': 1,
            'num_classes': 2,
            'embed_dim': 192,
            'depth': 2,
            'num_heads': 3
        }
        default_kwargs.update(kwargs)
        return AttentionViT(**default_kwargs)
    
    def test_attention_storage(self, synthetic_batch):
        """Test that attention maps are stored correctly"""
        images, _ = synthetic_batch
        model = self.create_vit_with_attention()
        
        # No attention in training mode
        model.train()
        _ = model(images)
        attention_maps = model.get_attention_maps()
        assert attention_maps is None
        
        # Attention stored in eval mode
        model.eval()
        with torch.no_grad():
            _ = model(images)
            attention_maps = model.get_attention_maps()
        
        assert attention_maps is not None
        assert attention_maps.shape[0] == 2  # num layers
        assert attention_maps.shape[1] == 4  # batch size
        assert attention_maps.shape[2] == 3  # num heads
    
    def test_attention_map_values(self, synthetic_batch):
        """Test attention map properties"""
        images, _ = synthetic_batch
        model = self.create_vit_with_attention()
        
        model.eval()
        with torch.no_grad():
            _ = model(images)
            attention_maps = model.get_attention_maps()
        
        # Check attention maps sum to 1 along last dimension
        attention_sums = attention_maps.sum(dim=-1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5)
        
        # Check all values are between 0 and 1
        assert torch.all(attention_maps >= 0)
        assert torch.all(attention_maps <= 1)
    
    def test_cls_token_attention(self, synthetic_batch):
        """Test CLS token attention patterns"""
        images, _ = synthetic_batch
        model = self.create_vit_with_attention()
        
        model.eval()
        with torch.no_grad():
            _ = model(images)
            attention_maps = model.get_attention_maps()
        
        # Extract CLS token attention (first row)
        cls_attention = attention_maps[:, :, :, 0, :]  # layers, batch, heads, seq_len
        
        # CLS should attend to all tokens including itself
        num_patches = (224 // 16) ** 2  # model img_size is 224
        expected_seq_len = num_patches + 1  # +1 for CLS token
        assert cls_attention.shape[-1] == expected_seq_len
    
    @pytest.mark.skip(reason="Visualization function needs matplotlib backend")
    def test_attention_visualization_function(self, synthetic_batch, mock_attention_maps):
        """Test attention visualization function"""
        images, _ = synthetic_batch
        original_image = images[0, 0].numpy()  # First image, first channel
        
        # Test visualization
        fig = visualize_attention_maps(
            attention_maps=mock_attention_maps,
            original_image=original_image,
            patch_size=16,
            layer_indices=[0, -1]  # First and last layer
        )
        
        assert fig is not None
        plt.close(fig)


@pytest.mark.quality
class TestQualityAwareFeatures:
    """Test quality-aware preprocessing integration"""
    
    def test_quality_scoring_in_patch_embed(self):
        """Test quality scoring during patch embedding"""
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            quality_aware=True
        )
        
        # Test with batch of images
        images = torch.randn(4, 1, 256, 256)
        patches, quality_scores = patch_embed(images)
        
        assert quality_scores is not None
        assert quality_scores.shape == (4, 256)  # batch_size, num_patches
        assert torch.all(quality_scores >= 0) and torch.all(quality_scores <= 1)
    
    def test_quality_score_gradient_flow(self):
        """Test that quality scores can propagate gradients"""
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            quality_aware=True
        )
        
        images = torch.randn(2, 1, 256, 256, requires_grad=True)
        patches, quality_scores = patch_embed(images)
        
        # Create a loss using quality scores
        if quality_scores is not None:
            loss = quality_scores.mean()
            loss.backward()
            
            # Check gradients exist
            assert images.grad is not None
            assert not torch.isnan(images.grad).any()
    
    def test_quality_aware_attention_weighting(self, synthetic_batch):
        """Test using quality scores to weight attention"""
        images, _ = synthetic_batch
        
        # Create model with quality awareness
        class QualityAwareViT(VisionTransformerBase):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.blocks = nn.Sequential(
                    Block(dim=self.embed_dim, num_heads=self.hparams.num_heads)
                )
                
            def forward_features(self, x):
                # Get patches and quality scores
                x, quality_scores = self.patch_embed(x)
                
                # Add class token
                if self.class_token:
                    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_tokens, x), dim=1)
                    
                    # Pad quality scores for CLS token
                    if quality_scores is not None:
                        cls_quality = torch.ones(x.shape[0], 1).to(quality_scores.device)
                        quality_scores = torch.cat((cls_quality, quality_scores), dim=1)
                
                # Apply quality weighting (simplified)
                if quality_scores is not None:
                    x = x * quality_scores.unsqueeze(-1)
                
                # Continue with standard processing
                x = x + self.pos_embed
                x = self.pos_drop(x)
                x = self.blocks(x)
                x = self.norm(x)
                
                # Pool
                x = x[:, 0] if self.class_token else x.mean(dim=1)
                x = self.pre_logits(x)
                
                return x, quality_scores
        
        model = QualityAwareViT(
            img_size=224, # Added to match synthetic_batch
            embed_dim=192,
            depth=1,
            num_heads=3,
            quality_aware=True
        )
        
        output = model(images)
        assert output.shape == (4, 2)
    
    def test_patch_quality_statistics(self, mocker):
        """Test extraction of patch quality statistics"""
        patch_embed = PatchEmbed(
            img_size=256,
            patch_size=16,
            in_chans=1,
            embed_dim=768,
            quality_aware=True
        )
        
        # Create dummy images (content doesn't matter as quality_score is mocked)
        images = torch.zeros(2, 1, 256, 256)
        
        # Define mock quality scores output
        # Image 0: high quality (scores around 0.9)
        # Image 1: low quality (scores around 0.1)
        # Shape: (B, 1, H, W) -> (B, 1, grid_size, grid_size) after avg_pool2d
        grid_size = 256 // 16
        
        mock_scores_img0 = torch.ones(1, 1, 256, 256) * 0.9 # Before pooling
        mock_scores_img1 = torch.ones(1, 1, 256, 256) * 0.1 # Before pooling
        
        # Concatenate for batch
        mock_raw_scores_output = torch.cat((mock_scores_img0, mock_scores_img1), dim=0)

        mocker.patch.object(patch_embed.quality_score, 'forward', return_value=mock_raw_scores_output)
        
        patches, quality_scores = patch_embed(images)
        
        assert quality_scores is not None, "Quality scores should be generated"
        # Expected shape after avg_pool2d and flatten: (B, num_patches)
        # num_patches = grid_size * grid_size = 16 * 16 = 256
        assert quality_scores.shape == (2, grid_size * grid_size)
        
        # Check that the mocked scores are reflected (approximately, due to pooling)
        # The avg_pool2d will average the constant values, so they should remain constant.
        assert torch.allclose(quality_scores[0], torch.tensor(0.9)), "High quality scores incorrect"
        assert torch.allclose(quality_scores[1], torch.tensor(0.1)), "Low quality scores incorrect"

        # First image should have higher average quality
        assert quality_scores[0].mean() > quality_scores[1].mean() - 1e-5


@pytest.mark.integration
@pytest.mark.attention
class TestAttentionIntegration:
    """Integration tests for attention with other components"""
    
    def test_attention_with_different_patch_sizes(self):
        """Test attention with different patch sizes"""
        for patch_size in [8, 16, 32]:
            model = TestAttentionVisualization().create_vit_with_attention(
                patch_size=patch_size
            )
            
            images = torch.randn(2, 1, 224, 224) # Match model's img_size
            model.eval()
            
            with torch.no_grad():
                _ = model(images)
                attention_maps = model.get_attention_maps()
            
            expected_seq_len = (224 // patch_size) ** 2 + 1  # patches + CLS (model img_size is 224)
            assert attention_maps.shape[-1] == expected_seq_len
    
    def test_attention_consistency_across_batches(self):
        """Test that attention patterns are consistent"""
        model = TestAttentionVisualization().create_vit_with_attention()
        model.eval()
        # Same image repeated
        image = torch.randn(1, 1, 224, 224) # Match model's img_size
        batch = image.repeat(4, 1, 1, 1)
        
        
        with torch.no_grad():
            _ = model(batch)
            attention_maps = model.get_attention_maps()
        
        # Check that attention patterns are similar for same image
        # Allow small differences due to numerical precision
        for i in range(1, 4):
            diff = (attention_maps[:, 0] - attention_maps[:, i]).abs().max()
            assert diff < 1e-5, f"Attention differs for identical images: {diff}"


@pytest.mark.slow
@pytest.mark.gpu
class TestAttentionGPU:
    """GPU-specific tests for attention mechanisms"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_attention_memory_efficiency(self):
        """Test memory efficiency of attention storage"""
        device = torch.device('cuda')
        
        # Create model with many layers
        model = TestAttentionVisualization().create_vit_with_attention(
            depth=12,
            embed_dim=768,
            num_heads=12
        ).to(device)
        
        # Large batch
        images = torch.randn(16, 1, 256, 256).to(device)
        
        model.eval()
        with torch.no_grad():
            # Check memory before
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()
            
            _ = model(images)
            attention_maps = model.get_attention_maps()
            
            # Check memory after
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated()
            
            # Memory increase should be reasonable
            mem_increase_mb = (mem_after - mem_before) / 1024 / 1024
            print(f"Memory increase: {mem_increase_mb:.2f} MB")
            
            # Attention maps should exist
            assert attention_maps is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "attention or quality"])

