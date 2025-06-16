"""
Tests for src.data.vit_transforms
"""
import pytest
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock, call

from src.data.vit_transforms import (
    RandAugment,
    QualityAwarePatchAugment,
    create_vit_transform,
    MixUp,
    CutMix
)

# Helper to create a dummy tensor image
def create_dummy_tensor_image(size=(3, 224, 224), grayscale=False, batch_size=None, dtype=torch.float32):
    channels = 1 if grayscale else size[0]
    if batch_size:
        return torch.rand(batch_size, channels, size[1], size[2], dtype=dtype)
    return torch.rand(channels, size[1], size[2], dtype=dtype)

# Helper to create a dummy PIL image
def create_dummy_pil_image(size=(224, 224), mode='L'): # L for grayscale, RGB for color
    return Image.new(mode, size)

# Tests for RandAugment
class TestRandAugment:
    def test_initialization_defaults(self):
        augmenter = RandAugment()
        assert augmenter.n == 2
        assert augmenter.m == 9
        assert augmenter.grayscale is True
        assert len(augmenter.augmentations) == 12 # 12 grayscale + 0 color

    def test_initialization_custom(self):
        augmenter = RandAugment(n=3, m=15, grayscale=False)
        assert augmenter.n == 3
        assert augmenter.m == 15
        assert augmenter.grayscale is False
        assert len(augmenter.augmentations) == 13 # 12 grayscale + 1 color ("Color")

    def test_get_magnitude(self):
        augmenter = RandAugment()
        # Test with m=0 (min_val), m=30 (max_val), m=15 (mid_val)
        assert augmenter._get_magnitude(0, 0.0, 1.0) == pytest.approx(0.0)
        assert augmenter._get_magnitude(30, 0.0, 1.0) == pytest.approx(1.0)
        assert augmenter._get_magnitude(15, 0.0, 1.0) == pytest.approx(0.5)
        assert augmenter._get_magnitude(9, 0.05, 1.95) == pytest.approx(0.05 + (1.90 * 9 / 30.0))

    @patch('random.choices')
    def test_forward_applies_n_ops(self, mock_choices):
        augmenter = RandAugment(n=2, m=5)
        mock_op1 = MagicMock(return_value=torch.tensor(1.0))
        mock_op2 = MagicMock(return_value=torch.tensor(2.0))
        mock_choices.return_value = [("Op1", mock_op1), ("Op2", mock_op2)]
        
        img_tensor = create_dummy_tensor_image(grayscale=True)
        output_img = augmenter(img_tensor)
        
        mock_choices.assert_called_once_with(augmenter.augmentations, k=2)
        mock_op1.assert_called_once_with(img_tensor, 5)
        mock_op2.assert_called_once_with(torch.tensor(1.0), 5) # output of op1 is input to op2
        assert torch.equal(output_img, torch.tensor(2.0))

    def test_auto_contrast_float(self):
        augmenter = RandAugment()
        img_tensor = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]], dtype=torch.float32) # C, H, W
        # Mock TF.autocontrast as its exact output is complex to replicate here
        with patch('torchvision.transforms.functional.autocontrast') as mock_tf_autocontrast:
            mock_tf_autocontrast.return_value = (img_tensor * 255).to(torch.uint8) # dummy return
            augmented_img = augmenter._auto_contrast(img_tensor.clone(), 0)
            mock_tf_autocontrast.assert_called_once()
            assert augmented_img.dtype == torch.float32
            assert augmented_img.shape == img_tensor.shape

    def test_auto_contrast_uint8(self):
        augmenter = RandAugment()
        img_tensor = (torch.rand(1, 32, 32) * 255).to(torch.uint8)
        with patch('torchvision.transforms.functional.autocontrast') as mock_tf_autocontrast:
            mock_tf_autocontrast.return_value = img_tensor # dummy return
            augmented_img = augmenter._auto_contrast(img_tensor.clone(), 0)
            mock_tf_autocontrast.assert_called_once()
            assert augmented_img.dtype == torch.uint8

    def test_brightness(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True)
        augmented_img = augmenter._brightness(img_tensor.clone(), 15) # m=15
        assert augmented_img.shape == img_tensor.shape
        # Brightness factor for m=15 should be 0.05 + (1.95-0.05)*15/30 = 1.0
        # So image should be roughly unchanged if TF.adjust_brightness(img, 1.0)
        # This is hard to assert precisely without knowing TF's internals or mocking it.
        # We'll trust TF.adjust_brightness works and we're calling it.

    def test_contrast(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True)
        augmented_img = augmenter._contrast(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape

    def test_color_rgb(self):
        augmenter = RandAugment(grayscale=False) # Enable color
        img_tensor = create_dummy_tensor_image(size=(3,32,32), grayscale=False)
        augmented_img = augmenter._color(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape
    
    def test_color_grayscale_no_op(self):
        # Color op is not added if grayscale=True
        augmenter = RandAugment(grayscale=True)
        assert not any(op_name == "Color" for op_name, _ in augmenter.augmentations)

    def test_equalize_float(self):
        augmenter = RandAugment()
        img_tensor = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]], dtype=torch.float32)
        with patch('torchvision.transforms.functional.equalize') as mock_tf_equalize:
            mock_tf_equalize.return_value = (img_tensor * 255).to(torch.uint8)
            augmented_img = augmenter._equalize(img_tensor.clone(), 0)
            mock_tf_equalize.assert_called_once()
            assert augmented_img.dtype == torch.float32

    def test_equalize_uint8(self):
        augmenter = RandAugment()
        img_tensor = (torch.rand(1, 32, 32) * 255).to(torch.uint8)
        with patch('torchvision.transforms.functional.equalize') as mock_tf_equalize:
            mock_tf_equalize.return_value = img_tensor
            augmented_img = augmenter._equalize(img_tensor.clone(), 0)
            mock_tf_equalize.assert_called_once()
            assert augmented_img.dtype == torch.uint8

    def test_posterize_float(self):
        augmenter = RandAugment()
        img_tensor = torch.tensor([[[0.1, 0.9], [0.2, 0.8]]], dtype=torch.float32)
        # m=9, bits = int(8 + (4-8)*9/30) = int(8 - 4*0.3) = int(8 - 1.2) = int(6.8) = 6
        with patch('torchvision.transforms.functional.posterize') as mock_tf_posterize:
            mock_tf_posterize.return_value = (img_tensor * 255).to(torch.uint8)
            augmented_img = augmenter._posterize(img_tensor.clone(), 9)
            mock_tf_posterize.assert_called_once()
            called_args, _ = mock_tf_posterize.call_args
            assert torch.equal(called_args[0], (img_tensor * 255).to(torch.uint8))
            assert called_args[1] == 6
            assert augmented_img.dtype == torch.float32

    def test_posterize_uint8(self):
        augmenter = RandAugment()
        img_tensor = (torch.rand(1, 32, 32) * 255).to(torch.uint8)
        with patch('torchvision.transforms.functional.posterize') as mock_tf_posterize:
            mock_tf_posterize.return_value = img_tensor
            augmented_img = augmenter._posterize(img_tensor.clone(), 9) # bits = 6
            mock_tf_posterize.assert_called_once()
            called_args, _ = mock_tf_posterize.call_args
            # img_tensor is cloned before being passed to _posterize.
            # The mocked TF.posterize is called with this clone.
            # We compare its content against the original img_tensor.
            assert torch.equal(called_args[0], img_tensor)
            assert called_args[1] == 6
            assert augmented_img.dtype == torch.uint8

    def test_rotate_tensor_float(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True, dtype=torch.float32)
        # m=15, angle = -30 + (30 - (-30)) * 15/30 = -30 + 60 * 0.5 = 0
        augmented_img = augmenter._rotate(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape
        # TF.rotate should be called with angle=0, fill=1.0

    def test_rotate_tensor_uint8(self):
        augmenter = RandAugment()
        img_tensor = (create_dummy_tensor_image(grayscale=True) * 255).to(torch.uint8)
        augmented_img = augmenter._rotate(img_tensor.clone(), 15) # angle=0
        assert augmented_img.shape == img_tensor.shape
        # TF.rotate should be called with angle=0, fill=255

    def test_sharpness(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True)
        augmented_img = augmenter._sharpness(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape

    def test_shear_x_tensor_float(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True, dtype=torch.float32)
        # m=15, shear = -0.3 + (0.3 - (-0.3)) * 15/30 = -0.3 + 0.6 * 0.5 = 0
        augmented_img = augmenter._shear_x(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape
        # TF.affine should be called with shear=(0,0), fill=1.0

    def test_shear_y_tensor_uint8(self):
        augmenter = RandAugment()
        img_tensor = (create_dummy_tensor_image(grayscale=True) * 255).to(torch.uint8)
        augmented_img = augmenter._shear_y(img_tensor.clone(), 15) # shear=0
        assert augmented_img.shape == img_tensor.shape
        # TF.affine should be called with shear=(0,0), fill=255

    def test_solarize_tensor(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(grayscale=True, dtype=torch.float32)
        # m=15, threshold = (256 + (0-256)*15/30) / 256 = (256 - 128) / 256 = 128/256 = 0.5
        augmented_img = augmenter._solarize(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape
        # TF.solarize should be called with threshold=0.5

    def test_translate_x_tensor_float(self):
        augmenter = RandAugment()
        img_tensor = create_dummy_tensor_image(size=(1, 32, 32), grayscale=True, dtype=torch.float32)
        # m=15, translation = -0.45 + (0.45 - (-0.45)) * 15/30 = 0
        # pixels = int(0 * 32) = 0
        augmented_img = augmenter._translate_x(img_tensor.clone(), 15)
        assert augmented_img.shape == img_tensor.shape
        # TF.affine should be called with translate=(0,0), fill=1.0

    def test_translate_y_tensor_uint8(self):
        augmenter = RandAugment()
        img_tensor = (create_dummy_tensor_image(size=(1,32,32), grayscale=True) * 255).to(torch.uint8)
        augmented_img = augmenter._translate_y(img_tensor.clone(), 15) # pixels=0
        assert augmented_img.shape == img_tensor.shape
        # TF.affine should be called with translate=(0,0), fill=255

# Tests for QualityAwarePatchAugment
class TestQualityAwarePatchAugment:
    def test_initialization(self):
        augmenter = QualityAwarePatchAugment(patch_size=8, quality_threshold=0.6, strong_aug_prob=0.7, patch_drop_prob=0.2)
        assert augmenter.patch_size == 8
        assert augmenter.quality_threshold == 0.6
        assert augmenter.strong_aug_prob == 0.7
        assert augmenter.patch_drop_prob == 0.2

    def test_compute_patch_quality_3d_input(self):
        augmenter = QualityAwarePatchAugment(patch_size=16)
        img_tensor = create_dummy_tensor_image(size=(1, 64, 64), grayscale=True) # C, H, W
        quality_map = augmenter.compute_patch_quality(img_tensor)
        assert quality_map.shape == (1, 64 // 16, 64 // 16) # B, pH, pW (B is squeezed out if C=1)
        assert quality_map.ndim == 3 # B, pH, pW (after unsqueeze(0) in compute_patch_quality)
        # After squeeze(1) for grayscale, it should be (B, pH, pW)
        # The method itself unsqueezes to 4D then squeezes channel for grayscale
        # So if input is (1,H,W), output is (1, H/p, W/p)
        img_tensor_3d = create_dummy_tensor_image(size=(1, 64, 64), grayscale=True) # C, H, W
        quality_map_3d = augmenter.compute_patch_quality(img_tensor_3d)
        assert quality_map_3d.shape == (1, 4, 4)


    def test_compute_patch_quality_4d_input(self):
        augmenter = QualityAwarePatchAugment(patch_size=16)
        img_tensor_4d = create_dummy_tensor_image(size=(1, 64, 64), grayscale=True, batch_size=2) # B, C, H, W
        quality_map_4d = augmenter.compute_patch_quality(img_tensor_4d)
        assert quality_map_4d.shape == (2, 4, 4) # B, pH, pW

    @patch('random.random')
    @patch('random.choice')
    def test_forward_low_quality_strong_aug_noise(self, mock_choice, mock_random):
        mock_random.side_effect = [
            0.7, 0.2,  # Patch 1: strong aug, no drop
            0.7, 0.2,  # Patch 2: strong aug, no drop
            0.7, 0.2,  # Patch 3: strong aug, no drop
            0.7, 0.2   # Patch 4: strong aug, no drop
        ] # For 4 patches (2x2), 2 random.random() calls each
        mock_choice.return_value = 'noise'
        
        augmenter = QualityAwarePatchAugment(patch_size=16, quality_threshold=0.9, strong_aug_prob=0.8)
        img_tensor = torch.ones(1, 1, 32, 32) * 0.1 # Low quality (mean 0.1)
        
        # Mock compute_patch_quality to return a very low quality
        with patch.object(augmenter, 'compute_patch_quality', return_value=torch.zeros(1, 2, 2)):
            augmented_img = augmenter(img_tensor.clone())
        
        assert augmented_img.shape == img_tensor.shape
        # Check if noise was added (img should not be all 0.1 anymore)
        assert not torch.allclose(augmented_img, img_tensor)
        mock_choice.assert_called() # strong aug type chosen

    @patch('random.random')
    @patch('random.choice')
    def test_forward_low_quality_patch_drop(self, mock_choice, mock_random):
        # random.random() calls:
        # 1. For strong_aug_prob (0.7 < 0.8 -> True)
        # 2. For patch_drop_prob (0.05 < 0.1 -> True, assuming patch_quality < 0.3)
        mock_random.side_effect = [
            0.7, 0.05, # Patch 1: strong aug, drop
            0.7, 0.2,  # Patch 2: strong aug, no drop
            0.7, 0.2,  # Patch 3: strong aug, no drop
            0.7, 0.2   # Patch 4: strong aug, no drop
        ] # For 4 patches (2x2), 2 random.random() calls each. First patch drops.
        mock_choice.return_value = 'noise' # Doesn't matter much here
        
        augmenter = QualityAwarePatchAugment(patch_size=16, quality_threshold=0.9, strong_aug_prob=0.8, patch_drop_prob=0.1)
        img_tensor = torch.ones(1, 1, 32, 32) * 0.1 # Low quality
        
        # Mock compute_patch_quality to return a very low quality (e.g. 0.2, which is < 0.3 for drop)
        with patch.object(augmenter, 'compute_patch_quality', return_value=torch.ones(1, 2, 2) * 0.2):
            augmented_img = augmenter(img_tensor.clone())
        
        assert augmented_img.shape == img_tensor.shape
        # At least one patch should be zeroed out
        assert torch.any(augmented_img == 0) 
        # Not all should be zero if only one patch was dropped (depends on random calls per patch)
        # This test is a bit tricky due to per-patch randomness.
        # For simplicity, we check if *any* part became zero.

    @patch('random.random')
    @patch('random.choice')
    def test_forward_high_quality_light_aug(self, mock_choice, mock_random):
        mock_random.return_value = 0.2 # For 30% chance of light aug
        mock_choice.return_value = 'slight_noise'
        
        augmenter = QualityAwarePatchAugment(patch_size=16, quality_threshold=0.1)
        img_tensor = torch.ones(1, 1, 32, 32) * 0.8 # High quality
        
        # Mock compute_patch_quality to return high quality
        with patch.object(augmenter, 'compute_patch_quality', return_value=torch.ones(1, 2, 2)):
            augmented_img = augmenter(img_tensor.clone())
            
        assert augmented_img.shape == img_tensor.shape
        assert not torch.allclose(augmented_img, img_tensor) # Slight noise applied
        mock_choice.assert_called_with(['slight_noise', 'slight_brightness'])


# Tests for create_vit_transform
class TestCreateVitTransform:
    def test_training_pipeline_basic(self):
        transform = create_vit_transform(img_size=64, is_training=True, pretrained=False, use_quality_aware=False, randaugment_n=0)
        assert isinstance(transform, T.Compose)
        # Expected: Resize, RandomHFlip, RandomVFlip, ToTensor, Normalize (grayscale)
        assert len(transform.transforms) == 5
        assert isinstance(transform.transforms[0], T.Resize)
        assert isinstance(transform.transforms[1], T.RandomHorizontalFlip)
        assert isinstance(transform.transforms[2], T.RandomVerticalFlip)
        assert isinstance(transform.transforms[3], T.ToTensor)
        assert isinstance(transform.transforms[4], T.Normalize)
        assert transform.transforms[4].mean == [0.5]

    def test_training_pipeline_with_randaugment_qualityaware(self):
        transform = create_vit_transform(img_size=64, is_training=True, pretrained=False, use_quality_aware=True, randaugment_n=2, patch_size=8)
        assert isinstance(transform, T.Compose)
        # Expected: Resize, RandomHFlip, RandomVFlip, RandAugment, ToTensor, QualityAware, Normalize
        assert len(transform.transforms) == 7
        assert isinstance(transform.transforms[3], RandAugment)
        assert isinstance(transform.transforms[5], QualityAwarePatchAugment)
        assert transform.transforms[5].patch_size == 8

    def test_validation_pipeline(self):
        transform = create_vit_transform(img_size=64, is_training=False, pretrained=False)
        # Expected: Resize, ToTensor, Normalize
        assert len(transform.transforms) == 3
        assert isinstance(transform.transforms[0], T.Resize)
        assert isinstance(transform.transforms[1], T.ToTensor)
        assert isinstance(transform.transforms[2], T.Normalize)

    def test_pretrained_imagenet_training(self):
        transform = create_vit_transform(img_size=224, is_training=True, pretrained=True, pretrained_type='imagenet', use_quality_aware=False, randaugment_n=0)
        # Expected: Resize(224), RandomHFlip, RandomVFlip, ToTensor, Lambda (repeat channels), Normalize (ImageNet)
        assert len(transform.transforms) == 6 
        assert transform.transforms[0].size == (224, 224)
        assert isinstance(transform.transforms[4], T.Lambda)
        assert transform.transforms[5].mean == [0.485, 0.456, 0.406]

    def test_input_size_override(self):
        transform = create_vit_transform(img_size=256, input_size_override=128, is_training=False, pretrained=False)
        assert transform.transforms[0].size == (128, 128)

    def test_transform_on_pil_image_grayscale(self):
        pil_img = create_dummy_pil_image(size=(64,64), mode='L')
        transform = create_vit_transform(img_size=32, is_training=False, pretrained=False)
        output_tensor = transform(pil_img)
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (1, 32, 32) # C, H, W
        assert output_tensor.dtype == torch.float32

    def test_transform_on_pil_image_pretrained_rgb_output(self):
        pil_img = create_dummy_pil_image(size=(256,256), mode='L') # Grayscale input
        transform = create_vit_transform(img_size=224, is_training=False, pretrained=True, pretrained_type='imagenet')
        output_tensor = transform(pil_img)
        assert isinstance(output_tensor, torch.Tensor)
        assert output_tensor.shape == (3, 224, 224) # Should be 3 channels for ImageNet
        assert output_tensor.dtype == torch.float32


# Tests for MixUp
class TestMixUp:
    def test_initialization(self):
        mixer = MixUp(alpha=0.5)
        assert mixer.alpha == 0.5

    def test_call_with_alpha_gt_0(self):
        mixer = MixUp(alpha=0.8)
        images = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        mixed_images, labels_a, labels_b, lam = mixer(images.clone(), labels.clone())
        
        assert mixed_images.shape == images.shape
        assert labels_a.shape == labels.shape
        assert labels_b.shape == labels.shape
        assert 0 < lam < 1
        # Check if images are mixed (not identical to original or permuted)
        # This is probabilistic, but with alpha=0.8, lam is unlikely to be 0 or 1
        assert not torch.allclose(mixed_images, images) 

    def test_call_with_alpha_0(self):
        mixer = MixUp(alpha=0.0)
        images = torch.rand(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        mixed_images, labels_a, labels_b, lam = mixer(images.clone(), labels.clone())
        
        assert lam == 1
        assert torch.allclose(mixed_images, images) # Should be original images
        assert torch.allclose(labels_a, labels)


# Tests for CutMix
class TestCutMix:
    def test_initialization(self):
        mixer = CutMix(alpha=0.7)
        assert mixer.alpha == 0.7

    def test_rand_bbox(self):
        mixer = CutMix(alpha=1.0)
        shape = (4, 3, 32, 32) # B, C, H, W
        lam = 0.5 # cut_rat = sqrt(0.5) approx 0.707
                  # cut_w = 32 * 0.707 = 22
                  # cut_h = 32 * 0.707 = 22
        
        for _ in range(10): # Run a few times due to randomness
            bbx1, bby1, bbx2, bby2 = mixer._rand_bbox(shape, lam)
            assert 0 <= bbx1 < bbx2 <= shape[3] # W
            assert 0 <= bby1 < bby2 <= shape[2] # H
            # Check if cut area is reasonable, though exact match is hard with int conversions
            # expected_area_ratio = 1 - lam = 0.5
            # actual_area_ratio = ((bbx2 - bbx1) * (bby2 - bby1)) / (shape[2] * shape[3])

    def test_call_with_alpha_gt_0(self):
        torch.manual_seed(42)
        np.random.seed(42)
        mixer = CutMix(alpha=1.0)
        images = torch.rand(4, 3, 32, 32)
        original_images_clone = images.clone()
        labels = torch.randint(0, 10, (4,))
        
        mixed_images, labels_a, labels_b, lam = mixer(images, labels.clone()) # images is modified in-place
        
        assert mixed_images.shape == original_images_clone.shape
        assert labels_a.shape == labels.shape
        assert labels_b.shape == labels.shape
        assert 0 < lam < 1 # Adjusted lambda
        
        # Check that some part of the image has changed
        assert not torch.allclose(mixed_images, original_images_clone)
        # More specific check: find the bbox and verify content from another image
        # This is harder without knowing the random bbox and index used.

    def test_call_with_alpha_0(self):
        mixer = CutMix(alpha=0.0)
        images = torch.rand(4, 3, 32, 32)
        original_images_clone = images.clone()
        labels = torch.randint(0, 10, (4,))
        
        mixed_images, labels_a, labels_b, lam = mixer(images, labels.clone())
        
        assert lam == 1 # Because _rand_bbox with lam=1 should result in full area or no cut
                       # If lam=1 from beta dist, cut_rat = 0, so cut_w, cut_h = 0.
                       # Then bbx1=bbx2, bby1=bby2.
                       # Adjusted lam = 1 - (0 * 0) / area = 1.
        assert torch.allclose(mixed_images, original_images_clone) # Should be original images
        assert torch.allclose(labels_a, labels)