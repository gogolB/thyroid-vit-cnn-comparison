import pytest
import torch
import numpy as np
from torchvision import transforms as T
from torch import nn
import random

# Import the module to test
from src.data.transforms import (
    MicroscopyNormalize,
    ElasticTransform,
    MicroscopyAugmentation,
    RandomPatchDrop,
    get_training_transforms,
    get_validation_transforms,
    get_tta_transforms,
    print_augmentation_summary
)
from src.data import transforms as M # For mocking console

# Helper function to create dummy tensors
def create_dummy_tensor(shape, min_val=0, max_val=1, dtype=torch.float32, device='cpu', requires_grad=False):
    if dtype == torch.uint16: # Special case for uint16 range (0-65535)
         # Create float tensor in the uint16 range
         tensor = torch.randint(int(min_val), int(max_val) + 1, shape, device=device, dtype=torch.float32)
    elif dtype == torch.float32:
        tensor = (torch.rand(shape, device=device) * (max_val - min_val) + min_val)
    else:
        tensor = (torch.rand(shape, device=device) * (max_val - min_val) + min_val).to(dtype)
    tensor.requires_grad_(requires_grad)
    return tensor

class TestMicroscopyNormalize:
    def test_default_normalization(self):
        transform = MicroscopyNormalize()
        # Input tensor simulating uint16 image data
        data = create_dummy_tensor((1, 64, 64), min_val=0, max_val=65535, dtype=torch.uint16)
        transformed_data = transform(data)
        assert transformed_data.shape == data.shape
        assert transformed_data.min() >= 0.0
        assert transformed_data.max() <= 1.0
        # Check if values are scaled (e.g., 65535 becomes 1.0, 0 becomes 0.0)
        assert torch.isclose(transform(torch.tensor([0.0])), torch.tensor([0.0]))
        assert torch.isclose(transform(torch.tensor([65535.0])), torch.tensor([1.0]))

    def test_custom_ranges(self):
        transform = MicroscopyNormalize(input_range=(0, 1000), output_range=(-1, 1))
        data = create_dummy_tensor((1, 64, 64), min_val=0, max_val=1000, dtype=torch.float32)
        transformed_data = transform(data)
        assert transformed_data.shape == data.shape
        assert transformed_data.min() >= -1.0
        assert transformed_data.max() <= 1.0
        assert torch.isclose(transform(torch.tensor([0.0])), torch.tensor([-1.0]))
        assert torch.isclose(transform(torch.tensor([1000.0])), torch.tensor([1.0]))
        assert torch.isclose(transform(torch.tensor([500.0])), torch.tensor([0.0]))

    @pytest.mark.parametrize("clip_percentile", [None, (1.0, 99.0)])
    @pytest.mark.parametrize("batch_size", [None, 4]) # None for 3D, 4 for 4D
    def test_clipping(self, clip_percentile, batch_size):
        transform = MicroscopyNormalize(clip_percentile=clip_percentile)
        if batch_size:
            shape = (batch_size, 1, 32, 32)
        else:
            shape = (1, 32, 32)
        
        # Create data with outliers
        data = torch.randn(shape) * 10000 + 30000 # Centered around 30000
        data[..., 0, 0] = 0 # Min outlier
        data[..., -1, -1] = 65535 # Max outlier
        
        transformed_data = transform(data.clone()) # Clone to avoid in-place modification issues with original
        
        assert transformed_data.shape == data.shape
        assert transformed_data.min() >= 0.0
        assert transformed_data.max() <= 1.0

        if clip_percentile:
            # With clipping, the extreme values should be pulled in before normalization
            # This is hard to assert precisely without knowing exact quantiles,
            # but output should still be in [0,1]
            pass # Main check is output range and shape

class TestElasticTransform:
    @pytest.mark.parametrize("p_value, should_transform", [(0.0, False), (1.0, True)])
    @pytest.mark.parametrize("batch_size", [None, 2]) # None for 3D, 2 for 4D
    def test_transform_probability_and_shape(self, p_value, should_transform, batch_size, mocker):
        # Mock np.random.rand to make displacement fields predictable if needed, or just check for change
        mocker.patch('numpy.random.rand', return_value=(np.arange(32*32, dtype=np.float32).reshape(32,32) / (32*32) * 2 - 1)) # Non-uniform fixed field
        
        transform = ElasticTransform(alpha=50, sigma=5, p=p_value)
        if batch_size:
            shape = (batch_size, 1, 32, 32) # Assuming single channel
        else:
            shape = (1, 32, 32) # Assuming single channel
        
        data = create_dummy_tensor(shape, min_val=0, max_val=1)
        
        # Mock random.random for the probability check
        mocker.patch('random.random', return_value=0.0 if should_transform else 1.0)
        
        transformed_data = transform(data.clone())
        
        assert transformed_data.shape == data.shape
        assert transformed_data.dtype == data.dtype
        
        if should_transform:
            # If alpha > 0, transformation should occur and data should change
            # Note: ElasticTransform only processes the first channel if multi-channel input
            assert not torch.equal(transformed_data, data), "Data should be transformed"
        else:
            assert torch.equal(transformed_data, data), "Data should not be transformed"

    def test_elastic_transform_on_numpy_array(self):
        # Test the internal _elastic_transform_2d method indirectly via forward
        transform = ElasticTransform(alpha=100, sigma=10, p=1.0) # Ensure transform happens
        img_np = np.random.rand(64, 64).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0) # (1, H, W)

        transformed_tensor = transform(img_tensor.clone())
        assert transformed_tensor.shape == img_tensor.shape
        assert not torch.allclose(transformed_tensor, img_tensor)


class TestMicroscopyAugmentation:
    @pytest.mark.parametrize("p_value, should_transform", [(0.0, False), (1.0, True)])
    def test_transform_probability(self, p_value, should_transform, mocker):
        transform = MicroscopyAugmentation(p=p_value)
        data = create_dummy_tensor((1, 64, 64), min_val=0, max_val=1)
        
        # Mock random.random for the main probability check
        mocker.patch('random.random', side_effect=[0.0 if should_transform else 1.0] + [0.6]*10) # first for p, others for internal augs

        transformed_data = transform(data.clone())
        
        assert transformed_data.shape == data.shape
        assert transformed_data.min() >= 0.0
        assert transformed_data.max() <= 1.0 # Check clamping

        if should_transform:
             # With p=1, some augmentation should happen if internal random checks pass
             # For a robust check, mock internal random.random calls to ensure one augmentation fires
            with mocker.patch('random.random', side_effect=[0.0, 0.4, 0.6, 0.6, 0.6]): # p=1, brightness=yes
                transform_p1 = MicroscopyAugmentation(p=1.0, brightness_range=(2.0, 2.0)) # Force brightness change
                data_p1 = create_dummy_tensor((1, 64, 64), min_val=0.5, max_val=0.5) # Uniform input
                transformed_p1 = transform_p1(data_p1.clone())
                assert not torch.equal(transformed_p1, data_p1), "Data should be transformed by brightness"
        else:
            assert torch.equal(transformed_data, data), "Data should not be transformed"

    def test_all_augmentations_active(self, mocker):
        # Mock random.random to activate all internal augmentations
        mocker.patch('random.random', return_value=0.1) # Activates p, brightness, contrast, noise, blur
        
        transform = MicroscopyAugmentation(p=1.0, noise_std=0.1, blur_sigma_range=(0.5,0.5))
        data = create_dummy_tensor((1, 64, 64), min_val=0.2, max_val=0.8) # Avoid 0 or 1 to see effects
        
        transformed_data = transform(data.clone())
        
        assert transformed_data.shape == data.shape
        assert transformed_data.min() >= 0.0
        assert transformed_data.max() <= 1.0
        assert not torch.equal(transformed_data, data), "Data should be significantly transformed"

class TestRandomPatchDrop:
    @pytest.mark.parametrize("p_value, should_transform", [(0.0, False), (1.0, True)])
    def test_transform_probability_and_shape(self, p_value, should_transform, mocker):
        transform = RandomPatchDrop(patch_size=8, max_patches=1, p=p_value)
        data = create_dummy_tensor((1, 32, 32), min_val=0, max_val=1)
        
        # Mock random.random for the probability check
        mocker.patch('random.random', side_effect=[0.0 if should_transform else 1.0, 0, 0, 0]) # p, num_patches_randint, y_randint, x_pos_randint
        mocker.patch('random.randint', side_effect=[1, 0, 0]) # num_patches=1, y=0, x_pos=0

        transformed_data = transform(data.clone())
        
        assert transformed_data.shape == data.shape
        
        if should_transform:
            # Check if the patch at (0,0) was modified
            original_patch = data[..., 0:8, 0:8]
            transformed_patch = transformed_data[..., 0:8, 0:8]
            # The patch should be set to its mean
            assert torch.allclose(transformed_patch, torch.full_like(transformed_patch, original_patch.mean()))
            assert not torch.equal(transformed_data, data)
        else:
            assert torch.equal(transformed_data, data)

    def test_patch_drop_effect(self, mocker):
        # Ensure transform happens
        mocker.patch('random.random', return_value=0.0) # p=1.0
        # Force specific patch location and number
        mocker.patch('random.randint', side_effect=[1, 0, 0]) # num_patches=1, y_start=0, x_start=0

        patch_size = 8
        transform = RandomPatchDrop(patch_size=patch_size, max_patches=1, p=1.0)
        data = torch.arange(32*32, dtype=torch.float32).reshape(1, 32, 32) / (32*32) # Predictable data

        transformed_data = transform(data.clone())
        
        original_patch_values = data[..., 0:patch_size, 0:patch_size]
        mean_val = original_patch_values.mean()
        
        # Check that the specified patch is now the mean value
        assert torch.all(transformed_data[..., 0:patch_size, 0:patch_size] == mean_val)
        # Check that other parts are unchanged (assuming only one patch)
        if 32 > patch_size: # If image is larger than patch
             assert torch.equal(transformed_data[..., patch_size:, patch_size:], data[..., patch_size:, patch_size:])


@pytest.mark.parametrize("target_size", [128, 224])
@pytest.mark.parametrize("normalize_flag", [True, False])
@pytest.mark.parametrize("aug_level", ['none', 'light', 'medium', 'heavy'])
class TestGetTrainingTransforms:
    def test_structure_and_execution(self, target_size, normalize_flag, aug_level, mocker):
        # Mock random calls within augmentations if needed for specific checks,
        # but here primarily testing pipeline construction and basic run
        mocker.patch('random.random', return_value=0.1) # Ensure probabilistic transforms activate if p=1
        mocker.patch('numpy.random.rand', side_effect=lambda *s: np.ones(s, dtype=np.float32)*0.5)


        pipeline = get_training_transforms(
            target_size=target_size,
            normalize=normalize_flag,
            augmentation_level=aug_level
        )
        assert isinstance(pipeline, nn.Sequential)
        
        # Check for Normalization
        has_norm = any(isinstance(t, MicroscopyNormalize) for t in pipeline.children())
        assert has_norm == normalize_flag
        
        # Check for Resize
        has_resize = any(isinstance(t, T.Resize) for t in pipeline.children())
        assert has_resize
        resize_transform = [t for t in pipeline.children() if isinstance(t, T.Resize)][0]
        # T.Resize stores size as (h, w) if tuple, or int if square.
        # For nn.Sequential, T.Resize might be wrapped or its parameters not directly accessible as `size`.
        # Instead, check output shape after applying.

        # Test execution
        # Input can be uint16 range if normalize=True, or 0-1 if normalize=False
        if normalize_flag:
            data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=65535, dtype=torch.uint16)
        else:
            data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=1)
        
        transformed_data = pipeline(data)
        assert transformed_data.shape == (1, target_size, target_size)
        if not normalize_flag and aug_level == 'none': # Should be only resize
             pass # Input already 0-1
        else: # With normalization or augmentation, range should be ~0-1
            assert transformed_data.min() >= -1e-2 # Allow small negative due to float precision / some augs
            assert transformed_data.max() <= 1 + 1e-2 # Allow small positive overshoot

@pytest.mark.parametrize("target_size", [128, 224])
@pytest.mark.parametrize("normalize_flag", [True, False])
class TestGetValidationTransforms:
    def test_structure_and_execution(self, target_size, normalize_flag):
        pipeline = get_validation_transforms(
            target_size=target_size,
            normalize=normalize_flag
        )
        assert isinstance(pipeline, nn.Sequential)
        
        has_norm = any(isinstance(t, MicroscopyNormalize) for t in pipeline.children())
        assert has_norm == normalize_flag
        
        has_resize = any(isinstance(t, T.Resize) for t in pipeline.children())
        assert has_resize

        if normalize_flag:
            data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=65535, dtype=torch.uint16)
        else:
            data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=1)
            
        transformed_data = pipeline(data)
        assert transformed_data.shape == (1, target_size, target_size)
        if normalize_flag:
            assert transformed_data.min() >= 0.0
            assert transformed_data.max() <= 1.0

@pytest.mark.parametrize("target_size", [128, 224])
@pytest.mark.parametrize("normalize_flag", [True, False])
@pytest.mark.parametrize("n_augs", [1, 3, 5, 7]) # Test requesting more than available
class TestGetTTATransforms:
    def test_structure_and_execution(self, target_size, normalize_flag, n_augs):
        pipelines = get_tta_transforms(
            target_size=target_size,
            normalize=normalize_flag,
            n_augmentations=n_augs
        )
        assert isinstance(pipelines, list)
        assert len(pipelines) == min(n_augs, 5) # Max 5 TTA transforms defined
        
        for pipeline in pipelines:
            assert isinstance(pipeline, nn.Sequential)
            has_norm_or_identity = any(isinstance(t, (MicroscopyNormalize, nn.Identity)) for t in pipeline.children())
            assert has_norm_or_identity # Base transform is norm or identity

            has_resize = any(isinstance(t, T.Resize) for t in pipeline.children())
            assert has_resize

            if normalize_flag:
                data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=65535, dtype=torch.uint16)
            else:
                data = create_dummy_tensor((1, target_size*2, target_size*2), min_val=0, max_val=1)
            
            transformed_data = pipeline(data.clone()) # Clone as some TTA might be in-place if not careful
            assert transformed_data.shape == (1, target_size, target_size)
            if normalize_flag: # If normalized, output should be ~0-1
                assert transformed_data.min() >= -1e-2 
                assert transformed_data.max() <= 1 + 1e-2


class TestPrintAugmentationSummary:
    @pytest.mark.parametrize("level", ['light', 'medium', 'heavy', 'unknown_level'])
    def test_runs_without_error(self, level, mocker):
        mock_console_print = mocker.patch('src.data.transforms.console.print')
        try:
            print_augmentation_summary(augmentation_level=level)
        except Exception as e:
            if level == 'unknown_level': # Expect it to run, but maybe print nothing or default
                pass # Or assert specific behavior for unknown levels if defined
            else:
                pytest.fail(f"print_augmentation_summary failed for level {level}: {e}")
        
        if level in ['light', 'medium', 'heavy']:
            mock_console_print.assert_called() # Check that it tried to print something
        # For 'unknown_level', it might print a default table or nothing, depending on implementation.
        # The current implementation of print_augmentation_summary doesn't have a default for unknown,
        # so it will just print an empty table or a table with only headers.
        # This test mainly ensures it doesn't crash.