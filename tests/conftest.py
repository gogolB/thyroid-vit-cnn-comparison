import pytest
from pathlib import Path
import torch
# import numpy as np # numpy was imported in the guide but not used in these specific fixtures

@pytest.fixture
def sample_image() -> torch.Tensor:
    """Generate sample CARS image (1 channel, 256x256)."""
    return torch.randn(1, 1, 256, 256) # Assuming CARS images are 1 channel (grayscale)

@pytest.fixture
def synthetic_batch_256_1chan() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a sample batch of 4 images (1x256x256) and corresponding labels."""
    images = torch.randn(4, 1, 256, 256) # Batch of 4, 1 channel, 256x256
    labels = torch.randint(0, 2, (4,)) # Batch of 4 labels, for 2 classes (0 or 1)
    return images, labels

@pytest.fixture
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic batch of 4 images (1x224x224) and labels for ViT tests."""
    images = torch.randn(4, 1, 224, 224) # Batch of 4, 1 channel, 224x224
    labels = torch.randint(0, 2, (4,))
    return images, labels

@pytest.fixture
def temp_dataset_path(tmp_path: Path) -> Path:
    """
    Create a temporary dataset directory structure for testing.
    tmp_path is a built-in pytest fixture providing a temporary directory.
    """
    # Create class subdirectories
    (tmp_path / "normal").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cancerous").mkdir(parents=True, exist_ok=True)
    
    # Optionally, create a few dummy image files if tests need to list/load them
    # For example:
    # (tmp_path / "normal" / "img1.png").touch()
    # (tmp_path / "normal" / "img2.png").touch()
    # (tmp_path / "cancerous" / "img3.png").touch()
    
    return tmp_path

# If you need a fixture that provides a dummy config object for tests:
# from omegaconf import OmegaConf # Assuming OmegaConf is used for configs

# @pytest.fixture
# def dummy_model_config():
#     # This is a very basic config. Adjust fields as necessary for your ModelBase/__init__
#     # and specific model tests.
#     conf = {
#         'name': 'test_model', # This will be overridden by specific model tests
#         'architecture': 'test_arch',
#         'pretrained': False,
#         'num_classes': 2,
#         # Add any other fields your ModelBase or specific models expect in their config
#         'img_size': 224, 
#         'patch_size': 16,
#     }
#     return OmegaConf.create(conf)
