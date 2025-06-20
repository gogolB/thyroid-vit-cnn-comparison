import pytest
import torch # Add missing import
from src.models.registry import ModelRegistry
# Assuming ModelBase and specific model classes (ResNet, VisionTransformer etc.) are importable
# and correctly registered by the time these tests run.
# For ModelRegistry.create_model, the config needs 'name' and other params expected by the model's __init__ / _build_model.

# A simple config structure for testing model creation.
# Real configs will come from YAML files loaded via Hydra/OmegaConf in actual runs.
class SimpleTestConfig:
    def __init__(self, name, architecture, num_classes=2, pretrained=False, **kwargs):
        self.name = name
        self.architecture = architecture # Added architecture as it's used by ModelBase
        self.num_classes = num_classes
        self.pretrained = pretrained # Most _build_model methods use this
        # Allow additional arbitrary keyword arguments to be set as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        """Provide a .get method similar to OmegaConf DictConfig for compatibility."""
        return getattr(self, key, default)

# Fixture to provide a list of all registered model names.
# This relies on ModelRegistry.list_models() being functional and models being registered
# when tests are collected. This might require models to be imported somewhere globally
# before pytest collection, e.g., by importing src.models or src.models.vit in conftest.py or a top-level __init__.
# For now, we'll define a placeholder list and adjust if ModelRegistry.list_models() is problematic at test time.
REGISTERED_MODEL_NAMES_CNN = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
    'inception_v3', # inception_v4 might need specific handling for aux_logits if not default
    'densenet121', # Add other densenets if registered
]
REGISTERED_MODEL_NAMES_VIT = [
    'vit_tiny', 'vit_small', 'vit_base',
    'deit_tiny', 'deit_small', 'deit_base',
    'swin_tiny', 'swin_small', 'swin_base',
]
ALL_REGISTERED_MODEL_NAMES = REGISTERED_MODEL_NAMES_CNN + REGISTERED_MODEL_NAMES_VIT

# To make ModelRegistry.list_models() work at test collection time,
# ensure model modules are imported. A simple way is to import them here or in conftest.
# This forces the @ModelRegistry.register decorators to run.
try:
    from src.models.cnn import resnet, efficientnet, inception, densenet
    from src.models.vit import vision_transformer, deit, swin 
    # Also ensure __init__.py in src/models/vit registers them
    import src.models.vit # This should trigger registrations in src/models/vit/__init__.py
except ImportError as e:
    print(f"Warning: Could not import all model modules for test_models.py: {e}")
    # Fallback to manually defined list if dynamic listing fails during test collection
    # ALL_REGISTERED_MODEL_NAMES = [...] # as defined above

class TestModels:
    # Use a predefined list for parametrization first, can switch to dynamic ModelRegistry.list_models() later
    # if imports and registration order at test time is reliable.
    @pytest.mark.parametrize("model_name", ALL_REGISTERED_MODEL_NAMES)
    def test_model_creation(self, model_name):
        """Test model can be created via ModelRegistry."""
        # Determine architecture based on name for simple config
        arch = 'cnn' if model_name in REGISTERED_MODEL_NAMES_CNN else 'vit'
        
        # Basic config for creation. Specific models might need more params (e.g., img_size for ViTs).
        # The _build_model methods use config.get('param', default_value), so this should be okay.
        config_data = {'name': model_name, 'architecture': arch, 'num_classes': 2, 'pretrained': False}
        if arch == 'vit': # ViTs often need img_size
            config_data['img_size'] = 224 # A common default
            if 'patch' in model_name: # e.g. vit_base_patch16_224
                 try:
                    config_data['patch_size'] = int(model_name.split('_patch')[1].split('_')[0])
                 except:
                    config_data['patch_size'] = 16 # fallback

        config = SimpleTestConfig(**config_data)
        
        model = ModelRegistry.create_model(config)
        assert model is not None, f"Failed to create model: {model_name}"
        assert hasattr(model, 'model'), f"Model {model_name} instance does not have a 'model' attribute after creation."
        assert model.model is not None, f"Inner 'model' attribute of {model_name} is None after creation."

    @pytest.mark.parametrize("model_name", ALL_REGISTERED_MODEL_NAMES)
    def test_forward_pass(self, model_name, synthetic_batch_256_1chan): # Use the renamed fixture
        """Test forward pass of the model."""
        arch = 'cnn' if model_name in REGISTERED_MODEL_NAMES_CNN else 'vit'
        config_data = {'name': model_name, 'architecture': arch, 'num_classes': 2, 'pretrained': False, 'extra_params': {}}
        current_img_size = 256 # Default
        in_chans = 1 # Default for CARS

        if arch == 'vit':
            current_img_size = 224
            config_data['img_size'] = current_img_size
            config_data['extra_params']['in_chans'] = 1 # ViTs should adapt to 1 channel via our wrapper
            in_chans = 1 # Input tensor will be 1 channel
            if 'patch' in model_name:
                try:
                    config_data['patch_size'] = int(model_name.split('_patch')[1].split('_')[0])
                except:
                    config_data['patch_size'] = 16
        elif model_name == 'inception_v3':
            current_img_size = 299
            in_chans = 1 # Test with 1 channel for CARS
            config_data['extra_params']['img_size'] = current_img_size
            config_data['extra_params']['in_chans'] = 1 # For InceptionV3, timm should adapt it
            in_chans = 1
        elif model_name.startswith('resnet'): # ResNets
            in_chans = 3 # ResNets from torchvision expect 3 channels
            config_data['extra_params']['in_chans'] = 3
            # current_img_size remains 256
        elif arch == 'cnn': # Other CNNs (EfficientNets, DenseNet)
            in_chans = 1 # Test with 1 channel for CARS compatibility
            config_data['extra_params']['in_chans'] = 1
            # current_img_size remains 256 for these

        config = SimpleTestConfig(**config_data)
        model = ModelRegistry.create_model(config)
        
        # Create images tensor dynamically based on current_img_size and in_chans
        # Batch size from sample_batch fixture (conftest.py) is 4
        batch_size = 4
        images = torch.randn(batch_size, in_chans, current_img_size, current_img_size)
        
        model.eval()
        
        try:
            output = model(images)
            if isinstance(output, tuple) and model_name == 'inception_v3': # Handle InceptionV3 tuple output in eval
                 output = output[0]
        except Exception as e:
            pytest.fail(f"Forward pass for model {model_name} (input shape {images.shape}) failed with error: {e}")
            
        assert output is not None, f"Forward pass for model {model_name} returned None."
        expected_shape = (batch_size, config.num_classes)
        assert output.shape == expected_shape, \
            f"Output shape for model {model_name} is {output.shape}, expected {expected_shape}."