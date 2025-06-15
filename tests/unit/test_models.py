import pytest
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
    def test_forward_pass(self, model_name, sample_batch): # sample_batch from conftest.py
        """Test forward pass of the model."""
        arch = 'cnn' if model_name in REGISTERED_MODEL_NAMES_CNN else 'vit'
        config_data = {'name': model_name, 'architecture': arch, 'num_classes': 2, 'pretrained': False}
        if arch == 'vit':
            config_data['img_size'] = 224 
            if 'patch' in model_name:
                 try:
                    config_data['patch_size'] = int(model_name.split('_patch')[1].split('_')[0])
                 except:
                    config_data['patch_size'] = 16

        config = SimpleTestConfig(**config_data)
        model = ModelRegistry.create_model(config)
        
        images, _ = sample_batch # Get images from the fixture
        
        # Ensure model is in eval mode for stable behavior if it has dropout/batchnorm
        model.eval() 
        
        try:
            output = model(images) # Calls ModelBase.forward -> self.model.forward
        except Exception as e:
            pytest.fail(f"Forward pass for model {model_name} failed with error: {e}")
            
        assert output is not None, f"Forward pass for model {model_name} returned None."
        # Expected output shape: (batch_size, num_classes)
        # sample_batch has batch_size 4, num_classes is 2 by default in SimpleTestConfig
        expected_shape = (4, config.num_classes)
        assert output.shape == expected_shape, \
            f"Output shape for model {model_name} is {output.shape}, expected {expected_shape}."