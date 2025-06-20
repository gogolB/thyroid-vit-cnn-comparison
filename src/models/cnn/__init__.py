from src.models.registry import ModelRegistry

# Attempt to import and register CNN model classes.
# These imports assume the model classes (e.g., ResNet, EfficientNet, DenseNet, InceptionV3)
# are defined in their respective files and are designed to be registered.

try:
    from .resnet import ResNet
    # Register common ResNet variants
    ModelRegistry.register(['resnet18', 'resnet34', 'resnet50', 'resnet101'], 'cnn')(ResNet)
except ImportError:
    print("Warning: ResNet not found or could not be imported for registration in src.models.cnn.")

try:
    from .efficientnet import EfficientNet
    # Register common EfficientNet variants
    ModelRegistry.register(['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'], 'cnn')(EfficientNet)
except ImportError:
    print("Warning: EfficientNet not found or could not be imported for registration in src.models.cnn.")

try:
    from .densenet import DenseNet
    # Register common DenseNet variants
    ModelRegistry.register(['densenet121', 'densenet169', 'densenet201'], 'cnn')(DenseNet)
except ImportError:
    print("Warning: DenseNet not found or could not be imported for registration in src.models.cnn.")

try:
    from .inception import Inception # Correctly import the Inception class
    # The Inception class uses a decorator for registration, so no explicit call here.
except ImportError:
    print("Warning: Inception model (which includes v3 and v4) not found or could not be imported for registration in src.models.cnn.")

# Add other CNN models as needed following the same pattern.