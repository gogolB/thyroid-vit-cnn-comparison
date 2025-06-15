"""
Centralized model registry for all architectures.
Handles model creation, registration, and configuration.
"""
import logging # Optional: for logging registration/creation events

logger = logging.getLogger(__name__) # Optional

class ModelRegistry:
    """
    A central registry for model classes.
    Allows models to be registered with a name and type, and then instantiated
    via configuration.
    """
    _registry = {} # Class variable to store registered models: {'type': {'name': class}}

    @classmethod
    def register(cls, names, model_type='default'):
        """
        A decorator to register a model class.

        Args:
            names (str or list): The name(s) under which to register the model.
                                 If a list, the model is registered under all names.
            model_type (str): The type/category of the model (e.g., 'cnn', 'vit').
        """
        if not isinstance(names, list):
            names = [names]

        def decorator(model_class):
            if model_type not in cls._registry:
                cls._registry[model_type] = {}
            
            for name in names:
                if name in cls._registry[model_type]:
                    logger.warning(
                        f"Model {name} of type {model_type} is already registered. "
                        f"Overwriting with {model_class.__name__}."
                    )
                cls._registry[model_type][name] = model_class
                logger.info(f"Registered model: {name} (Type: {model_type}) -> {model_class.__name__}")
            return model_class
        return decorator

    @classmethod
    def create_model(cls, config):
        """
        Creates a model instance based on the provided configuration.

        The config object is expected to have at least a 'name' attribute
        and optionally a 'type' attribute (defaults to 'default' if not present,
        though our refactored models use 'cnn' or 'vit').
        The entire config object is passed to the model's constructor.

        Args:
            config (object): A configuration object (e.g., DictConfig from OmegaConf,
                             or a simple dictionary) containing model parameters.
                             Must include 'name' and optionally 'type'.

        Returns:
            An instance of the registered model class.

        Raises:
            ValueError: If the model name or type is not found in the registry,
                        or if config is missing the 'name' attribute.
        """
        if not hasattr(config, 'name'):
            raise ValueError("Configuration for model creation must include a 'name' attribute.")

        model_name = config.name
        # Infer model_type from config if present, otherwise try to find model_name across all types
        # For this project, model configs usually specify architecture which can map to type.
        # Let's assume for now the type is implicitly handled by how models were registered
        # (e.g. ResNet registered with type 'cnn').
        # A more robust way might be to require config.type or config.architecture.
        
        # Attempt to find the model by name across registered types
        found_model_class = None
        found_model_type = None

        for m_type, models_in_type in cls._registry.items():
            if model_name in models_in_type:
                found_model_class = models_in_type[model_name]
                found_model_type = m_type
                break
        
        if found_model_class:
            logger.info(f"Creating model: {model_name} (Type: {found_model_type}) using class {found_model_class.__name__}")
            # The model's __init__ should take the config object
            return found_model_class(config=config)
        else:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {cls.list_available_models()}"
            )

    @classmethod
    def list_models(cls, model_type=None):
        """
        Lists all registered model names, optionally filtered by type.
        (This method was used in the refactoring guide's test example).
        """
        if model_type:
            return list(cls._registry.get(model_type, {}).keys())
        else:
            all_models = []
            for models_in_type in cls._registry.values():
                all_models.extend(models_in_type.keys())
            return list(set(all_models)) # Use set to ensure uniqueness if a name spans types

    @classmethod
    def list_available_models(cls):
        """Returns a dictionary of all registered models, structured by type."""
        return cls._registry

# Example usage (for testing, not part of the class itself):
if __name__ == '__main__':
    # This block would typically be in a test file.
    # Define dummy config and model classes for demonstration.

    class DummyConfig:
        def __init__(self, name, model_type='default', **kwargs):
            self.name = name
            self.type = model_type
            for key, value in kwargs.items():
                setattr(self, key, value)

    @ModelRegistry.register('dummy_cnn_model', model_type='cnn')
    class DummyCNNModel:
        def __init__(self, config):
            print(f"DummyCNNModel initialized with config: name={config.name}, type={config.type}")
            self.config = config
            self._build_model()
        def _build_model(self):
            print(f"DummyCNNModel _build_model called for {self.config.name}")

    @ModelRegistry.register(['dummy_vit_1', 'dummy_vit_2'], model_type='vit')
    class DummyViTModel:
        def __init__(self, config):
            print(f"DummyViTModel initialized with config: name={config.name}, type={config.type}")
            self.config = config
            self._build_model()
        def _build_model(self):
            print(f"DummyViTModel _build_model called for {self.config.name}")
    
    print("Available models in registry:", ModelRegistry.list_available_models())
    print("All model names:", ModelRegistry.list_models())
    print("CNN model names:", ModelRegistry.list_models(model_type='cnn'))

    try:
        cnn_config = DummyConfig(name='dummy_cnn_model', model_type='cnn', num_classes=10)
        cnn_instance = ModelRegistry.create_model(config=cnn_config)
        
        vit_config = DummyConfig(name='dummy_vit_1', model_type='vit', img_size=224)
        vit_instance = ModelRegistry.create_model(config=vit_config)

        # Test creation of a model not explicitly providing type in config, relying on name uniqueness
        # (Our current create_model iterates types, so this should work if name is unique)
        # vit2_config = DummyConfig(name='dummy_vit_2') 
        # vit2_instance = ModelRegistry.create_model(config=vit2_config)


    except ValueError as e:
        print(f"Error creating model: {e}")