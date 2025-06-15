import sys
import os
import logging

# Configure basic logging to see potential import issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to sys.path")

# Import ModelRegistry and model subpackages to ensure registration
try:
    from src.models.registry import ModelRegistry
    logger.info("Successfully imported ModelRegistry.")
except ImportError as e:
    logger.error(f"Failed to import ModelRegistry: {e}")
    sys.exit(1)

try:
    import src.models.cnn
    logger.info("Successfully imported src.models.cnn (for CNN model registration).")
except ImportError as e:
    logger.warning(f"Could not import src.models.cnn: {e}")
except Exception as e:
    logger.warning(f"An unexpected error occurred during import of src.models.cnn: {e}")


try:
    import src.models.vit
    logger.info("Successfully imported src.models.vit (for ViT model registration).")
except ImportError as e:
    logger.warning(f"Could not import src.models.vit: {e}")
except Exception as e:
    logger.warning(f"An unexpected error occurred during import of src.models.vit: {e}")

try:
    import src.models.ensemble # Assuming ensemble models might be registered here
    logger.info("Attempted import of src.models.ensemble.")
except ImportError:
    logger.info("src.models.ensemble not found or not a package, skipping.")
except Exception as e:
    logger.warning(f"An unexpected error occurred during import of src.models.ensemble: {e}")

# Add other model subpackages if they exist (e.g., src.models.hybrid)

# Get all registered model names
try:
    all_model_names = ModelRegistry.list_models()
    if not all_model_names:
        logger.warning("ModelRegistry.list_models() returned an empty list. Ensure models are defined and registered.")
    all_model_names = sorted(list(set(all_model_names))) # Ensure uniqueness and sort
    logger.info(f"Found {len(all_model_names)} models: {all_model_names}")
except Exception as e:
    logger.error(f"Error calling ModelRegistry.list_models(): {e}")
    all_model_names = [] # Ensure it's a list

print("---MODEL_LIST_START---")
if not all_model_names:
    print("NO_MODELS_FOUND")
for name in all_model_names:
    print(name)
print("---MODEL_LIST_END---")

print("\n---COMMAND_LIST_START---")
if not all_model_names:
    print("NO_COMMANDS_GENERATED")
for model_name in all_model_names:
    command = f"python scripts/experiment_runner.py model={model_name} training.max_epochs=1 dataset=default dataset.use_kfold=true dataset.fold=1 training.run_test_after_fit=False"
    # The experiment_runner.py script itself handles img_size adjustments based on model_name
    # when 'model' is passed as a string argument from CLI.
    print(command)
print("---COMMAND_LIST_END---")

logger.info("Finished generating model list and commands.")