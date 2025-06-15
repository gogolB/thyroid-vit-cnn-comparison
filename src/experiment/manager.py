"""
Comprehensive experiment lifecycle management.
Handles versioning, tracking, and reproducibility using Hydra.
"""
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import importlib
from typing import Type, Any, Dict

from src.experiment.config import ExperimentConfig, AblationConfig, KFoldConfig
from src.experiment.base_experiment import BaseExperiment
from src.experiment.kfold_experiment import KFoldExperiment
from src.experiment.ablation_experiment import AblationExperiment # Import AblationExperiment

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages the lifecycle of experiments, including configuration loading,
    experiment instantiation, and execution.
    """

    def __init__(self): # Removed config_path and config_name, Hydra handles this
        """
        Initializes the ExperimentManager.
        """
        logger.info("ExperimentManager initialized.")

    def _load_experiment_class(self, class_path: str) -> Type[BaseExperiment]:
        """
        Dynamically loads an experiment class given its module and class name.
        Example: "src.experiment_types.standard_experiment.StandardExperiment"
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            experiment_class = getattr(module, class_name)
            if not issubclass(experiment_class, BaseExperiment):
                raise TypeError(f"Class {class_path} must be a subclass of BaseExperiment.")
            return experiment_class
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(f"Error loading experiment class '{class_path}': {e}")
            raise

    def run_experiment_from_config(self, cfg: DictConfig) -> None:
        """
        Loads configuration and runs the specified experiment.
        This method is called by a Hydra-decorated entry point.

        Args:
            cfg: The DictConfig object loaded by Hydra.
        """
        try:
            logger.info("Starting experiment run...")
            logger.info(f"Hydra raw configuration:\n{OmegaConf.to_yaml(cfg)}")

            experiment_config: ExperimentConfig
            # Attempt to directly convert OmegaConf config to the target dataclass
            obj = OmegaConf.to_object(cfg)

            if isinstance(obj, ExperimentConfig):
                experiment_config = obj
            else:
                logger.warning(
                    f"OmegaConf.to_object(cfg) did not return an ExperimentConfig instance (got {type(obj)}). "
                    "Falling back to manual dict conversion and instantiation."
                )
                # Fallback: convert to a Python dictionary first
                cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
                
                if not isinstance(cfg_container, dict):
                    raise TypeError(
                        f"Configuration could not be converted to a dictionary. "
                        f"OmegaConf.to_container returned type: {type(cfg_container)}"
                    )

                # Ensure keys are strings for dataclass instantiation.
                # This is crucial as **kwargs expects string keys.
                final_cfg_dict: Dict[str, Any] = {}
                for key, value in cfg_container.items():
                    final_cfg_dict[str(key)] = value
                
                try:
                    experiment_config = ExperimentConfig(**final_cfg_dict)
                except TypeError as e_manual:
                    logger.error(f"Error instantiating ExperimentConfig from manually processed dict: {e_manual}")
                    logger.error(f"Final dictionary used for instantiation: {final_cfg_dict}")
                    raise ValueError(
                        "Failed to instantiate ExperimentConfig. Ensure your Hydra configuration "
                        "matches the ExperimentConfig structure."
                    ) from e_manual

            logger.info(f"Successfully loaded experiment configuration: {experiment_config.name}")

            experiment_instance: BaseExperiment
            experiment_cls_name: str = "UnknownExperiment"

            # Determine experiment type: K-Fold, Ablation, or Standard
            if experiment_config.kfold and experiment_config.kfold.is_primary_kfold_experiment:
                logger.info(f"Detected K-Fold experiment: {experiment_config.name}")
                experiment_instance = KFoldExperiment(config=experiment_config)
                experiment_cls_name = KFoldExperiment.__name__
            elif experiment_config.ablation and experiment_config.ablation.is_primary_ablation_experiment:
                logger.info(f"Detected Ablation experiment: {experiment_config.name}")
                experiment_instance = AblationExperiment(config=experiment_config)
                experiment_cls_name = AblationExperiment.__name__
            else:
                # Standard experiment loading via class_path from ExperimentConfig
                experiment_class_path = experiment_config.experiment_class_path # Get from ExperimentConfig dataclass
                if not experiment_class_path:
                    logger.error(
                        "No 'experiment_class_path' specified in the ExperimentConfig, "
                        "and not a primary K-Fold or Ablation experiment. Cannot determine experiment type."
                    )
                    raise ValueError(
                        "experiment_class_path must be defined in the ExperimentConfig "
                        "for standard experiments."
                    )
                
                experiment_cls = self._load_experiment_class(experiment_class_path)
                experiment_instance = experiment_cls(config=experiment_config)
                experiment_cls_name = experiment_cls.__name__

            logger.info(f"Instantiated experiment: {experiment_config.name} of type {experiment_cls_name}")

            experiment_instance.execute()
            logger.info(f"Experiment {experiment_config.name} finished successfully.")

            hydra_cfg_runtime = HydraConfig.get()
            logger.info(f"Experiment outputs are in: {hydra_cfg_runtime.runtime.output_dir}")

        except hydra.errors.MissingConfigException as e:
            logger.error(f"Hydra configuration error: {e}. Ensure your config files are correct.")
            raise
        except ImportError as e:
            logger.error(f"Failed to import experiment class: {e}")
            raise
        except TypeError as e:
            logger.error(f"Type error during experiment setup or instantiation: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during the experiment: {e}", exc_info=True)
            raise

# Hydra entry point function
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def launch_experiment(cfg: DictConfig) -> None:
    """
    Main entry point for running experiments via Hydra.
    It instantiates the ExperimentManager and runs the experiment using the loaded config.
    The `config_path` should point to the directory containing `config.yaml` and `experiment/` subdir.
    The `config_name` is typically "config" which then uses defaults to load an experiment config.
    Example `config.yaml`:
    ```yaml
    defaults:
      - experiment: base_experiment_config # or your specific experiment like kfold_exp
      # - override hydra/job_logging: colorlog # Example override
      # - override hydra/hydra_logging: colorlog # Example override

    hydra:
      run:
        dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.name}
      sweep:
        dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
        subdir: ${hydra.job.num}

    # Experiment specific configs will be merged from experiment/*.yaml
    experiment_class_path: "src.experiment.dummy_experiment.DummyExperiment" # Default or overridden by experiment config
    name: "my_experiment_from_main_config"
    # ... other base ExperimentConfig fields ...
    ```
    To run a k-fold experiment, you would have a config in `configs/experiment/my_kfold_exp.yaml`:
    ```yaml
    defaults:
      - base_experiment_config # Inherits base fields
      - kfold_config_schema # Includes kfold schema structure

    name: "my_kfold_experiment"
    description: "A K-Fold cross-validation test."
    # experiment_class_path: not needed if is_primary_kfold_experiment is true

    kfold:
      is_primary_kfold_experiment: true
      num_folds: 3 # Override default
      split_dir: "data/custom_splits"
      # ... other kfold params ...

    model:
      name: "resnet18"
      num_classes: 2
    # ... other model, dataset, trainer configs ...
    ```
    And run: `python src/experiment/manager.py experiment=my_kfold_exp`
    """
    manager = ExperimentManager()
    manager.run_experiment_from_config(cfg)


if __name__ == "__main__":
    # This will be executed when the script is run directly.
    # Hydra will take over and call launch_experiment with the loaded configuration.
    launch_experiment()