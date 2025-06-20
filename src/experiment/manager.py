"""
Comprehensive experiment lifecycle management.
Handles versioning, tracking, and reproducibility using Hydra.
"""
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import importlib
from typing import Type, Any, Dict
import dataclasses # Import dataclasses

from src.experiment.config import ExperimentConfig, AblationConfig, KFoldConfig, AblationParameterConfig # Added AblationParameterConfig
from src.experiment.base_experiment import BaseExperiment
from src.experiment.kfold_experiment import KFoldExperiment
from src.experiment.ablation_experiment import AblationExperiment # Import AblationExperiment

logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    Manages the lifecycle of experiments, including configuration loading,
    experiment instantiation, and execution.
    """

    def __init__(self, project_root: Path): # Removed config_path and config_name, Hydra handles this
        self.project_root = project_root
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
            
            # The actual experiment configuration is expected under the 'experiment' key
            # if an experiment is selected via command line (e.g., +experiment=my_exp)
            # or if the main config.yaml defaults to an experiment.
            # If 'experiment' node exists, use that. Otherwise, assume cfg itself is the ExperimentConfig.
            
            experiment_config: ExperimentConfig
            
            # Construct the dictionary for ExperimentConfig manually
            # Fields like name, description, kfold, ablation, params come from cfg.experiment
            # Fields like model, dataset, trainer come from the global cfg's corresponding nodes
            
            exp_node_for_core_attrs: DictConfig = cfg # Default to root cfg
            if 'experiment' in cfg:
                logger.info("Found 'experiment' node in config, using it for core ExperimentConfig attributes (name, kfold, etc.).")
                exp_node_for_core_attrs = cfg.experiment
            else:
                logger.info("No 'experiment' node in config, using root config for core ExperimentConfig attributes.")

            # Convert the node for core attributes to a Python dict
            exp_core_attrs_container = OmegaConf.to_container(exp_node_for_core_attrs, resolve=True, throw_on_missing=False)
            if not isinstance(exp_core_attrs_container, dict):
                raise TypeError("Experiment node for core attributes could not be converted to a dictionary.")

            # Prepare arguments for ExperimentConfig constructor
            exp_constructor_args: Dict[str, Any] = {}
            known_fields = {f.name for f in dataclasses.fields(ExperimentConfig)}

            for field_name in known_fields:
                if field_name == 'model':
                    model_source_node = cfg.model # Default to global model config
                    if 'experiment' in cfg and hasattr(cfg.experiment, 'model') and cfg.experiment.model is not None:
                        logger.info("Using model config from cfg.experiment.model for ExperimentConfig.model")
                        model_source_node = cfg.experiment.model
                    elif 'model' not in cfg:
                         raise ValueError("Global 'model' config node is missing.")
                    exp_constructor_args['model'] = OmegaConf.create(OmegaConf.to_container(model_source_node, resolve=True))
                elif field_name == 'dataset':
                    # Dataset config for ExperimentConfig.dataset typically comes from global cfg.dataset
                    if 'dataset' not in cfg:
                        raise ValueError("Global 'dataset' config node is missing.")
                    exp_constructor_args['dataset'] = OmegaConf.create(OmegaConf.to_container(cfg.dataset, resolve=True))
                elif field_name == 'trainer':
                    trainer_source_node = cfg.trainer # Default to global trainer config (PL Trainer args)
                    if 'experiment' in cfg and hasattr(cfg.experiment, 'trainer') and cfg.experiment.trainer is not None:
                        logger.info("Using trainer config from cfg.experiment.trainer for ExperimentConfig.trainer (PL Trainer args)")
                        trainer_source_node = cfg.experiment.trainer
                    elif 'trainer' not in cfg:
                         raise ValueError("Global 'trainer' config node (for PL Trainer args) is missing.")
                    exp_constructor_args['trainer'] = OmegaConf.create(OmegaConf.to_container(trainer_source_node, resolve=True))
                elif field_name == 'training_content': # Populate the new field
                    # training_content for ExperimentConfig.training_content always comes from global cfg.training
                    if 'training' in cfg:
                        logger.info("Populating ExperimentConfig.training_content from cfg.training")
                        exp_constructor_args['training_content'] = OmegaConf.create(OmegaConf.to_container(cfg.training, resolve=True))
                    else:
                        # Collect top-level training parameters
                        training_params = {}
                        keys = ['seed', 'epochs', 'batch_size', 'num_workers', 'loss',
                                'optimizer_params', 'early_stopping_patience', 'monitor_metric',
                                'monitor_mode', 'save_top_k', 'save_last', 'log_every_n_steps',
                                'gradient_clip_val', 'accumulate_grad_batches', 'deterministic']
                        
                        # Collect all available keys from config
                        for key in keys:
                            if hasattr(cfg, key):
                                training_params[key] = getattr(cfg, key)
                        
                        # Also include any other keys under 'training' if present
                        if hasattr(cfg, 'training'):
                            training_node = OmegaConf.to_container(cfg.training, resolve=True)
                            if isinstance(training_node, dict):
                                training_params.update(training_node)
                        
                        if training_params:
                            logger.info("Populating ExperimentConfig.training_content from top-level and training config")
                            exp_constructor_args['training_content'] = OmegaConf.create(training_params)
                        else:
                            raise ValueError("Global training parameters are missing from config")
                elif field_name == 'kfold' and 'kfold' in exp_core_attrs_container and isinstance(exp_core_attrs_container['kfold'], dict):
                    exp_constructor_args['kfold'] = KFoldConfig(**exp_core_attrs_container['kfold'])
                elif field_name == 'ablation' and 'ablation' in exp_core_attrs_container and isinstance(exp_core_attrs_container['ablation'], dict):
                    ablation_data = exp_core_attrs_container['ablation']
                    if 'parameter_space' in ablation_data and isinstance(ablation_data['parameter_space'], list):
                        ablation_data['parameter_space'] = [
                            AblationParameterConfig(**p) if isinstance(p, dict) else p
                            for p in ablation_data['parameter_space']
                        ]
                    exp_constructor_args['ablation'] = AblationConfig(**ablation_data)
                elif field_name in exp_core_attrs_container: # For name, description, seed, output_dir, params, etc.
                    exp_constructor_args[field_name] = exp_core_attrs_container[field_name]
                # distillation, student_model, experiment_class_path will be picked up if present in exp_core_attrs_container

            # Ensure all required fields for ExperimentConfig that don't have defaults are present
            # (name, output_dir, seed, model, dataset, trainer are effectively required or have defaults)
            
            logger.info(f"Arguments for ExperimentConfig: { {k: type(v) for k,v in exp_constructor_args.items()} }")
            try:
                experiment_config = ExperimentConfig(**exp_constructor_args)
            except TypeError as e_manual_fallback:
                    logger.error(f"Error instantiating ExperimentConfig from manually processed dict (fallback): {e_manual_fallback}")
                    logger.error(f"Final dictionary used for fallback instantiation: {final_cfg_dict}")
                    raise ValueError(
                        "Failed to instantiate ExperimentConfig even with manual conversion. "
                        "Ensure your experiment configuration structure matches ExperimentConfig."
                    ) from e_manual_fallback
            
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
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
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
    manager = ExperimentManager(project_root=Path(__file__).resolve().parent.parent)
    manager.run_experiment_from_config(cfg)


if __name__ == "__main__":
    # This will be executed when the script is run directly.
    # Hydra will take over and call launch_experiment with the loaded configuration.
    launch_experiment()