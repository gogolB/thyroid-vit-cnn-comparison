"""
Defines the AblationExperiment class for conducting ablation studies.
"""
import os
import logging
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Type, cast

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, MISSING, ListConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

from src.experiment.base_experiment import BaseExperiment
from src.experiment.config import ExperimentConfig, AblationConfig, AblationParameterConfig
from src.data.datamodule import ThyroidDataModule
from src.models.registry import ModelRegistry
from src.config.schemas import DatasetConfig as PydanticDatasetConfig, TrainingConfig as PydanticTrainingConfig

# Import specific LightningModules - adapt as needed or make more generic
from src.training.lightning_modules import (
    ThyroidCNNModule,
    ThyroidViTModule,
    ThyroidDistillationModule
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AblationExperiment(BaseExperiment):
    """
    Manages an ablation study, running multiple experiment variations.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the Ablation experiment.

        Args:
            config: The main experiment configuration, which should include ablation settings.
        """
        super().__init__(config)
        if self.config.ablation is None:
            raise ValueError("AblationConfig is not provided in the experiment configuration.")
        self.ablation_config: AblationConfig = self.config.ablation
        self.ablation_run_results: List[Dict[str, Any]] = []
        self.summary_results: Dict[str, Any] = {}

    def setup(self) -> None:
        """
        Sets up the overall Ablation experiment.
        Individual run setups will happen within the run loop.
        """
        logger.info(f"Setting up Ablation experiment: {self.config.name}")
        logger.info(f"Parameter space: {self.ablation_config.parameter_space}")
        logger.info(f"Base config path: {self.ablation_config.base_config_path}")

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        Generates all combinations of parameters from the parameter_space.
        Each combination is a dictionary of {parameter_path: value}.
        """
        if not self.ablation_config.parameter_space:
            return [{}] # No parameters to ablate, run with base config

        param_names = [p.path for p in self.ablation_config.parameter_space]
        value_lists = [p.values for p in self.ablation_config.parameter_space]

        combinations = []
        for value_combination in itertools.product(*value_lists):
            combo_dict = dict(zip(param_names, value_combination))
            combinations.append(combo_dict)
        return combinations

    def _get_lightning_module_class(self, current_run_config: DictConfig) -> Union[Type[ThyroidCNNModule], Type[ThyroidViTModule], Type[ThyroidDistillationModule]]:
        """Determines the LightningModule class based on configuration for the current run."""
        model_name = str(current_run_config.model.get("name", "")).lower()
        distillation_cfg = current_run_config.get("distillation")

        if "vit" in model_name or "swin" in model_name or "deit" in model_name:
            if distillation_cfg and distillation_cfg.get("enabled", False):
                logger.info("Using ThyroidDistillationModule for ViT based model with distillation.")
                return ThyroidDistillationModule
            logger.info("Using ThyroidViTModule.")
            return ThyroidViTModule
        elif "cnn" in model_name or "resnet" in model_name or "efficientnet" in model_name or "densenet" in model_name:
            logger.info("Using ThyroidCNNModule.")
            return ThyroidCNNModule
        else:
            logger.warning(f"Could not determine LightningModule for model: {model_name}. Defaulting to ThyroidCNNModule.")
            return ThyroidCNNModule


    def run_single_ablation(self, ablation_params: Dict[str, Any], run_index: int) -> Dict[str, Any]:
        """
        Runs a single experiment variation based on the given ablation parameters.

        Args:
            ablation_params: A dictionary where keys are parameter paths (e.g., "model.lr")
                             and values are the specific values for this run.
            run_index: The index of the current ablation run.

        Returns:
            A dictionary containing metrics for this ablation run.
        """
        ablation_name_parts = [f"{key.split('.')[-1]}_{value}" for key, value in ablation_params.items()]
        ablation_suffix = "_".join(ablation_name_parts) if ablation_name_parts else "base"
        run_name = self.ablation_config.name_pattern.format(ablation_count=run_index, ablation_suffix=ablation_suffix)
        
        logger.info(f"Starting ablation run {run_index}: {run_name} with params: {ablation_params}")

        # 1. Create the configuration for this specific run
        # Start with a deep copy of the original experiment config's relevant parts
        # The self.config is an ExperimentConfig dataclass. We need to convert its parts to DictConfig
        # to merge with overrides.
        
        base_run_cfg_dict = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False)
        # Ensure base_run_cfg_dict is a dictionary
        if not isinstance(base_run_cfg_dict, dict):
            raise TypeError(f"Could not convert self.config to a dictionary. Got {type(base_run_cfg_dict)}")
        
        # Remove kfold and ablation sections as they are not relevant for a single run
        # and can cause issues if their structure doesn't match what sub-components expect.
        base_run_cfg_dict.pop("kfold", None)
        base_run_cfg_dict.pop("ablation", None)

        current_run_config = OmegaConf.create(base_run_cfg_dict)

        # If a base_config_path is specified in AblationConfig, load it and merge.
        # This allows ablation to run on top of a different base than the main experiment config.
        if self.ablation_config.base_config_path:
            try:
                # Temporarily clear Hydra's global state if it's initialized,
                # to load a config file as if it's a new Hydra application.
                # This is a bit of a hack and might have side effects.
                # A cleaner way would be to use Hydra's compose API if available and suitable.
                gh = GlobalHydra.instance()
                if gh.is_initialized():
                    gh.clear() # Clear it before re-initializing for loading base_config
                
                # Initialize Hydra minimally to load the config file
                # The config_path should be relative to the original Hydra config root
                # This assumes `self.ablation_config.base_config_path` is like `experiment/my_base_exp`
                # and the main config dir is `../configs` relative to manager.py
                # We need to find the absolute path to the config directory.
                # This is tricky as we don't have direct access to Hydra's original config path here.
                # For now, assume base_config_path is relative to a known config root or an absolute path.
                # A common pattern is that hydra.main has `config_path` set.
                # Let's assume for now that the path is resolvable by OmegaConf directly
                # or we might need to pass the original config root.

                # This part is complex because we are "inside" a Hydra app trying to load another config.
                # A simpler approach: assume base_config_path refers to a named config group.
                # However, the spec says "Path to a base Hydra configuration file".
                
                # Let's assume the path is relative to the original config directory.
                # This is hard to get robustly. A placeholder for now.
                # For a robust solution, the ExperimentManager might need to pass down the original config_path.
                # As a fallback, if OmegaConf.load fails, we log a warning.
                # base_hydra_cfg_for_ablation = OmegaConf.load(self.ablation_config.base_config_path)

                # Alternative: If base_config_path is a named config (e.g., "experiment/base_model")
                # and the config structure is already loaded by Hydra, we might be able to compose it.
                # This requires more advanced Hydra usage (compose API).

                # For now, let's assume `self.ablation_config.base_config_path` is a file path
                # that OmegaConf can load. This might need adjustment based on project structure.
                # A common setup is `configs/experiment/my_base.yaml`.
                # If `self.config.output_dir` is like `outputs/some_job_name`, we can't easily go to `configs`.
                # This needs a more robust way to specify and resolve `base_config_path`.
                # For this implementation, we'll skip loading from file if it's too complex,
                # and rely on the main `self.config` as the base.
                logger.warning(f"Loading from base_config_path ('{self.ablation_config.base_config_path}') is not fully implemented yet. Using current experiment config as base.")

            except Exception as e:
                logger.error(f"Could not load or merge base_config_path '{self.ablation_config.base_config_path}': {e}")
                # Decide: raise error or continue with self.config as base? For now, continue.

        # Apply ablation parameter overrides
        for param_path, value in ablation_params.items():
            try:
                OmegaConf.update(current_run_config, param_path, value, merge=True)
            except Exception as e:
                logger.error(f"Failed to update config with {param_path}={value}: {e}")
                raise
        
        logger.info(f"Effective config for run {run_name}:\n{OmegaConf.to_yaml(current_run_config)}")

        # 2. Setup DataModule
        # Convert relevant parts of current_run_config to Pydantic models for DataModule
        dataset_cfg_dict = OmegaConf.to_container(current_run_config.dataset, resolve=True)
        if not isinstance(dataset_cfg_dict, dict):
            raise TypeError("Ablation run dataset_cfg_dict is not a dict.")
        # Ensure keys are strings for Pydantic
        dataset_cfg_dict_str_keys: Dict[str, Any] = {str(k): v for k, v in dataset_cfg_dict.items()}
        pydantic_dataset_config = PydanticDatasetConfig(**dataset_cfg_dict_str_keys)

        trainer_cfg_dict = OmegaConf.to_container(current_run_config.trainer, resolve=True)
        if not isinstance(trainer_cfg_dict, dict):
            raise TypeError("Ablation run trainer_cfg_dict is not a dict.")
        trainer_cfg_dict_str_keys: Dict[str, Any] = {str(k): v for k, v in trainer_cfg_dict.items()}
        pydantic_training_config = PydanticTrainingConfig(**trainer_cfg_dict_str_keys)

        data_module = ThyroidDataModule(
            dataset_config=pydantic_dataset_config,
            training_config=pydantic_training_config
        )
        data_module.setup(stage='fit')
        data_module.setup(stage='test')

        # 3. Instantiate Model
        model_instance = ModelRegistry.create_model(current_run_config.model) # Pass DictConfig part

        # 4. Instantiate LightningModule
        # The LightningModule needs a config object. We pass the current_run_config (DictConfig)
        # which contains model, dataset, training, distillation sections.
        LightningModuleClass = self._get_lightning_module_class(current_run_config)
        lightning_module = LightningModuleClass(config=current_run_config)


        # 5. Instantiate PyTorch Lightning Trainer
        run_output_dir = Path(self.config.output_dir) / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        tb_logger = TensorBoardLogger(save_dir=str(run_output_dir), name="lightning_logs", version="")
        loggers_list: List[LightningLoggerBase] = [tb_logger]
        
        # Add WandB logger if configured in the main config's params
        if self.config.params.get("use_wandb", False):
            wandb_project = self.config.params.get("wandb_project", "thyroid_ablation")
            # Ensure wandb_run_name is unique for each ablation run
            wandb_run_name_for_ablation = f"{self.config.name}_{run_name}"
            wandb_logger_instance = WandbLogger(
                project=wandb_project, 
                name=wandb_run_name_for_ablation, 
                save_dir=str(run_output_dir),
                # config=OmegaConf.to_container(current_run_config, resolve=True) # Log the specific run config
            )
            if wandb_logger_instance.experiment is not None: # Check if wandb initialized
                 # Log the specific run config to wandb
                wandb_logger_instance.experiment.config.update(OmegaConf.to_container(current_run_config, resolve=True))
                wandb_logger_instance.experiment.config.update({"ablation_run_name": run_name, "ablation_params": ablation_params})

            loggers_list.append(wandb_logger_instance)


        # Trainer parameters from PydanticTrainingConfig
        dumped_trainer_params = pydantic_training_config.model_dump(exclude_none=True)
        trainer_params_for_pl: Dict[str, Any] = {str(k): v for k, v in dumped_trainer_params.items()}

        monitor_metric = trainer_params_for_pl.pop("monitor_metric", "val_loss")
        monitor_mode = trainer_params_for_pl.pop("monitor_mode", "min")
        early_stopping_patience = trainer_params_for_pl.pop("early_stopping_patience", None)
        save_top_k = trainer_params_for_pl.pop("save_top_k", 1)
        
        trainer_params_for_pl.pop("optimizer_params", None) # Handled by LightningModule
        trainer_params_for_pl.pop("scheduler_params", None) # Handled by LightningModule

        trainer_params_for_pl["default_root_dir"] = str(run_output_dir)

        callbacks_list_for_trainer: List[Callback] = []
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_output_dir / "checkpoints"),
            filename="{epoch}-{" + monitor_metric + ":.2f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=save_top_k,
        )
        callbacks_list_for_trainer.append(checkpoint_callback)

        if early_stopping_patience is not None and early_stopping_patience > 0:
            early_stop_callback = EarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                mode=monitor_mode
            )
            callbacks_list_for_trainer.append(early_stop_callback)

        trainer = pl.Trainer(
            logger=loggers_list,
            callbacks=callbacks_list_for_trainer,
            **trainer_params_for_pl
        )

        # 6. Run training and testing
        logger.info(f"Starting training for ablation run: {run_name}")
        trainer.fit(lightning_module, datamodule=data_module)
        logger.info(f"Training complete for ablation run: {run_name}")

        logger.info(f"Starting testing for ablation run: {run_name}")
        # Use the best checkpoint for testing
        test_results_list = trainer.test(lightning_module, datamodule=data_module, ckpt_path="best")
        
        run_metrics: Dict[str, Any] = {}
        if test_results_list:
            # test_results_list is a list of dicts, one per test dataloader. Assume one for now.
            run_metrics = dict(test_results_list[0]) 
        
        run_metrics["ablation_run_name"] = run_name
        run_metrics["ablation_params"] = ablation_params
        logger.info(f"Ablation run {run_name} metrics: {run_metrics}")

        # Finalize WandB run if used
        if self.config.params.get("use_wandb", False) and 'wandb_logger_instance' in locals():
            if wandb_logger_instance.experiment is not None:
                wandb_logger_instance.experiment.finish()
        
        # Clean up Hydra's global state if we modified it for loading base_config
        # This is part of the hacky solution and might need refinement.
        gh = GlobalHydra.instance()
        if gh.is_initialized(): # If we cleared and re-init'd it
            # It's probably best not to clear it here, but ensure the main hydra context is restored
            # if it was ever changed. This part is risky.
            pass


        return run_metrics

    def run(self) -> None:
        """
        Runs the full ablation study.
        Iterates through each parameter combination, runs an experiment, and collects results.
        """
        logger.info(f"Running Ablation Study: {self.config.name}")
        self.ablation_run_results = []
        
        parameter_combinations = self._generate_parameter_combinations()
        if not parameter_combinations:
            logger.warning("No parameter combinations generated for ablation. Check config.")
            # Potentially run a single base experiment if parameter_space is empty
            # For now, just return.
            return

        for i, params in enumerate(parameter_combinations):
            try:
                run_metrics = self.run_single_ablation(params, i)
                self.ablation_run_results.append(run_metrics)
            except Exception as e:
                run_name_failed = f"ablation_run_{i}_failed"
                logger.error(f"Error during ablation run {i} with params {params}: {e}", exc_info=True)
                self.ablation_run_results.append({
                    "ablation_run_name": run_name_failed,
                    "ablation_params": params,
                    "error": str(e)
                })
        
        self.summarize_results()
        logger.info("Ablation study run complete.")

    def summarize_results(self) -> None:
        """
        Summarizes results from all ablation runs.
        (e.g., find best run, average metrics if meaningful).
        """
        logger.info("Summarizing ablation study results...")
        if not self.ablation_run_results:
            logger.warning("No ablation run results to summarize.")
            self.summary_results = {"status": "No results"}
            return

        self.summary_results["all_runs"] = self.ablation_run_results
        
        # Example: Find the run with the best 'test_accuracy' (adjust metric name as needed)
        best_run = None
        best_accuracy = -1.0 # Assuming accuracy is positive

        for run_result in self.ablation_run_results:
            if "error" not in run_result:
                # Common metrics: test_acc, test_accuracy, val_acc, val_accuracy
                # Check for common accuracy keys
                acc_keys_to_check = ['test_accuracy', 'test_acc', 'val_accuracy', 'val_acc']
                current_accuracy = None
                for acc_key in acc_keys_to_check:
                    if acc_key in run_result:
                        current_accuracy = run_result[acc_key]
                        break
                
                if current_accuracy is not None and isinstance(current_accuracy, (float, int)):
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_run = run_result
        
        if best_run:
            self.summary_results["best_run"] = best_run
            logger.info(f"Best ablation run: {best_run.get('ablation_run_name')} with params {best_run.get('ablation_params')} and accuracy {best_accuracy}")
        else:
            logger.info("Could not determine best run based on accuracy.")
            self.summary_results["best_run"] = "N/A (Could not determine based on accuracy)"


        logger.info(f"Ablation summary: {json.dumps(self.summary_results, indent=2, default=str)}")


    def log_results(self) -> Dict[str, Any]:
        """
        Logs the summary of the ablation study.
        """
        logger.info("Logging ablation study summary...")
        summary_file_path = Path(self.config.output_dir) / "ablation_summary.json"
        try:
            with open(summary_file_path, 'w') as f:
                # Use a custom serializer for Path objects or other non-serializable types if they appear
                json.dump(self.summary_results, f, indent=4, default=str) 
            logger.info(f"Ablation study summary saved to: {summary_file_path}")
        except Exception as e:
            logger.error(f"Failed to save ablation summary: {e}")

        return self.summary_results

    def execute(self) -> Dict[str, Any]:
        """
        Executes the full ablation study lifecycle.
        """
        logger.info(f"Starting Ablation Experiment execution: {self.config.name}")
        self.setup()
        self.run()  # This runs all ablation variations and summarizes
        results = self.log_results()
        logger.info(f"Ablation Experiment execution finished for: {self.config.name}")
        return results