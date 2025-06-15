"""
Defines the KFoldExperiment class for running k-fold cross-validation.
"""
import os
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Type # Added Type

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, MISSING
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase # Correct base class for loggers list
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping # For callbacks list

from src.experiment.base_experiment import BaseExperiment
from src.experiment.config import ExperimentConfig # KFoldConfig is part of ExperimentConfig
from src.data.datamodule import ThyroidDataModule
from src.models.registry import ModelRegistry
from src.config.schemas import DatasetConfig as PydanticDatasetConfig, TrainingConfig as PydanticTrainingConfig

# Import specific LightningModules that might be used.
# A more generic way might be needed if the module type is also configurable.
from src.training.lightning_modules import (
    ThyroidCNNModule,
    ThyroidViTModule,
    ThyroidDistillationModule
)
from src.utils.logging import get_logger # Assuming a utility for logger setup

logger = get_logger(__name__) # Use the utility

class KFoldExperiment(BaseExperiment):
    """
    Manages a K-Fold cross-validation experiment.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the K-Fold experiment.

        Args:
            config: The main experiment configuration, which should include kfold settings.
        """
        super().__init__(config)
        if self.config.kfold is None:
            raise ValueError("KFoldConfig is not provided in the experiment configuration.")
        self.kfold_config = self.config.kfold
        self.fold_results: List[Dict[str, Any]] = []
        self.aggregated_results: Dict[str, Any] = {}

    def setup(self) -> None:
        """
        Sets up the overall K-Fold experiment.
        Individual fold setups will happen within the run loop.
        """
        logger.info(f"Setting up K-Fold experiment: {self.config.name}")
        logger.info(f"Number of folds: {self.kfold_config.num_folds}")
        logger.info(f"Split directory: {self.kfold_config.split_dir}")
        logger.info(f"Split file prefix: {self.kfold_config.split_file_prefix}")

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_lightning_module_class(self) -> Union[Type[ThyroidCNNModule], Type[ThyroidViTModule], Type[ThyroidDistillationModule]]:
        """Determines the LightningModule class based on configuration."""
        # This is a simplified way. A more robust approach might involve a registry
        # or a dedicated config field for the lightning module type.
        model_name = self.config.model.get("name", "").lower() # self.config.model is DictConfig or similar
        if "vit" in model_name or "swin" in model_name or "deit" in model_name:
            # self.config is ExperimentConfig (dataclass), self.config.distillation is Optional[Dict[str, Any]]
            if self.config.distillation and self.config.distillation.get("enabled", False):
                logger.info("Using ThyroidDistillationModule for ViT based model with distillation.")
                return ThyroidDistillationModule
            logger.info("Using ThyroidViTModule.")
            return ThyroidViTModule
        elif "cnn" in model_name or "resnet" in model_name or "efficientnet" in model_name or "densenet" in model_name:
            logger.info("Using ThyroidCNNModule.")
            return ThyroidCNNModule
        else:
            # Fallback or raise error
            logger.warning(f"Could not determine LightningModule for model: {model_name}. Defaulting to ThyroidCNNModule.")
            return ThyroidCNNModule


    def run_fold(self, fold_num: int) -> Dict[str, Any]:
        """
        Runs training and evaluation for a single fold.

        Args:
            fold_num: The current fold number (1-indexed).

        Returns:
            A dictionary containing metrics for this fold.
        """
        logger.info(f"Starting fold {fold_num}/{self.kfold_config.num_folds}")

        # 1. Determine split file for the current fold
        split_file_name = f"{self.kfold_config.split_file_prefix}{fold_num}.json"
        split_file_path = Path(self.kfold_config.split_dir) / split_file_name

        if not split_file_path.exists():
            logger.error(f"Split file not found for fold {fold_num}: {split_file_path}")
            raise FileNotFoundError(f"Split file not found: {split_file_path}")

        logger.info(f"Loading data split for fold {fold_num} from: {split_file_path}")

        # 2. Configure DataModule for the current fold
        # We need to pass the specific split file to the DataModule or CARSThyroidDataset
        # Assuming DatasetConfig can accept a `split_file` parameter.
        # Create a mutable copy of the dataset config to override the split file for this fold.
        fold_dataset_config_dict = OmegaConf.to_container(self.config.dataset, resolve=True) # self.config.dataset is Any (likely DictConfig)
        if not isinstance(fold_dataset_config_dict, dict):
            logger.error(f"Dataset config from ExperimentConfig.dataset is not a dictionary after OmegaConf.to_container, got {type(fold_dataset_config_dict)}. Cannot set split_file.")
            raise TypeError("Dataset config could not be converted to a dictionary.")

        fold_dataset_config_dict["split_file"] = str(split_file_path)
        # Add other k-fold specific fields from self.kfold_config if DatasetConfig expects them
        fold_dataset_config_dict["use_kfold"] = True
        fold_dataset_config_dict["fold"] = fold_num
        fold_dataset_config_dict["split_dir"] = self.kfold_config.split_dir
        fold_dataset_config_dict["split_file_prefix"] = self.kfold_config.split_file_prefix

        # Create Pydantic model for type safety with ThyroidDataModule
        try:
            # Ensure fold_dataset_config_dict is Dict[str, Any]
            validated_fold_dataset_dict: Dict[str, Any] = {k: v for k, v in fold_dataset_config_dict.items()}
            pydantic_fold_dataset_config = PydanticDatasetConfig(**validated_fold_dataset_dict)
        except Exception as e:
            logger.error(f"Failed to instantiate PydanticDatasetConfig for fold {fold_num}: {e}")
            logger.error(f"Data used: {fold_dataset_config_dict}")
            raise

        if not hasattr(self.config, 'trainer') or self.config.trainer is None:
             raise ValueError("Trainer configuration (self.config.trainer) is missing in ExperimentConfig.")

        trainer_config_container = OmegaConf.to_container(self.config.trainer, resolve=True)
        if not isinstance(trainer_config_container, dict):
            logger.error(f"Trainer config from ExperimentConfig.trainer is not a dictionary after OmegaConf.to_container, got {type(trainer_config_container)}. Cannot instantiate PydanticTrainingConfig.")
            raise TypeError("Trainer config could not be converted to a dictionary.")
        
        # Ensure trainer_config_dict is Dict[str, Any]
        validated_trainer_config_dict: Dict[str, Any] = {k: v for k, v in trainer_config_container.items()}
        try:
            pydantic_training_config = PydanticTrainingConfig(**validated_trainer_config_dict)
        except Exception as e:
            logger.error(f"Failed to instantiate PydanticTrainingConfig for fold {fold_num}: {e}")
            logger.error(f"Data used: {validated_trainer_config_dict}")
            raise

        data_module = ThyroidDataModule(
            dataset_config=pydantic_fold_dataset_config,
            training_config=pydantic_training_config
        )
        data_module.setup(stage='fit') # Prepare train/val for this fold
        data_module.setup(stage='test') # Prepare test for this fold (if applicable, or use val set as test for fold)

        # 3. Instantiate the model
        # Model instantiation should be consistent across folds, using self.config.model
        model_instance = ModelRegistry.create_model(self.config.model)

        # 4. Instantiate LightningModule
        # The LightningModule needs the overall experiment config, or parts of it
        # It typically takes the full Hydra config (DictConfig)
        # We need to pass a config object that ThyroidCNNModule/ThyroidViTModule expects.
        # Let's assume it expects a DictConfig similar to what Hydra passes.
        # We can reconstruct a relevant DictConfig for the LightningModule.
        # self.config.model, self.config.trainer are DictConfig-like from ExperimentConfig
        # fold_dataset_config is a DictConfig created for the fold
        # Ensure these are converted to basic dicts for OmegaConf.create if they are not already.
        model_cfg_dict = OmegaConf.to_container(self.config.model, resolve=True)
        dataset_cfg_dict_for_lm = OmegaConf.to_container(pydantic_fold_dataset_config.model_dump(), resolve=True) # Use the pydantic model's dict
        trainer_cfg_dict_for_lm = OmegaConf.to_container(pydantic_training_config.model_dump(), resolve=True) # Use the pydantic model's dict

        lightning_module_cfg_parts = {
            "model": model_cfg_dict,
            "dataset": dataset_cfg_dict_for_lm,
            "training": trainer_cfg_dict_for_lm, # This now contains optimizer_params, scheduler_params, loss from PydanticTrainingConfig
             # Access distillation and student_model directly from self.config (ExperimentConfig dataclass)
            "distillation": self.config.distillation, # This is Optional[Dict[str, Any]]
            "student_model": self.config.student_model, # This is Optional[Dict[str, Any]]
        }
        # Filter out None values before creating DictConfig
        filtered_lm_cfg_parts = {k: v for k, v in lightning_module_cfg_parts.items() if v is not None}
        lightning_module_config = OmegaConf.create(filtered_lm_cfg_parts)

        LightningModuleClass = self._get_lightning_module_class()
        lightning_module = LightningModuleClass(config=lightning_module_config)

        # 5. Instantiate PyTorch Lightning Trainer
        # Configure logger for this specific fold
        fold_log_dir = Path(self.config.output_dir) / f"fold_{fold_num}"
        fold_log_dir.mkdir(parents=True, exist_ok=True)

        # Setup loggers (e.g., TensorBoard)
        tb_logger = TensorBoardLogger(save_dir=str(fold_log_dir), name="lightning_logs", version="")
        loggers_list: List[LightningLoggerBase] = [tb_logger] # Correct type hint
        wandb_logger_instance: Optional[WandbLogger] = None


        # Add WandB logger if configured
        if self.config.params.get("use_wandb", False): # self.config.params is Dict[str, Any]
            wandb_project = self.config.params.get("wandb_project", "thyroid_kfold")
            wandb_run_name = f"{self.config.name}_fold_{fold_num}"
            wandb_logger_instance = WandbLogger(project=wandb_project, name=wandb_run_name, save_dir=str(fold_log_dir))
            loggers_list.append(wandb_logger_instance)
            if wandb_logger_instance.experiment is not None: # Check if wandb initialized
                 wandb_logger_instance.experiment.config.update(OmegaConf.to_container(self.config, resolve=True))
                 wandb_logger_instance.experiment.config.update({"fold_number": fold_num})

        # Trainer configuration - adapt from self.config.trainer (which is DictConfig-like)
        # pydantic_training_config is the Pydantic model instance
        # Ensure model_dump returns a Dict[str, Any] suitable for unpacking
        dumped_trainer_params = pydantic_training_config.model_dump(exclude_none=True)
        if not isinstance(dumped_trainer_params, dict):
            raise TypeError(f"Pydantic model_dump did not return a dict for trainer_params, got {type(dumped_trainer_params)}")
        trainer_params_for_pl: Dict[str, Any] = dumped_trainer_params

        # Remove params not directly accepted by pl.Trainer or handled by callbacks/loggers
        trainer_params_for_pl.pop("optimizer_params", None)
        trainer_params_for_pl.pop("scheduler_params", None)
        # monitor_metric, monitor_mode, early_stopping_patience are for callbacks, not Trainer directly
        monitor_metric = trainer_params_for_pl.pop("monitor_metric", "val_loss")
        monitor_mode = trainer_params_for_pl.pop("monitor_mode", "min")
        early_stopping_patience = trainer_params_for_pl.pop("early_stopping_patience", None)
        # save_top_k, save_last are for ModelCheckpoint
        save_top_k = trainer_params_for_pl.pop("save_top_k", 1)
        # save_last = trainer_params_for_pl.pop("save_last", True) # ModelCheckpoint handles this

        trainer_params_for_pl["default_root_dir"] = str(fold_log_dir)

        # Instantiate callbacks
        callbacks_list_for_trainer: List[Callback] = []
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(fold_log_dir / "checkpoints"),
            filename="{epoch}-{"+monitor_metric+":.2f}", # Use monitor_metric
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=save_top_k,
            # save_last=save_last # save_last is a ModelCheckpoint param
        )
        callbacks_list_for_trainer.append(checkpoint_callback)

        if early_stopping_patience is not None:
            # Construct params for EarlyStopping carefully
            early_stop_constructor_params: Dict[str, Any] = {
                "monitor": monitor_metric,
                "patience": early_stopping_patience,
                "mode": monitor_mode
            }
            early_stop_callback = EarlyStopping(**early_stop_constructor_params)
            callbacks_list_for_trainer.append(early_stop_callback)

        # Ensure trainer_params_for_pl is a Dict[str, Any] before unpacking
        if not isinstance(trainer_params_for_pl, dict):
            raise TypeError(f"trainer_params_for_pl is not a dict, got {type(trainer_params_for_pl)}. Cannot unpack for pl.Trainer.")

        trainer = pl.Trainer(
            logger=loggers_list,
            callbacks=callbacks_list_for_trainer,
            **trainer_params_for_pl
        )

        # 6. Run training and evaluation for the current fold
        logger.info(f"Starting training for fold {fold_num}...")
        trainer.fit(lightning_module, datamodule=data_module)
        logger.info(f"Training complete for fold {fold_num}.")

        logger.info(f"Starting testing/validation for fold {fold_num}...")
        test_results_list = trainer.test(lightning_module, datamodule=data_module, ckpt_path="best")
        
        fold_metrics_dict: Dict[str, Any] = {}
        if test_results_list:
            fold_metrics_dict = dict(test_results_list[0]) # Ensure it's a Dict[str, Any]
        
        logger.info(f"Fold {fold_num} metrics: {fold_metrics_dict}")

        if wandb_logger_instance and wandb_logger_instance.experiment is not None:
            wandb_logger_instance.experiment.finish()

        return fold_metrics_dict

    def run(self) -> None:
        """
        Runs the K-Fold cross-validation experiment.
        Iterates through each fold, runs training and evaluation, and aggregates results.
        """
        logger.info(f"Running K-Fold experiment: {self.config.name}")
        self.fold_results = []

        for i in range(self.kfold_config.num_folds):
            fold_num = i + 1
            try:
                fold_metrics = self.run_fold(fold_num)
                self.fold_results.append(fold_metrics)
            except Exception as e:
                logger.error(f"Error during fold {fold_num}: {e}", exc_info=True)
                # Decide if to continue with other folds or stop
                # For now, log error and continue
                self.fold_results.append({"error": str(e), "fold": fold_num})

        self.aggregate_results()
        logger.info("K-Fold experiment run complete.")

    def aggregate_results(self) -> None:
        """
        Aggregates results from all folds (e.g., average metrics).
        """
        logger.info("Aggregating results across all folds...")
        if not self.fold_results:
            logger.warning("No fold results to aggregate.")
            self.aggregated_results = {"status": "No results"}
            return

        # Example: Averaging 'test_acc' and 'test_loss' (keys might vary based on LightningModule)
        # Filter out folds with errors for metric aggregation
        valid_fold_metrics = [res for res in self.fold_results if "error" not in res]

        if not valid_fold_metrics:
            logger.warning("No valid fold results to aggregate metrics from.")
            self.aggregated_results = {"status": "All folds failed or no metrics"}
            # Store raw fold results if any
            self.aggregated_results["raw_fold_results"] = self.fold_results
            return

        aggregated: Dict[str, Any] = {}
        metric_keys = valid_fold_metrics[0].keys() # Get keys from the first valid result

        for key in metric_keys:
            try:
                # Attempt to average numerical metrics
                values = [res[key] for res in valid_fold_metrics if isinstance(res.get(key), (int, float))]
                if values:
                    aggregated[f"avg_{key}"] = np.mean(values)
                    aggregated[f"std_{key}"] = np.std(values)
            except TypeError:
                logger.warning(f"Could not aggregate metric '{key}' due to non-numeric data.")
            except KeyError:
                 logger.warning(f"Metric '{key}' not present in all fold results.")


        self.aggregated_results = aggregated
        self.aggregated_results["num_successful_folds"] = len(valid_fold_metrics)
        self.aggregated_results["total_folds"] = self.kfold_config.num_folds
        self.aggregated_results["raw_fold_results"] = self.fold_results # Include raw results for inspection

        logger.info(f"Aggregated results: {self.aggregated_results}")


    def log_results(self) -> Dict[str, Any]:
        """
        Logs the aggregated results of the K-Fold experiment.
        """
        logger.info("Logging K-Fold experiment results...")
        log_file_path = Path(self.config.output_dir) / "kfold_summary.json"
        try:
            with open(log_file_path, 'w') as f:
                json.dump(self.aggregated_results, f, indent=4)
            logger.info(f"Aggregated K-Fold results saved to: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to save K-Fold summary: {e}")

        # If WandB was used globally for the KFoldExperiment itself (not just per fold)
        # one could log aggregated metrics here.
        # For now, assuming WandB logging is handled per-fold or not at all at this level.

        return self.aggregated_results

    def execute(self) -> Dict[str, Any]:
        """
        Executes the full K-Fold experiment lifecycle.
        Overrides BaseExperiment.execute to handle K-Fold specifics.
        """
        logger.info(f"Starting K-Fold experiment execution: {self.config.name}")
        self.setup()
        self.run() # This now runs all folds and aggregates
        results = self.log_results()
        logger.info(f"K-Fold experiment execution finished for: {self.config.name}")
        return results