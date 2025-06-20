import os
import glob
import logging
import yaml
import json
import numpy
import shutil
from pathlib import Path
import yaml # Added for loading model config YAML

from src.experiment.kfold_experiment import KFoldExperiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

class AllModelsFullKFoldExperiment(KFoldExperiment):
    """
    Experiment class to run k-fold cross-validation for all configured models.
    It scans for model configurations, runs k-fold for each, aggregates results,
    and manages checkpoints.
    """

    def __init__(self, experiment_config_path: str, base_output_dir: str = "experiments"):
        """
        Initializes the AllModelsFullKFoldExperiment.

        Args:
            experiment_config_path (str): Path to the main experiment configuration YAML file.
            base_output_dir (str): Base directory where experiment outputs will be saved.
                                   A subdirectory for this specific experiment will be created.
        """
        super().__init__(experiment_config_path, base_output_dir)
        # self.experiment_config is loaded by the parent KFoldExperiment's __init__
        # If specific global settings for AllModelsFullKFoldExperiment are needed from
        # the config, they can be accessed via self.config
        logger.info(f"Initialized AllModelsFullKFoldExperiment with config: {experiment_config_path}")
        logger.info(f"Output directory for this experiment run: {self.output_dir}")

    def _get_model_configs(self) -> list[Path]:
        """
        Scans specified directories for model configuration files (*.yaml).

        Excludes 'base.yaml', 'base_cnn.yaml', 'base_transformer.yaml', and any '__init__.yaml'.

        Returns:
            list[Path]: A list of Path objects to the model configuration files.
        """
        model_config_dirs = [
            Path("configs/model/cnn"),
            Path("configs/model/vit")
        ]
        excluded_files = ["base.yaml", "base_cnn.yaml", "base_transformer.yaml", "__init__.yaml"]
        
        model_config_paths = []
        for config_dir in model_config_dirs:
            if not config_dir.is_dir():
                logger.warning(f"Model configuration directory not found: {config_dir}")
                continue
            for file_path in config_dir.glob("*.yaml"):
                if file_path.name not in excluded_files:
                    model_config_paths.append(file_path)
        
        logger.info(f"Found {len(model_config_paths)} model configurations.")
        return model_config_paths

    def _get_model_name_from_path(self, config_path: Path) -> str:
        """
        Extracts a clean model name from the configuration file path.
        Example: Path("configs/model/cnn/resnet50.yaml") -> "resnet50"

        Args:
            config_path (Path): The path to the model configuration file.

        Returns:
            str: The extracted model name.
        """
        return config_path.stem

    def run_experiment(self):
        """
        Runs the full k-fold cross-validation experiment for all discovered models.
        """
        logger.info("Starting AllModelsFullKFoldExperiment...")
        all_models_summary = {}
        
        model_config_paths = self._get_model_configs()
        logger.info(f"Found {len(model_config_paths)} models to process.")

        if not model_config_paths:
            logger.warning("No model configurations found. Exiting experiment.")
            return

        for model_config_path in model_config_paths:
            model_name = self._get_model_name_from_path(model_config_path)
            logger.info(f"Processing model: {model_name} from config: {model_config_path}")

            # --- Configuration for the current model ---
            logger.info(f"Configuring experiment for model: {model_name}")

            # 1. Load Model Config
            # Store the path to the model config, as per instruction
            self.config.model_config = str(model_config_path.resolve())
            # Load the model configuration content into self.config.model
            # KFoldExperiment uses self.config.model to instantiate the model.
            try:
                self.config.model = yaml.safe_load(model_config_path.read_text())
                logger.info(f"Loaded model config for {model_name} from {model_config_path}")
            except Exception as e:
                logger.error(f"Failed to load model config {model_config_path} for {model_name}: {e}")
                continue # Skip to the next model

            # 2. Set Quality Preprocessing for Dataset
            # self.config.dataset is initialized as a dict by ExperimentConfig
            quality_dataset_config_path = Path('configs/dataset/quality_preprocessing/quality_preprocessing.yaml').resolve()
            self.config.dataset['path'] = str(quality_dataset_config_path)
            # Optionally, ensure other quality flags are set if the loaded dataset config relies on them
            # self.config.dataset['quality_preprocessing'] = True # If schema expects this explicitly
            logger.info(f"Set dataset path to quality preprocessing: {quality_dataset_config_path}")

            # 3. Ensure Pre-trained Weights
            # self.config.trainer is initialized as a dict by ExperimentConfig
            # As per explicit instruction "Set self.config.training.pretrained = True"
            # Assuming 'training' and 'trainer' config sections are closely related or 'trainer' is the primary one used.
            # KFoldExperiment uses self.config.trainer for PydanticTrainingConfig.
            self.config.trainer['pretrained'] = True
            logger.info(f"Set self.config.trainer['pretrained'] = True for {model_name}")
            # Also, ensure the model's own configuration reflects pretrained=True, if it has such a flag.
            if isinstance(self.config.model, dict):
                self.config.model['pretrained'] = True
            else:
                logger.warning(f"self.config.model for {model_name} is not a dict, cannot directly set pretrained flag in model config.")


            # --- Output Directory for this model ---
            model_output_dir = self.output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory for {model_name}: {model_output_dir}")

            # --- Execute K-Fold Cross-Validation for the Current Model ---
            logger.info(f"Starting k-fold cross-validation for model: {model_name}")
            model_fold_results = []
            
            # Temporarily change self.config.output_dir for KFoldExperiment's run_fold method
            # so that its outputs (logs, checkpoints) go into model_output_dir/fold_X
            original_experiment_output_dir = self.config.output_dir
            self.config.output_dir = str(model_output_dir) # Must be string for Path() constructor

            if self.config.kfold is None:
                logger.error(f"KFoldConfig (self.config.kfold) is None for model {model_name}. Cannot determine number of folds. Skipping.")
                self.config.output_dir = original_experiment_output_dir # Restore
                continue # Skip to next model
                
            num_folds_to_run = self.config.kfold.num_folds

            for fold_idx in range(num_folds_to_run):
                current_fold_num = fold_idx + 1 # KFoldExperiment.run_fold expects 1-indexed
                logger.info(f"Running fold {current_fold_num}/{num_folds_to_run} for model {model_name}...")
                
                try:
                    # KFoldExperiment.run_fold uses the current self.config, which we've updated.
                    # It returns a dictionary of metrics from trainer.test().
                    fold_run_metrics = super().run_fold(fold_num=current_fold_num)

                    # Determine the path to the best checkpoint for this fold.
                    # KFoldExperiment.run_fold saves checkpoints to:
                    # Path(self.config.output_dir) / f"fold_{fold_num}" / "checkpoints"
                    # which, due to our temporary change, is: model_output_dir / f"fold_{current_fold_num}" / "checkpoints"
                    fold_checkpoints_dir = model_output_dir / f"fold_{current_fold_num}" / "checkpoints"
                    
                    # Find the .ckpt file. Assuming save_top_k=1 and save_last=False (or not explicitly True for ModelCheckpoint)
                    # means there should be one primary .ckpt file (the best one).
                    found_ckpt_files = sorted(list(fold_checkpoints_dir.glob("*.ckpt")))
                    
                    actual_checkpoint_path = None
                    if found_ckpt_files:
                        # Prefer checkpoint not named "last.ckpt" if multiple exist and one is "last.ckpt"
                        non_last_ckpts = [p for p in found_ckpt_files if p.name != "last.ckpt"]
                        if non_last_ckpts:
                            actual_checkpoint_path = str(non_last_ckpts[0])
                            if len(non_last_ckpts) > 1:
                                logger.warning(f"Multiple non-last checkpoints found in {fold_checkpoints_dir}. Using first one: {actual_checkpoint_path}")
                        elif found_ckpt_files: # Only last.ckpt or other non-standard names
                             actual_checkpoint_path = str(found_ckpt_files[0]) # Fallback to first found
                             logger.warning(f"Only last.ckpt or other specific checkpoints found. Using first one: {actual_checkpoint_path}")
                    else:
                        logger.warning(f"No checkpoint files found in {fold_checkpoints_dir} for model {model_name}, fold {current_fold_num}.")

                    # Construct the result dictionary for this fold
                    # The test accuracy key might be 'test_accuracy' or 'test_acc'.
                    # KFoldExperiment.aggregate_results looks for 'avg_test_acc', implying 'test_acc'.
                    test_accuracy_value = fold_run_metrics.get('test_accuracy', fold_run_metrics.get('test_acc', 0.0))

                    current_fold_result = {
                        'fold': current_fold_num,
                        'test_accuracy': test_accuracy_value,
                        'checkpoint_path': actual_checkpoint_path
                    }
                    model_fold_results.append(current_fold_result)
                    logger.info(f"Fold {current_fold_num} for model {model_name} completed. Result: {current_fold_result}")

                except Exception as e_fold:
                    logger.error(f"Error during fold {current_fold_num} for model {model_name}: {e_fold}", exc_info=True)
                    model_fold_results.append({
                        'fold': current_fold_num,
                        'test_accuracy': 0.0,
                        'checkpoint_path': None,
                        'error': str(e_fold)
                    })
            
            # Restore the original output directory in self.config
            self.config.output_dir = original_experiment_output_dir
            logger.info(f"Finished k-fold CV for model: {model_name}. Results for {len(model_fold_results)} folds obtained.")

            aggregated_metrics = self._aggregate_model_results(model_fold_results)
            logger.info(f"Aggregated metrics for {model_name}: {aggregated_metrics}")
            
            self._manage_model_checkpoints(model_name, model_fold_results, aggregated_metrics, model_output_dir)

            all_models_summary[model_name] = aggregated_metrics
            
            logger.info(f"Completed processing for model: {model_name}")
            
            # Restore original dataset path if it was overridden
            # if 'original_dataset_path' in locals():
            #    self.config.dataset.path = original_dataset_path


        self._save_overall_summary(all_models_summary)
        
        logger.info("AllModelsFullKFoldExperiment finished.")

    def _aggregate_model_results(self, model_fold_results: list[dict]) -> dict:
        """
        Aggregates results from all folds for a single model.
        Calculates average, standard deviation, highest, and lowest test accuracies,
        along with the fold numbers where the highest and lowest accuracies occurred.

        Args:
            model_fold_results (list[dict]): A list of dictionaries, where each
                dictionary represents results from one fold. Expected keys include
                'fold' (int) and 'test_accuracy' (float).

        Returns:
            dict: A dictionary containing aggregated metrics:
                - 'average_test_accuracy' (float)
                - 'std_dev_test_accuracy' (float)
                - 'highest_test_accuracy' (float)
                - 'highest_accuracy_fold' (int)
                - 'lowest_test_accuracy' (float)
                - 'lowest_accuracy_fold' (int)
        """
        logger.debug(f"Aggregating model results from {len(model_fold_results)} folds.")

        if not model_fold_results:
            logger.warning("No model fold results provided for aggregation. Returning default metrics.")
            return {
                'average_test_accuracy': 0.0,
                'std_dev_test_accuracy': 0.0,
                'highest_test_accuracy': 0.0,
                'highest_accuracy_fold': -1,
                'lowest_test_accuracy': 0.0,
                'lowest_accuracy_fold': -1,
            }

        valid_accuracies = []
        for fold_data in model_fold_results:
            acc = fold_data.get('test_accuracy')
            if acc is not None:
                try:
                    valid_accuracies.append(float(acc))
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert accuracy to float for fold {fold_data.get('fold', 'N/A')}: {acc}. Skipping this accuracy.")
        
        if not valid_accuracies:
            logger.warning("No valid numerical test accuracies found after processing all folds. Returning default metrics for calculations.")
            avg_acc = 0.0
            std_acc = 0.0
        else:
            avg_acc = numpy.mean(valid_accuracies)
            std_acc = numpy.std(valid_accuracies)

        highest_test_accuracy = 0.0
        highest_accuracy_fold = -1
        lowest_test_accuracy = 0.0
        lowest_accuracy_fold = -1

        first_valid_fold_for_min_max = True

        for fold_result in model_fold_results:
            accuracy_val = fold_result.get('test_accuracy')
            fold_num = fold_result.get('fold')

            if accuracy_val is None or fold_num is None:
                logger.debug(f"Skipping fold entry for min/max search due to missing accuracy or fold number: {fold_result}")
                continue
            
            try:
                current_accuracy = float(accuracy_val)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert accuracy to float for fold {fold_num} in min/max search: {accuracy_val}. Skipping this entry for min/max.")
                continue

            if first_valid_fold_for_min_max:
                highest_test_accuracy = current_accuracy
                highest_accuracy_fold = fold_num
                lowest_test_accuracy = current_accuracy
                lowest_accuracy_fold = fold_num
                first_valid_fold_for_min_max = False
            else:
                if current_accuracy > highest_test_accuracy:
                    highest_test_accuracy = current_accuracy
                    highest_accuracy_fold = fold_num
                
                if current_accuracy < lowest_test_accuracy:
                    lowest_test_accuracy = current_accuracy
                    lowest_accuracy_fold = fold_num
        
        return {
            'average_test_accuracy': float(avg_acc),
            'std_dev_test_accuracy': float(std_acc),
            'highest_test_accuracy': float(highest_test_accuracy),
            'highest_accuracy_fold': int(highest_accuracy_fold),
            'lowest_test_accuracy': float(lowest_test_accuracy),
            'lowest_accuracy_fold': int(lowest_accuracy_fold),
        }

    def _manage_model_checkpoints(self, model_name: str, model_fold_results: list[dict], aggregated_metrics: dict, model_output_dir: Path):
        """
        Manages model checkpoints for a given model after k-fold cross-validation:
        1. Identifies the checkpoint file from the fold with the highest test accuracy.
        2. Creates a 'best_checkpoint' subdirectory within the model's output directory.
        3. Copies the identified best checkpoint file to this 'best_checkpoint' subdirectory.
        4. Removes all individual fold-specific directories (e.g., 'fold_0', 'fold_1', ...)
           that were created during the k-fold process to store per-fold checkpoints and logs.

        Args:
            model_name (str): Name of the model.
            model_fold_results (list[dict]): List of dictionaries, each containing results
                                             from one fold, including 'fold', 'test_accuracy',
                                             and 'checkpoint_path'.
            aggregated_metrics (dict): Dictionary from _aggregate_model_results, containing
                                       'highest_test_accuracy' and 'highest_accuracy_fold'.
            model_output_dir (Path): Path object for the model's output directory
                                     (e.g., experiments/AllModelsFullKFold/resnet50/).
        """
        logger.info(f"[{model_name}] Managing checkpoints. Aggregated metrics: {aggregated_metrics}")

        # 1. Identify Best Checkpoint
        best_fold_number = aggregated_metrics.get('highest_accuracy_fold')
        
        # highest_accuracy_fold can be -1 if no valid accuracies were found or all folds errored.
        if best_fold_number is None or best_fold_number < 0: # Fold numbers are 1-indexed or -1 if error
            logger.error(f"[{model_name}] No valid best fold identified (fold: {best_fold_number}). Cannot manage checkpoints.")
            return

        best_checkpoint_path_str = None
        for fold_result in model_fold_results:
            if fold_result.get('fold') == best_fold_number:
                best_checkpoint_path_str = fold_result.get('checkpoint_path')
                break
        
        if not best_checkpoint_path_str:
            logger.error(f"[{model_name}] Checkpoint path for best fold {best_fold_number} not found in model_fold_results. This might be due to an error during that fold's training or checkpoint saving.")
            # If the best fold had an error, its checkpoint_path might be None.
            # In this scenario, we cannot copy a "best" checkpoint. We stop to avoid deleting other potentially useful fold data.
            return

        try:
            best_checkpoint_path = Path(best_checkpoint_path_str)
            if not best_checkpoint_path.is_file(): # Check if it's a file and exists
                logger.error(f"[{model_name}] Best checkpoint is not a file or does not exist: {best_checkpoint_path}")
                return
        except TypeError: # Handles if best_checkpoint_path_str was None (though 'if not' should catch it)
             logger.error(f"[{model_name}] Invalid checkpoint path string '{best_checkpoint_path_str}' for best fold {best_fold_number}.")
             return

        logger.info(f"[{model_name}] Identified best checkpoint file: {best_checkpoint_path} from fold {best_fold_number}")

        # 2. Create 'best_checkpoint' directory
        best_checkpoint_dir = model_output_dir / "best_checkpoint"
        try:
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{model_name}] Ensured 'best_checkpoint' directory exists: {best_checkpoint_dir}")
        except OSError as e:
            logger.error(f"[{model_name}] Error creating directory {best_checkpoint_dir}: {e}", exc_info=True)
            return # Cannot proceed if this directory cannot be created

        # 3. Copy Best Checkpoint
        destination_path = best_checkpoint_dir / best_checkpoint_path.name
        try:
            shutil.copy2(best_checkpoint_path, destination_path)
            logger.info(f"[{model_name}] Successfully copied best checkpoint from {best_checkpoint_path} to {destination_path}")
        except Exception as e:
            logger.error(f"[{model_name}] Error copying best checkpoint: {e}", exc_info=True)
            # If copy fails, we do not proceed to delete other checkpoints to prevent data loss.
            return

        # 4. Remove Other Per-Fold Directories
        logger.info(f"[{model_name}] Starting cleanup of per-fold directories in {model_output_dir}")
        removed_count = 0
        error_count = 0
        # It's safer to list directory contents before iterating and modifying
        try:
            items_in_model_dir = list(model_output_dir.iterdir())
        except FileNotFoundError:
            logger.error(f"[{model_name}] Model output directory not found for cleanup: {model_output_dir}", exc_info=True)
            return


        for item_path in items_in_model_dir:
            if item_path.is_dir() and item_path.name.startswith("fold_"):
                # This is a fold-specific directory, e.g., experiments/AllModelsFullKFold/resnet50/fold_1/
                try:
                    shutil.rmtree(item_path)
                    logger.info(f"[{model_name}] Removed per-fold directory: {item_path}")
                    removed_count += 1
                except OSError as e:
                    logger.error(f"[{model_name}] Error removing directory {item_path}: {e}", exc_info=True)
                    error_count += 1
            elif item_path.resolve() == best_checkpoint_dir.resolve():
                logger.debug(f"[{model_name}] Skipping 'best_checkpoint' directory during cleanup: {item_path}")
            # else:
                # logger.debug(f"[{model_name}] Skipping item during cleanup (not a fold_ dir or best_checkpoint_dir): {item_path.name}")
        
        if removed_count > 0 or error_count > 0:
            logger.info(f"[{model_name}] Cleanup of per-fold directories completed. Removed: {removed_count}, Errors: {error_count}.")
        else:
            logger.info(f"[{model_name}] No per-fold directories found or removed during cleanup in {model_output_dir}.")

    def _save_overall_summary(self, summary_data: dict):
        """
        Saves the overall summary of all models to a JSON file.

        Args:
            summary_data (dict): A dictionary where keys are model names
                                 and values are their aggregated metrics.
        """
        summary_file_path = self.output_dir / "all_models_summary.json"
        logger.info(f"Attempting to save overall experiment summary to: {summary_file_path}")
        try:
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=4)
            logger.info(f"Overall summary successfully saved to: {summary_file_path}")
        except IOError as e:
            logger.error(f"IOError occurred while writing summary file {summary_file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving overall summary to {summary_file_path}: {e}", exc_info=True)

if __name__ == '__main__':
    # This is an example of how to run the experiment.
    # Ensure 'kfold_experiment_config.yaml' is correctly set up.
    # It should define 'n_splits', 'dataset' config, 'training' config, etc.
    # The 'model' config part will be overridden by each model found.
    
    # Create a dummy experiment config for testing
    dummy_config_content = {
        "experiment_name": "AllModelsFullKFold_TestRun",
        "n_splits": 3, # Example: 3 folds
        "dataset": {
            "name": "ThyroidDataset",
            "path": "configs/dataset/default.yaml", # This will be used as a base
            "params": {"target_size": [224, 224]}
        },
        "training": {
            "epochs": 1, # Minimal for testing
            "batch_size": 2,
            "optimizer": {"name": "Adam", "lr": 0.001},
            "scheduler": {"name": "StepLR", "step_size": 10, "gamma": 0.1},
            "pretrained": False # This might be overridden by model-specific needs
        },
        # Model config will be dynamically loaded from scanned files
        # "model": { ... }
    }
    dummy_config_path = Path("configs/experiment/dummy_all_models_kfold_config.yaml")
    dummy_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dummy_config_path, 'w') as f:
        yaml.dump(dummy_config_content, f)

    # Create dummy model configs for testing
    Path("configs/model/cnn").mkdir(parents=True, exist_ok=True)
    Path("configs/model/vit").mkdir(parents=True, exist_ok=True)
    with open("configs/model/cnn/test_cnn_model.yaml", 'w') as f:
        yaml.dump({"model_family": "CNN", "model_name": "TestCNN", "params": {"num_classes": 2}}, f)
    with open("configs/model/vit/test_vit_model.yaml", 'w') as f:
        yaml.dump({"model_family": "ViT", "model_name": "TestViT", "params": {"num_classes": 2, "patch_size": 16}}, f)
    with open("configs/model/cnn/base.yaml", 'w') as f: # Excluded
        yaml.dump({"base_config": True}, f)


    logger.info("Running AllModelsFullKFoldExperiment example...")
    try:
        # experiment = AllModelsFullKFoldExperiment(experiment_config_path=str(dummy_config_path))
        # experiment.run_experiment()
        logger.info("Example AllModelsFullKFoldExperiment run initiated (actual run commented out).")
        logger.info(f"To run, ensure KFoldExperiment and its dependencies are fully implemented and uncomment the lines above.")
        logger.info(f"Also ensure dummy data/dataset setup if KFoldExperiment.run_kfold_cv() attempts data loading.")

    except ImportError as e:
        logger.error(f"ImportError: {e}. Ensure all dependencies like PyTorch, Torchvision, etc., are installed.")
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}. Ensure all config paths are correct.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during example run: {e}", exc_info=True)
    finally:
        # Clean up dummy files
        # if dummy_config_path.exists():
        #     os.remove(dummy_config_path)
        # if Path("configs/model/cnn/test_cnn_model.yaml").exists():
        #     os.remove("configs/model/cnn/test_cnn_model.yaml")
        # if Path("configs/model/vit/test_vit_model.yaml").exists():
        #     os.remove("configs/model/vit/test_vit_model.yaml")
        # if Path("configs/model/cnn/base.yaml").exists():
        #    os.remove("configs/model/cnn/base.yaml")
        logger.info("Dummy files for example run can be manually reviewed or deleted from configs/experiment and configs/model.")