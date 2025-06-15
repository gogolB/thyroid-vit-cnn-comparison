import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, ListConfig # Import ListConfig
from typing import Optional, Union # Import Optional and Union

from src.models.registry import ModelRegistry
# from src.experiment.manager import ExperimentManager # Remains placeholder - Will define locally
from src.data.datamodule import ThyroidDataModule

# Import the LightningModules
from src.training.lightning_modules import ThyroidCNNModule, ThyroidViTModule, ThyroidDistillationModule

# Placeholder for ConfigManager if not using OmegaConf directly in type hints
ConfigManager = object # This is kept as per instructions

# Placeholder for ExperimentManager as it's not defined in src.experiment.manager yet
class ExperimentManager: # Placeholder
    def __init__(self):
        print("Dummy ExperimentManager initialized.")

import logging # Add import for logging

class ExperimentRunner:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.model_registry = ModelRegistry()
        self.experiment_manager = ExperimentManager() # Placeholder
        self.logger = logging.getLogger(__name__) # Initialize logger
        print("ExperimentRunner initialized.")

    def run(self, experiment_config: DictConfig):
        """
        Runs a single experiment based on the provided OmegaConf DictConfig.
        """
        print(f"Attempting to run experiment with config: \n{OmegaConf.to_yaml(experiment_config)}")

        # 1. Instantiate the LightningModule
        architecture = experiment_config.model.get('architecture', 'cnn').lower() # Default to cnn
        
        model_module: Optional[pl.LightningModule] = None # Corrected type hint
        if architecture == 'cnn':
            print("Instantiating ThyroidCNNModule...")
            model_module = ThyroidCNNModule(config=experiment_config)
        elif architecture == 'vit' or architecture == 'transformer':
            print("Instantiating ThyroidViTModule...")
            model_module = ThyroidViTModule(config=experiment_config)
        elif architecture == 'distillation': # You'll need a way to specify this in config
            print("Instantiating ThyroidDistillationModule...")
            model_module = ThyroidDistillationModule(config=experiment_config)
        else:
            raise ValueError(f"Unsupported model architecture '{architecture}' for selecting a LightningModule.")
        
        print(f"Successfully instantiated model_module: {type(model_module).__name__}")

        # 2. Instantiate PyTorch Lightning Trainer
        trainer_params = {}
        if hasattr(experiment_config, 'training'):
            trainer_params['max_epochs'] = experiment_config.training.get('epochs', 10)
            trainer_params['accelerator'] = experiment_config.training.get('accelerator', 'auto')
            trainer_params['devices'] = experiment_config.training.get('devices', 'auto')
            trainer_params['precision'] = experiment_config.training.get('precision', '32-true')
        else:
            print("Warning: experiment_config.training not found. Using default Trainer params.")
            trainer_params['max_epochs'] = 1

        print(f"Instantiating pl.Trainer with params: {trainer_params}...")
        trainer = pl.Trainer(**trainer_params)
        print("Successfully instantiated pl.Trainer.")

        # 3. Instantiate DataModule
        print("Instantiating ThyroidDataModule...")
        if not hasattr(experiment_config, 'dataset') or not hasattr(experiment_config, 'training'):
            raise ValueError("Experiment config must contain 'dataset' and 'training' sections for ThyroidDataModule.")
        
        data_module = ThyroidDataModule(
            dataset_config=experiment_config.dataset,
            training_config=experiment_config.training
        )
        print("Successfully instantiated ThyroidDataModule.")

        # Prepare data and setup for 'fit' stage
        # data_module.prepare_data() # prepare_data is often called by Lightning internally or if specific one-time setup is needed.
                                  # For CARSThyroidDataset, it seems data is loaded on demand.
        print("Setting up data_module for 'fit' stage...")
        data_module.setup(stage='fit')
        print("DataModule setup for 'fit' complete.")

        # 4. Run trainer.fit()
        print(f"Starting training for experiment: {experiment_config.get('experiment_name', 'N/A')}...")
        try:
            trainer.fit(model_module, datamodule=data_module)
            print("Training finished.")
        except Exception as e:
            print(f"Error during trainer.fit: {e}")
            # Potentially re-raise or log more detailed traceback
            raise

        # Optional: Add testing phase
        if experiment_config.training.get('run_test_after_fit', False): # Example flag to control testing
            print("Starting testing...")
            data_module.setup(stage='test')
            print("DataModule setup for 'test' complete.")
            try:
                trainer.test(model_module, datamodule=data_module)
                print("Testing finished.")
            except Exception as e:
                print(f"Error during trainer.test: {e}")
                # Potentially re-raise or log
        
        self.logger.info(f"Experiment '{experiment_config.get('experiment_name', 'N/A')}' processing finished.")

if __name__ == '__main__':
    print("Running ExperimentRunner main block for testing...")

    sample_conf_dict = {
        'paths': {
            'data_dir': "data/sample",
            'output_dir': "outputs/sample_run",
            'log_dir': "logs/sample_run",
            'checkpoint_dir': "checkpoints/sample_run"
        },
        'model': {
            'name': 'resnet18',
            'architecture': 'cnn', 
            'pretrained': False,
            'num_classes': 2
        },
        'training': {
            'epochs': 1,
            'batch_size': 2,
            'num_workers': 0,
            'accelerator': 'cpu',
            'devices': 1,
            'precision': '32-true',
            'log_every_n_steps': 1,
            'module_type': 'cnn',
            'loss': {
                '_target_': 'torch.nn.CrossEntropyLoss'
            },
            'optimizer_params': { # Renamed from optimizer to optimizer_params
                'lr': 0.0001,
                'weight_decay': 0.00001
            }
            # No scheduler_params added for now, to test optional scheduler
        },
        'dataset': {
            'name': 'cars_thyroid_default',  # Reflects dataset=default
            'data_path': "data/raw",
            'use_kfold': True,              # User override for k-fold
            'split_dir': "data/splits",
            'fold': 1,                      # User override for k-fold
            'val_split_ratio': 0.2,
            'test_split_ratio': None,
            'img_size': 256,
            'channels': 1,
            'mean': [0.5],
            'std': [0.5],
            'apply_augmentations': False,
            'quality_preprocessing': False,
            'extreme_dark_threshold': 150.0,
            'low_contrast_threshold': 20.0,
            'artifact_percentile': 99.5
        },
        'project_name': "test_project",
        'experiment_name': "test_experiment_runner_main"
    }
    base_config = OmegaConf.create(sample_conf_dict)
    
    # Parse CLI arguments
    cli_conf = OmegaConf.from_cli()
    
    # Merge CLI arguments into the base config
    # CLI arguments will override the defaults in sample_conf_dict
    experiment_config = OmegaConf.merge(base_config, cli_conf) # Initial merge

    # Handle the case where CLI `model=model_name_str` makes `experiment_config.model` a string
    if isinstance(experiment_config.model, str):
        model_name_from_cli = experiment_config.model
        # Reconstruct experiment_config.model as a DictConfig
        # Start with a copy of the original base_config.model
        new_model_conf = OmegaConf.create(OmegaConf.to_container(base_config.model, resolve=True))
        new_model_conf.name = model_name_from_cli # Set the name from CLI
        experiment_config.model = new_model_conf # Assign the new DictConfig back
    
    # Now, experiment_config.model is guaranteed to be a DictConfig.
    # Proceed with adjustments based on experiment_config.model.name.
    # This logic assumes that if 'model.name' is specified (either from base or CLI),
    # 'model.architecture' and 'training.module_type' should align.
    
    current_model_name = experiment_config.model.name

    if "inception_v3" in current_model_name:
        experiment_config.model.architecture = "cnn"
        experiment_config.training.module_type = "cnn"
        # Ensure dataset img_size is appropriate for InceptionV3
        if experiment_config.dataset.img_size != 299: # InceptionV3 default input size
            print(f"Adjusting dataset.img_size to 299 for InceptionV3 (was {experiment_config.dataset.img_size})")
            experiment_config.dataset.img_size = 299
    elif "vit_" in current_model_name or "swin_" in current_model_name or "deit_" in current_model_name: # Add other ViT families if needed
        experiment_config.model.architecture = "vit" # or "transformer" if you use that key
        experiment_config.training.module_type = "vit"
        # Default ViT image size, can be overridden by specific model configs or CLI
        default_vit_img_size = 224
        if experiment_config.model.name == "swin_large_blackwell": # Example specific override
            default_vit_img_size = 384 # Or whatever it needs
        elif experiment_config.model.name == "swin_target_944":
             default_vit_img_size = 224 # Example, adjust as needed

        if experiment_config.dataset.img_size != default_vit_img_size:
             print(f"Adjusting dataset.img_size to {default_vit_img_size} for {current_model_name} (was {experiment_config.dataset.img_size})")
             experiment_config.dataset.img_size = default_vit_img_size
    # Add more elif blocks for other model families (e.g., resnet, efficientnet) if their
    # architecture or module_type needs to be explicitly set when `model.name` is given via CLI.
    # For many CNNs, the default 'cnn' architecture/module_type from base_config might be fine.

    class DummyConfigManager:
        pass
    
    config_manager = DummyConfigManager()
    
    runner = ExperimentRunner(config_manager=config_manager)
    
    try:
        # Run the primary experiment, now potentially overridden by CLI args
        if not isinstance(experiment_config, DictConfig):
            print(f"Warning: experiment_config is not a DictConfig, attempting to convert. Type: {type(experiment_config)}")
            # Attempt to create a DictConfig if it's a compatible type like a dict
            try:
                experiment_config = OmegaConf.create(experiment_config)
            except Exception as e_conv:
                print(f"Error converting experiment_config to DictConfig: {e_conv}")
                raise TypeError(f"experiment_config must be a DictConfig, got {type(experiment_config)}")
        
        runner.run(experiment_config=experiment_config)
    except Exception as e:
        print(f"Error during ExperimentRunner.run: {e}")
        import traceback
        traceback.print_exc()

# The InceptionV3 timm test snippet and the subsequent hardcoded ViT test run have been removed.
# The script now only runs the experiment defined by the (potentially CLI-overridden) experiment_config.