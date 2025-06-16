"""
Pydantic schemas for configuration validation.
Ensures type safety and validation.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from pydantic.fields import FieldInfo # For type hinting if needed, or ValidationInfo
from pydantic_core.core_schema import ValidationInfo # Correct import for ValidationInfo

# Example: Basic Python logging, can be enhanced
import logging
logger = logging.getLogger(__name__)

class PathsConfig(BaseModel):
    data_dir: str = "data/"
    output_dir: str = "outputs/"
    log_dir: str = "logs/"
    checkpoint_dir: str = "checkpoints/"

class BaseModelConfig(BaseModel):
    name: str = Field(..., description="Name of the model variant, e.g., resnet50, vit_base_patch16_224")
    architecture: str = Field(..., description="Model architecture type, e.g., cnn, vit")
    pretrained: bool = True
    num_classes: int = 2
    img_size: Optional[int] = None # Add img_size here for ViTs primarily
    
    # For any additional model-specific parameters not covered explicitly
    # For example, dropout_rate, hidden_dim for ResNet50, or img_size, patch_size for ViTs
    # These would typically be part of the specific model's config which inherits/extends this.
    # For a generic base, we might not include them or use a Dict.
    # Let's assume specific model configs will add their own fields.
    # Or, we can add a placeholder for extra params:
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class OptimizerParams(BaseModel):
    lr: float = 1e-4
    weight_decay: float = 1e-5
    # Add other common optimizer params like beta1, beta2, eps for Adam/AdamW if needed
    # name: str = "adamw" # Could also make optimizer choice explicit here

class SchedulerParams(BaseModel): # Optional, if you plan to use schedulers extensively
    name: Optional[str] = None # e.g., "cosine", "step_lr"
    eta_min: Optional[float] = None # For cosine scheduler
    step_size: Optional[int] = None # For StepLR
    gamma: Optional[float] = None # For StepLR
    # Add other common scheduler params

class TrainingConfig(BaseModel):
    seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    
    optimizer_params: OptimizerParams = Field(default_factory=OptimizerParams)
    scheduler_params: Optional[SchedulerParams] = None # Make scheduler optional

    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping_patience: Optional[int] = 10 # Optional early stopping
    
    save_top_k: int = 1
    save_last: bool = True
    log_every_n_steps: int = 50
    
    precision: Optional[str] = None # e.g., "16-mixed", "32-true"
    gradient_clip_val: Optional[float] = None

    # Example validator
    @field_validator('monitor_mode')
    def mode_must_be_min_or_max(cls, value: str) -> str: # Keep simple signature if info not needed
        if value not in ['min', 'max']:
            raise ValueError('monitor_mode must be "min" or "max"')
        return value

# This schema would represent the structure of a fully resolved Hydra config
# after all defaults and overrides are applied.
class HydraJobPaths(BaseModel): # For hydra.run.dir etc.
    run_dir: Optional[str] = None # Placeholder, actual path is dynamic
    sweep_dir: Optional[str] = None
    # Add other hydra paths if needed

class HydraConfig(BaseModel):
    run: Dict[str, Any] = Field(default_factory=lambda: {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"})
    sweep: Dict[str, Any] = Field(default_factory=lambda: {"dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}", "subdir": "${hydra.job.num}"})
    job_logging: Optional[str] = "colorlog" # Default from our base.yaml
    hydra_logging: Optional[str] = "colorlog" # Default from our base.yaml

class DatasetConfig(BaseModel):
    name: str = "cars_thyroid" # Default or specific name
    data_path: str = Field(..., description="Path to the root raw data directory (e.g., containing 'normal', 'cancerous' subfolders)")
    
    # Split configuration
    use_kfold: bool = False
    split_dir: Optional[str] = "data/splits" # Directory containing split JSON files
    fold: Optional[int] = None # Current fold number if use_kfold is True and split_file is not directly provided
    split_file_prefix: Optional[str] = "split_fold_" # Prefix for k-fold split files, e.g., "split_fold_"
    split_file: Optional[str] = None # Direct path to a specific split file (e.g., for a k-fold iteration)

    val_split_ratio: float = 0.2 # Ratio for validation set if creating splits on the fly (not from file)
    test_split_ratio: Optional[float] = None # Ratio for test set if creating splits on the fly
    random_seed: int = 42 # Seed for reproducible splits
    
    # Dataloader parameters (can also be part of TrainingConfig, but often grouped with dataset)
    # batch_size: int = 32 # Already in TrainingConfig, decide where it primarily lives. Let's keep it in TrainingConfig for now.
    # num_workers: int = 4 # Already in TrainingConfig

    # Image/Transform parameters (can be nested in a TransformConfig)
    img_size: int = 256 # Default based on sample_batch in conftest.py
    # For CARS dataset, it's likely grayscale.
    channels: int = 1 # 1 for grayscale, 3 for RGB
    mean: List[float] = Field(default_factory=lambda: [0.5]) # Default for 1 channel
    std: List[float] = Field(default_factory=lambda: [0.5])   # Default for 1 channel

    # Augmentation settings (can be more detailed)
    apply_augmentations: bool = False
    # augmentation_strength: Optional[str] = "medium" # Example

    # Quality preprocessing config (can be nested if complex)
    # Based on refactoring-guide.md line 374 for quality_preprocessing.py
    quality_preprocessing: bool = False # Whether to apply quality preprocessing
    extreme_dark_threshold: Optional[float] = 150.0
    low_contrast_threshold: Optional[float] = 20.0
    artifact_percentile: Optional[float] = 99.5

    # Example validator for mean/std based on channels
    @field_validator('mean', 'std', mode='before')
    def check_channels_mean_std(cls, v: Any, info: ValidationInfo) -> List[float]:
        channels = info.data.get('channels', 1) if info.data else 1
        field_name = info.field_name if info.field_name else "field"

        if not isinstance(v, list) or len(v) != channels:
            # Attempt to fix if a single float is given for single channel
            if channels == 1 and isinstance(v, (float, int)):
                return [float(v)]
            # Attempt to fix if a list of 3 floats is given but channels is 1 (take first)
            if channels == 1 and isinstance(v, list) and len(v) == 3:
                 logger.warning(f"Using first value of {field_name} for single channel image.")
                 return [v[0]]

            raise ValueError(f"{field_name} must be a list of {channels} floats, got {v}")
        return [float(x) for x in v] # Ensure elements are floats

class MainAppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    model: BaseModelConfig # This will be populated by a specific model config, e.g. ResNet50Config
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig # Dataset configuration is required

    project_name: str = "thyroid-vit-cnn-comparison"
    experiment_name: str = "default_experiment"
    
    hydra: Optional[HydraConfig] = Field(default_factory=HydraConfig) # Hydra's own config block

    # You might add specific model configs here for validation if not using OmegaConf's structured configs directly
    # For example:
    # resnet50: Optional[ResNet50ModelConfig] = None
    # vit_base: Optional[ViTBaseModelConfig] = None

    # This allows for additional, unstructured parameters at the top level if needed
    # extra: Dict[str, Any] = Field(default_factory=dict)

# Example of a more specific model config inheriting from BaseModelConfig
# class ResNetModelConfig(BaseModelConfig):
#     architecture: str = "cnn" # Could be fixed
#     dropout_rate: Optional[float] = 0.0
#     hidden_dim: Optional[int] = None
    # Add other ResNet specific fields

# class ViTModelConfig(BaseModelConfig):
#     architecture: str = "vit" # Could be fixed
#     patch_size: int = 16
#     img_size: int = 224
    # Add other ViT specific fields

if __name__ == '__main__':
    # Example usage for testing the schemas
    try:
        paths_data = {"data_dir": "/path/to/data"}
        paths_cfg = PathsConfig(**paths_data)
        logger.info(f"PathsConfig: {paths_cfg.model_dump_json(indent=2)}")

        model_data = {"name": "resnet50", "architecture": "cnn", "num_classes": 3}
        model_cfg = BaseModelConfig(**model_data)
        logger.info(f"BaseModelConfig: {model_cfg.model_dump_json(indent=2)}")

        opt_data = {"lr": 0.0005}
        opt_cfg = OptimizerParams(**opt_data)
        logger.info(f"OptimizerParams: {opt_cfg.model_dump_json(indent=2)}")

        train_data = {"epochs": 5, "optimizer_params": opt_data, "monitor_metric": "val_acc", "monitor_mode": "max"}
        train_cfg = TrainingConfig(**train_data)
        logger.info(f"TrainingConfig: {train_cfg.model_dump_json(indent=2)}")

        dataset_example_data = {
            "name": "example_cars_thyroid",
            "data_path": "/tmp/dummy_data", # Example path
            "img_size": 256,
            "channels": 1,
            "val_split_ratio": 0.2,
            "test_split_ratio": 0.1
            # Add other fields as necessary for DatasetConfig to be valid
        }
        dataset_cfg = DatasetConfig(**dataset_example_data)
        logger.info(f"DatasetConfig: {dataset_cfg.model_dump_json(indent=2)}")
        
        main_conf_data = {
            "paths": paths_data,
            "model": model_data,
            "training": train_data,
            "dataset": dataset_example_data, # Provide dataset config
            "project_name": "TestProject",
            "experiment_name": "TestExperiment001"
        }
        main_app_cfg = MainAppConfig(**main_conf_data)
        logger.info(f"MainAppConfig: {main_app_cfg.model_dump_json(indent=2)}")

        # Example of validation error
        # invalid_train_data = {"epochs": 5, "monitor_mode": "sideways"}
        # TrainingConfig(**invalid_train_data)

    except Exception as e:
        logger.error(f"Schema validation/usage error: {e}")