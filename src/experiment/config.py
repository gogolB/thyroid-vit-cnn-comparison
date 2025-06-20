"""
Configuration for experiments using Hydra.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


@dataclass
class KFoldConfig:
    """
    Configuration for K-Fold cross-validation.
    """
    num_folds: int = 5
    split_dir: str = "data/splits"
    split_file_prefix: str = "split_fold_"
    # Indicates if this k-fold process is part of a larger experiment
    # or if it's the main experiment type.
    is_primary_kfold_experiment: bool = False
    # Whether to save models for each fold
    save_fold_models: bool = True
    # Whether to log metrics for each fold
    log_fold_metrics: bool = True


@dataclass
class ExperimentConfig:
    """
    Base configuration for an experiment.
    Specific experiment types can extend this.
    """
    name: str = "base_experiment"
    description: Optional[str] = None
    output_dir: str = "outputs/"  # Hydra will resolve this
    seed: int = 42

    # Placeholder for model-specific configurations
    model: Any = field(default_factory=dict)

    # Placeholder for dataset-specific configurations
    dataset: Any = field(default_factory=dict)

    # Placeholder for trainer-specific configurations
    trainer: Any = field(default_factory=dict)

    # Placeholder for training content configurations
    training_content: Any = field(default_factory=dict)

    # K-Fold specific configuration
    kfold: Optional[KFoldConfig] = None

    # Distillation specific configuration (optional)
    distillation: Optional[Dict[str, Any]] = None

    # Student model configuration (optional)
    student_model: Optional[Dict[str, Any]] = None

    # Path to the experiment class to be instantiated
    experiment_class_path: Optional[str] = None

    # Additional custom parameters
    params: Dict[str, Any] = field(default极factory=dict)


# Register the configs with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_experiment_config_schema", node=ExperimentConfig)
cs.store(name="kfold_config_schema", node=KFoldConfig)


def load_config(config_path: str) -> DictConfig:
    """Load a configuration file and resolve it.

    Args:
        config_path: Path to the configuration file.

    Returns:
        DictConfig: The loaded configuration.
    """
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    return config