"""
Configuration for experiments using Hydra.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


from hydra.core.config_store import ConfigStore


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


@dataclass
class AblationParameterConfig:
    """
    Defines a single parameter to be ablated.
    """
    path: str  # Path in the Hydra config, e.g., "model.optimizer.lr"
    values: List[Any]  # List of values to try for this parameter


@dataclass
class AblationConfig:
    """
    Configuration for ablation studies.
    """
    name_pattern: str = "ablation_run_{ablation_count}" # Pattern for naming ablation sub-runs
    # List of parameters and their values to iterate over for ablation.
    # Each dict should specify 'path' (e.g., 'model.lr') and 'values' (e.g., [0.01, 0.001])
    parameter_space: List[AblationParameterConfig] = field(default_factory=list)
    base_config_path: Optional[str] = None  # Path to a base Hydra config (e.g., 'experiment/base_model')
    # If true, this ablation study is the primary experiment type.
    is_primary_ablation_experiment: bool = False


@dataclass
class ExperimentConfig:
    """
    Base configuration for an experiment.
    Specific experiment types can extend this.
    """
    name: str = "base_experiment"
    description: Optional[str] = None
    output_dir: str = "outputs/${hydra.job.name}" # Hydra will resolve this
    seed: int = 42

    # Placeholder for model-specific configurations
    model: Any = field(default_factory=dict)

    # Placeholder for dataset-specific configurations
    dataset: Any = field(default_factory=dict)

    # Placeholder for trainer-specific configurations
    trainer: Any = field(default_factory=dict)

    # K-Fold specific configuration
    kfold: Optional[KFoldConfig] = None

    # Ablation study specific configuration
    ablation: Optional[AblationConfig] = None

    # Distillation specific configuration (optional)
    distillation: Optional[Dict[str, Any]] = None

    # Student model configuration, e.g. for distillation (optional)
    student_model: Optional[Dict[str, Any]] = None

    # Path to the experiment class to be instantiated (e.g., "src.experiment.standard_experiment.StandardExperiment")
    # This is used by the ExperimentManager if not a kfold or ablation primary experiment.
    experiment_class_path: Optional[str] = None

    # Additional custom parameters
    params: Dict[str, Any] = field(default_factory=dict)


# Register the configs with Hydra's ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_experiment_config_schema", node=ExperimentConfig) # Renamed for clarity
cs.store(name="kfold_config_schema", node=KFoldConfig)
cs.store(name="ablation_config_schema", node=AblationConfig)
cs.store(name="ablation_parameter_config_schema", node=AblationParameterConfig)