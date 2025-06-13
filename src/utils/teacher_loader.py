"""
Teacher Model Loading Utility for Knowledge Distillation
Handles loading pre-trained CNN and ViT models as teachers
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf
import warnings

from src.training.train_cnn import ThyroidCNNModule
from src.training.train_vit import ThyroidViTModule
from src.utils.device import get_device


class TeacherModelLoader:
    """Utility class for loading and managing teacher models for distillation"""
    
    @staticmethod
    def load_teacher_from_checkpoint(
        checkpoint_path: Union[str, Path],
        model_type: str = 'cnn',
        device: Optional[str] = None,
        eval_mode: bool = True,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Load a pre-trained teacher model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_type: Type of model ('cnn' or 'vit')
            device: Device to load model on (auto-detect if None)
            eval_mode: Whether to set model to eval mode
            verbose: Print loading information
            
        Returns:
            Tuple of (model, metrics_dict) where metrics_dict contains
            the model's performance metrics from training
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Auto-detect device
        if device is None:
            device = get_device()
        
        if verbose:
            print(f"Loading teacher model from: {checkpoint_path}")
            print(f"Model type: {model_type}")
            print(f"Device: {device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract metrics if available
        metrics = {}
        if 'callbacks' in checkpoint:
            # Extract best validation accuracy
            for callback in checkpoint.get('callbacks', {}).values():
                if isinstance(callback, dict) and 'best_model_score' in callback:
                    metrics['val_acc'] = float(callback['best_model_score'])
                    break
        
        # Try to get test accuracy from checkpoint
        if 'test_acc' in checkpoint:
            metrics['test_acc'] = float(checkpoint['test_acc'])
        
        # Load the appropriate module
        try:
            if model_type.lower() == 'cnn':
                # Load CNN model
                lightning_module = ThyroidCNNModule.load_from_checkpoint(
                    checkpoint_path,
                    map_location=device,
                    strict=False  # Allow minor mismatches
                )
                # Extract the underlying model
                model = lightning_module.model
                
            elif model_type.lower() == 'vit':
                # Load ViT model
                lightning_module = ThyroidViTModule.load_from_checkpoint(
                    checkpoint_path,
                    map_location=device,
                    strict=False
                )
                # Extract the underlying model
                model = lightning_module.model
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Move to correct device
            model = model.to(device)
            
            # Set eval mode if requested
            if eval_mode:
                model.eval()
                # Disable dropout and batch norm updates
                for module in model.modules():
                    if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
                        module.eval()
            
            if verbose:
                param_count = sum(p.numel() for p in model.parameters())
                print(f"Successfully loaded teacher model")
                print(f"Parameters: {param_count:,}")
                if metrics:
                    print(f"Performance metrics: {metrics}")
            
            return model, metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to load teacher model: {str(e)}")
    
    @staticmethod
    def load_ensemble_teachers(
        checkpoint_paths: List[Union[str, Path]],
        model_types: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        device: Optional[str] = None,
        eval_mode: bool = True,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Load multiple teacher models and create an ensemble.
        
        Args:
            checkpoint_paths: List of paths to teacher checkpoints
            model_types: List of model types (auto-detect if None)
            weights: Weights for ensemble averaging (uniform if None)
            device: Device to load models on
            eval_mode: Whether to set models to eval mode
            verbose: Print loading information
            
        Returns:
            Tuple of (EnsembleTeacher, combined_metrics)
        """
        if len(checkpoint_paths) == 0:
            raise ValueError("No checkpoint paths provided")
        
        # Auto-detect model types if not provided
        if model_types is None:
            model_types = []
            for path in checkpoint_paths:
                path_str = str(path).lower()
                if any(cnn in path_str for cnn in ['resnet', 'efficientnet', 'densenet', 'inception']):
                    model_types.append('cnn')
                else:
                    model_types.append('vit')
        
        # Load all teachers
        teachers = []
        all_metrics = {}
        
        for i, (ckpt_path, model_type) in enumerate(zip(checkpoint_paths, model_types)):
            if verbose:
                print(f"\nLoading teacher {i+1}/{len(checkpoint_paths)}")
            
            model, metrics = TeacherModelLoader.load_teacher_from_checkpoint(
                ckpt_path, model_type, device, eval_mode, verbose
            )
            teachers.append(model)
            
            # Store metrics with model identifier
            model_name = Path(ckpt_path).stem.split('-')[0]
            for key, value in metrics.items():
                all_metrics[f"{model_name}_{key}"] = value
        
        # Create ensemble wrapper
        ensemble = EnsembleTeacher(teachers, weights, device)
        
        if eval_mode:
            ensemble.eval()
        
        return ensemble, all_metrics
    
    @staticmethod
    def create_teacher_from_config(
        config: DictConfig,
        device: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Create teacher model(s) from Hydra configuration.
        
        Args:
            config: Hydra configuration with distillation section
            device: Device to load model on
            verbose: Print loading information
            
        Returns:
            Tuple of (model, metrics)
        """
        dist_cfg = config.distillation
        
        # Check if ensemble
        if dist_cfg.get('ensemble_teachers', False):
            # Load ensemble
            return TeacherModelLoader.load_ensemble_teachers(
                checkpoint_paths=dist_cfg.teacher_checkpoints,
                model_types=dist_cfg.get('teacher_model_types'),
                weights=dist_cfg.get('teacher_weights'),
                device=device,
                eval_mode=dist_cfg.get('teacher_eval_mode', True),
                verbose=verbose
            )
        else:
            # Load single teacher
            return TeacherModelLoader.load_teacher_from_checkpoint(
                checkpoint_path=dist_cfg.teacher_checkpoint,
                model_type=dist_cfg.get('teacher_model_type', 'cnn'),
                device=device,
                eval_mode=dist_cfg.get('teacher_eval_mode', True),
                verbose=verbose
            )


class EnsembleTeacher(nn.Module):
    """
    Ensemble wrapper for multiple teacher models.
    Combines predictions using weighted averaging.
    """
    
    def __init__(
        self,
        teachers: List[nn.Module],
        weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.num_teachers = len(teachers)
        
        # Set weights
        if weights is None:
            weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.weights = torch.tensor(weights, device=device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        Returns weighted average of teacher logits.
        """
        # Collect predictions from all teachers
        all_logits = []
        
        with torch.no_grad():  # Teachers don't need gradients
            for teacher in self.teachers:
                logits = teacher(x)
                all_logits.append(logits)
        
        # Stack and weight
        stacked_logits = torch.stack(all_logits, dim=0)  # [n_teachers, batch, classes]
        weighted_logits = stacked_logits * self.weights.view(-1, 1, 1)
        
        # Return weighted average
        return weighted_logits.sum(dim=0)
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from each teacher individually."""
        predictions = []
        with torch.no_grad():
            for teacher in self.teachers:
                predictions.append(teacher(x))
        return predictions


# Utility functions for common use cases
def load_best_teacher(
    model_name: str,
    checkpoint_dir: Union[str, Path] = "checkpoints",
    verbose: bool = True
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Load the best checkpoint for a given model name.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50')
        checkpoint_dir: Directory containing checkpoints
        verbose: Print loading information
        
    Returns:
        Tuple of (model, metrics)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find checkpoints for this model
    pattern = f"{model_name}*.ckpt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {model_name} in {checkpoint_dir}")
    
    # Sort by modification time and take the latest
    # Could also parse validation accuracy from filename
    best_checkpoint = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    
    if verbose:
        print(f"Found best checkpoint: {best_checkpoint}")
    
    # Determine model type
    model_type = 'vit' if any(vit in model_name for vit in ['vit', 'deit', 'swin']) else 'cnn'
    
    return TeacherModelLoader.load_teacher_from_checkpoint(
        best_checkpoint,
        model_type=model_type,
        verbose=verbose
    )