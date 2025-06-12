"""
CNN Ensemble Model for Thyroid Classification
Combines ResNet50, EfficientNet-B0, and DenseNet121
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
import numpy as np
from pathlib import Path

from rich.console import Console
console = Console()


class ThyroidCNNEnsemble(pl.LightningModule):
    """
    Ensemble model combining top 3 CNN architectures.
    Supports both averaging and weighted voting strategies.
    """
    
    def __init__(
        self,
        checkpoint_paths: Dict[str, str],
        num_classes: int = 2,
        ensemble_method: str = 'weighted_avg',  # 'avg', 'weighted_avg', 'voting'
        weights: Optional[List[float]] = None,
        temperature: float = 1.0,
        device_type: str = 'cuda'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model names and their test accuracies for default weights
        self.model_info = {
            'resnet50': {'accuracy': 0.9118, 'params': 23.5e6},
            'efficientnet_b0': {'accuracy': 0.8971, 'params': 4.0e6},
            'densenet121': {'accuracy': 0.8824, 'params': 7.8e6}
        }
        
        # Set weights based on performance if not provided
        if weights is None:
            if ensemble_method == 'weighted_avg':
                # Weight proportional to accuracy
                accuracies = [info['accuracy'] for info in self.model_info.values()]
                total = sum(accuracies)
                self.weights = [acc/total for acc in accuracies]
            else:
                # Equal weights for simple averaging
                self.weights = [1/3, 1/3, 1/3]
        else:
            self.weights = weights
            
        self.temperature = temperature
        self.ensemble_method = ensemble_method
        
        # Load models
        self.models = nn.ModuleDict()
        self._load_models(checkpoint_paths, device_type)
        
        # Freeze all models (inference only)
        self.freeze()
        
        # Metrics
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_auc = AUROC(task='multiclass', num_classes=num_classes)
        self.test_f1 = F1Score(task='multiclass', num_classes=num_classes)
        
        console.print(f"[green]Ensemble created with method: {ensemble_method}[/green]")
        console.print(f"Weights: ResNet50={self.weights[0]:.3f}, "
                     f"EfficientNet-B0={self.weights[1]:.3f}, "
                     f"DenseNet121={self.weights[2]:.3f}")
    
    def _load_models(self, checkpoint_paths: Dict[str, str], device_type: str):
        """Load individual models from checkpoints."""
        
        # Import model classes
        from src.models.cnn.resnet import ResNet18Lightning
        from src.training.train_cnn import ThyroidCNNModule
        
        for model_name, ckpt_path in checkpoint_paths.items():
            console.print(f"Loading {model_name} from {ckpt_path}")
            
            try:
                # Load the Lightning module
                if 'resnet' in model_name:
                    model = ResNet18Lightning.load_from_checkpoint(
                        ckpt_path,
                        map_location=device_type
                    )
                else:
                    # For EfficientNet and DenseNet
                    model = ThyroidCNNModule.load_from_checkpoint(
                        ckpt_path,
                        map_location=device_type
                    )
                
                # Extract the actual model
                if hasattr(model, 'model'):
                    self.models[model_name] = model.model
                else:
                    self.models[model_name] = model
                    
                console.print(f"[green]✓ Loaded {model_name}[/green]")
                
            except Exception as e:
                console.print(f"[red]Error loading {model_name}: {e}[/red]")
                raise
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            Dictionary with 'logits', 'probs', and individual model outputs
        """
        outputs = {}
        all_logits = []
        all_probs = []
        
        # Get predictions from each model
        for name, model in self.models.items():
            with torch.no_grad():
                logits = model(x)
                
                # Handle different output formats
                if isinstance(logits, dict):
                    logits = logits['logits']
                
                # Apply temperature scaling
                logits = logits / self.temperature
                probs = F.softmax(logits, dim=1)
                
                outputs[f'{name}_logits'] = logits
                outputs[f'{name}_probs'] = probs
                
                all_logits.append(logits)
                all_probs.append(probs)
        
        # Stack predictions
        all_logits = torch.stack(all_logits, dim=0)  # [n_models, batch, classes]
        all_probs = torch.stack(all_probs, dim=0)
        
        # Ensemble predictions
        if self.ensemble_method == 'avg':
            # Simple averaging
            ensemble_logits = all_logits.mean(dim=0)
            ensemble_probs = all_probs.mean(dim=0)
            
        elif self.ensemble_method == 'weighted_avg':
            # Weighted averaging
            weights = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
            ensemble_logits = (all_logits * weights).sum(dim=0)
            ensemble_probs = (all_probs * weights).sum(dim=0)
            
        elif self.ensemble_method == 'voting':
            # Majority voting
            predictions = all_probs.argmax(dim=2)  # [n_models, batch]
            ensemble_preds = []
            
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                # Weighted voting
                vote_counts = torch.zeros(self.hparams.num_classes, device=x.device)
                for j, vote in enumerate(votes):
                    vote_counts[vote] += self.weights[j]
                ensemble_preds.append(vote_counts.argmax())
            
            ensemble_preds = torch.stack(ensemble_preds)
            # Convert back to logits (one-hot style)
            ensemble_logits = F.one_hot(ensemble_preds, self.hparams.num_classes).float()
            ensemble_probs = ensemble_logits  # Already like probabilities
        
        outputs['logits'] = ensemble_logits
        outputs['probs'] = ensemble_probs
        outputs['all_logits'] = all_logits
        outputs['all_probs'] = all_probs
        
        return outputs
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step with detailed metrics."""
        x, y = batch
        
        outputs = self(x)
        ensemble_probs = outputs['probs']
        ensemble_preds = ensemble_probs.argmax(dim=1)
        
        # Metrics
        self.test_acc(ensemble_preds, y)
        if self.hparams.num_classes == 2:
            self.test_auc(ensemble_probs[:, 1], y)
        else:
            self.test_auc(ensemble_probs, y)
        self.test_f1(ensemble_preds, y)
        
        # Log individual model performances
        for name in self.models.keys():
            model_probs = outputs[f'{name}_probs']
            model_preds = model_probs.argmax(dim=1)
            acc = (model_preds == y).float().mean()
            self.log(f'test/{name}_acc', acc)
        
        # Log ensemble metrics
        self.log('test/ensemble_acc', self.test_acc)
        self.log('test/ensemble_auc', self.test_auc)
        self.log('test/ensemble_f1', self.test_f1)
        
        return {
            'ensemble_preds': ensemble_preds,
            'true_labels': y,
            'all_probs': outputs['all_probs']
        }
    
    def on_test_epoch_end(self):
        """Summary of test results."""
        acc = self.test_acc.compute()
        auc = self.test_auc.compute()
        f1 = self.test_f1.compute()
        
        console.print("\n[bold green]Ensemble Test Results:[/bold green]")
        console.print(f"  Accuracy: {acc:.4f}")
        console.print(f"  AUC: {auc:.4f}")
        console.print(f"  F1: {f1:.4f}")
        
        # Reset metrics
        self.test_acc.reset()
        self.test_auc.reset()
        self.test_f1.reset()
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Returns:
            Dictionary with predictions, probabilities, and uncertainty
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            
            # Calculate uncertainty as disagreement between models
            all_probs = outputs['all_probs']
            uncertainty = all_probs.std(dim=0).mean(dim=1)  # Average std across classes
            
            # Prediction confidence
            ensemble_probs = outputs['probs']
            confidence = ensemble_probs.max(dim=1)[0]
            
            predictions = ensemble_probs.argmax(dim=1)
            
            return {
                'predictions': predictions,
                'probabilities': ensemble_probs,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'all_model_probs': all_probs
            }


def create_ensemble_from_best_models(
    checkpoint_dir: Path,
    model_names: List[str] = ['resnet50', 'efficientnet_b0', 'densenet121'],
    ensemble_method: str = 'weighted_avg'
) -> ThyroidCNNEnsemble:
    """
    Helper function to create ensemble from best checkpoints.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints
        model_names: List of model names to include
        ensemble_method: Ensemble strategy
        
    Returns:
        Configured ensemble model
    """
    # Find best checkpoints for each model
    checkpoint_paths = {}
    
    for model_name in model_names:
        # Look for checkpoints matching pattern
        pattern = f"{model_name}*.ckpt"
        checkpoints = list(checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint found for {model_name}")
        
        # Sort by modification time and take the latest
        best_ckpt = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
        checkpoint_paths[model_name] = str(best_ckpt)
        
        console.print(f"Found checkpoint for {model_name}: {best_ckpt.name}")
    
    # Create ensemble
    ensemble = ThyroidCNNEnsemble(
        checkpoint_paths=checkpoint_paths,
        ensemble_method=ensemble_method
    )
    
    return ensemble


# Example usage
if __name__ == "__main__":
    # Example checkpoint paths
    checkpoint_paths = {
        'resnet50': 'checkpoints/resnet50-epoch=10-val_acc=0.9412.ckpt',
        'efficientnet_b0': 'checkpoints/efficientnet_b0-epoch=15-val_acc=0.9412.ckpt',
        'densenet121': 'checkpoints/densenet121-epoch=12-val_acc=0.8971.ckpt'
    }
    
    # Create ensemble
    ensemble = ThyroidCNNEnsemble(
        checkpoint_paths=checkpoint_paths,
        ensemble_method='weighted_avg'
    )
    
    # Test on dummy data
    dummy_input = torch.randn(4, 1, 256, 256)
    outputs = ensemble(dummy_input)
    
    console.print(f"Ensemble output shape: {outputs['logits'].shape}")
    console.print(f"Individual model outputs available: {[k for k in outputs.keys() if '_' in k]}")