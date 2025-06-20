# --- START: Fully Instrumented Debugging Script ---

from ast import mod
import glob
import os
import json
from tabnanny import check
from matplotlib.font_manager import weight_dict
import torch
import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    f1_score, confusion_matrix
)

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config.schemas import DatasetConfig

from src.data.datamodule import ThyroidDataModule
from omegaconf import OmegaConf

from rich.console import Console

from torch.utils.data import DataLoader

# Setup console for rich output
console = Console()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define model types and paths
MODEL_TYPES = ['densenet169', 'vit_small', 'vit_tiny']
CHECKPOINT_DIR = "checkpoints"
SPLIT_DIR = "data/splits"
OUTPUT_PATH = (
    "reports/results/"
    "kfold_summary_ensemble_densenet169_vit_small_vit_tiny.json"
)
NUM_CLASSES = 2
FOLDS = range(1, 8)  # 7 folds

# Load configs for datamodule
dataset_config = OmegaConf.load("configs/dataset/default.yaml")
training_config = OmegaConf.load("configs/training/base.yaml")


def load_model_for_fold(
    model_type: str,
    fold: int,
    device: torch.device
) -> torch.nn.Module:
    """Load trained model for specific fold"""
    checkpoint_pattern = os.path.join(
        CHECKPOINT_DIR, model_type, f"fold_{fold}", "epoch=*.ckpt"
    )
    
    from src.models.cnn.densenet import DenseNet
    from src.models.cnn.resnet import ResNet
    from src.models.vit.vision_transformer import VisionTransformer
    from src.models.vit.swin import SwinTransformer 
    
    checkpoint_paths = glob.glob(checkpoint_pattern)
    
    # --- DIAGNOSTIC CHECK 1: Verify Checkpoint Loading ---
    if not checkpoint_paths:
        raise FileNotFoundError(f"FATAL: No checkpoint found for pattern: {checkpoint_pattern}")
    checkpoint_path = checkpoint_paths[0]
    console.print(f"[yellow]>>> DEBUG: For model_type '{model_type}', found checkpoint: {os.path.basename(checkpoint_path)}[/yellow]")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model_type.startswith('swin'):
        cfg = {'name': "swin_tiny"}
        cfg = OmegaConf.create(cfg)
        model = SwinTransformer(config=cfg)
    elif model_type.startswith('vit'):
        cfg = {'name': model_type}
        cfg = OmegaConf.create(cfg)
        model = VisionTransformer(config=cfg)
    elif model_type.startswith('densenet'):
        cfg = {'name': "densenet169"}
        cfg = OmegaConf.create(cfg)
        model = DenseNet(config=cfg)
    elif model_type.startswith('resnet'):
        cfg = {'name': "resnet18"}
        cfg = OmegaConf.create(cfg)
        model = ResNet(config=cfg)
    
    for key in list(checkpoint['state_dict'].keys()):
        if key.startswith('model.model.'):
            new_key = key.replace('model.model.', 'model.')
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
    
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model.to(device)
    

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray
) -> dict:
    """Compute all required metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'auroc': roc_auc_score(y_true, y_probs[:, 1]),
        'f1': f1_score(y_true, y_pred),
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
    }
    return metrics


def evaluate_ensemble(
    models: list[torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    weights: torch.Tensor, 
    device: torch.device
) -> dict:
    """Evaluate a true weighted ensemble model on a dataset."""
    all_preds, all_probs, all_labels = [], [], []
    
    assert len(models) == len(weights), "Number of models must match number of weights."

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            model_probs = [torch.softmax(model(images), dim=1) for model in models]
            stacked_probs = torch.stack(model_probs, dim=0)
            reshaped_weights = weights.view(-1, 1, 1)
            weighted_sum_probs = torch.sum(stacked_probs * reshaped_weights, dim=0)
            _, preds = torch.max(weighted_sum_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(weighted_sum_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def main():
    console.print("[bold red]>>> RUNNING FINAL DIAGNOSTIC SCRIPT (v.4) <<<[/bold red]")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    logger.info(f"Using device: {device}")
    
    results = {'folds': {}, 'average': {}, 'std': {}}
    metrics_list = ['accuracy', 'sensitivity', 'specificity', 'auroc', 'f1', 'ppv', 'npv']
    fold_metrics = {metric: [] for metric in metrics_list}
    
    MODEL_WEIGHTS = {'densenet169': 0.50, 'vit_small': 0.25, 'vit_tiny': 0.25}
    MODEL_TYPES = list(MODEL_WEIGHTS.keys())
    weights_tensor = torch.tensor(list(MODEL_WEIGHTS.values()), device=device)
    
    for fold in FOLDS:
        logger.info(f"Processing fold {fold}")
        
        models = []
        for model_type in MODEL_TYPES:
            logger.info(f"Loading {model_type} for fold {fold}")
            model = load_model_for_fold(model_type, fold, device)
            models.append(model)
        
        # --- DIAGNOSTIC CHECK 2: Verify Models in Ensemble ---
        console.print(f"[bold magenta]>>> DEBUG: VERIFYING MODELS LOADED FOR FOLD {fold} <<<[/bold magenta]")
        for i, m in enumerate(models):
            # Accessing the config name is a good way to verify the instantiated model variant
            model_variant_name = m.config.name if hasattr(m, 'config') else 'N/A'
            console.print(f"Model {i+1}: Class={m.__class__.__name__}, Expected Variant='{MODEL_TYPES[i]}', Actual Variant='{model_variant_name}'")
        
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        transform = create_quality_aware_transform(target_size=224, split='test')
        
        dataset_config = DatasetConfig(
            data_path='data/processed',
            split='test',
            transform=transform,
            target_size=224,
            use_kfold=True,
            fold=fold
        )
        dataset = CARSThyroidDataset(config=dataset_config, mode="test")
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        metrics = evaluate_ensemble(models, test_loader, weights_tensor, device)
        results['folds'][f'fold_{fold}'] = metrics
        
        for metric in metrics_list:
            fold_metrics[metric].append(metrics[metric])
    
    for metric, values in fold_metrics.items():
        results['average'][metric] = np.mean(values)
        results['std'][metric] = np.std(values)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Ensemble evaluation complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

# --- END: Fully Instrumented Debugging Script ---