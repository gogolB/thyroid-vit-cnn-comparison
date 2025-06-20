#!/usr/bin/env python3
"""
Standalone script for DenseNet169 to DeiT-Tiny knowledge distillation
with k-fold cross-validation
"""

import os
import json
import copy
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import OmegaConf
from src.data.datamodule import ThyroidDataModule
from src.training.lightning_modules import ThyroidDistillationModule


def run_distillation_kfold():
    """Run k-fold distillation experiment and save aggregated results"""
    # Load base configuration
    config_path = "configs/experiment/densenet169_distill_deit_tiny_kfold.yaml"
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    
    # Experiment parameters
    num_folds = 7
    base_teacher_ckpt = "checkpoints/densenet169/fold_{fold}/last.ckpt"
    base_split_file = "data/splits/split_fold_{fold}.json"
    
    # Collect metrics across folds
    metrics = defaultdict(list)
    
    for fold in range(1, num_folds + 1):
        print(f"\n{'='*40}")
        print(f"Starting fold {fold}/{num_folds}")
        print(f"{'='*40}")
        
        try:
            # Create config copy for this fold
            fold_config = copy.deepcopy(config)
            
            # Set fold-specific paths
            fold_config.split_file = base_split_file.format(fold=fold)
            fold_config.teacher_checkpoint = (
                base_teacher_ckpt.format(fold=fold))
            
            # Set seed for reproducibility
            if fold_config.get("seed"):
                pl.seed_everything(fold_config.seed)
                
            from src.config.schemas import DatasetConfig
            from src.data.quality_preprocessing import create_quality_aware_transform
            
            transform = create_quality_aware_transform(target_size=224, split='train')
            
            dataset_config = DatasetConfig(
                data_path='data/processed',
                split='test',
                transform=transform,
                target_size=224,
                use_kfold=True,
                fold=fold
            )
            
            # Initialize data module with required config
            data_module = ThyroidDataModule(
                training_config=fold_config.training,
                dataset_config=dataset_config,
            )
            
            trainer = Trainer(**fold_config.trainer)

            # Initialize distillation model
            model = ThyroidDistillationModule(fold_config, trainer=trainer)
            
            # Initialize trainer            
            # Train model
            trainer.fit(model, data_module)
            
            # Test model
            test_results = trainer.test(model, datamodule=data_module)
            
            # Collect metrics
            if test_results:
                for key, value in test_results[0].items():
                    if key.startswith("test_"):
                        metrics[key].append(value)
            
            print(f"Completed fold {fold}/{num_folds}")
        
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            raise
    
    # Aggregate metrics across folds
    summary = {}
    for metric, values in metrics.items():
        summary[metric] = {
            "mean": torch.tensor(values).mean().item(),
            "std": torch.tensor(values).std().item()
        }
    
    # Save results
    output_path = "reports/results/kfold_summary_densenet169_deit_tiny.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to {output_path}")
    return summary


if __name__ == "__main__":
    run_distillation_kfold()