#!/usr/bin/env python3
"""
K-Fold Experiment Runner for CARS Thyroid Classification

This script automates k-fold cross-validation sweeps across multiple models.
This definitive version fixes the Hydra/OmegaConf struct error to ensure that
Vision Transformer configurations are handled correctly.
"""

import os
import sys
import traceback
import argparse
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Optional
import json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.data.dataset import CARSThyroidDataset, create_data_loaders
from src.data.quality_preprocessing import create_quality_aware_transform
from src.training.train_cnn import ThyroidCNNModule
from src.training.train_vit import ThyroidViTModule

console = Console()


class KFoldExperimentRunner:
    """Automates k-fold cross-validation sweeps and aggregates results."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir.absolute()
        self.all_results = {}
        GlobalHydra.instance().clear()
        
        console.print(Panel.fit(
            "[bold cyan]K-Fold Cross-Validation Experiment Runner[/bold cyan]",
            border_style="blue"
        ))
    
    def get_available_models(self) -> Dict[str, List[str]]:
        models = {'cnn': [], 'vit': []}
        for model_type in ['cnn', 'vit']:
            model_dir = self.config_dir / 'model' / model_type
            if model_dir.exists():
                for config_file in model_dir.glob('*.yaml'):
                    if not config_file.stem.startswith('__'):
                        models[model_type].append(config_file.stem)
        return models

    def create_experiment_config(self, model_name: str, model_type: str, fold: int, overrides: Dict) -> DictConfig:
        try:
            GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            
            training_config = "vit_standard" if model_type == 'vit' and (self.config_dir / 'training' / 'vit_standard.yaml').exists() else "standard"
            
            config_overrides = [f"model={model_type}/{model_name}", f"training={training_config}"]
            
            final_overrides = {**overrides, 'dataset.fold': fold}
            for key, value in final_overrides.items():
                config_overrides.append(f"{key}={value}")

            cfg = compose(config_name="config", overrides=config_overrides)

            # --- THE DEFINITIVE FIX: Un-structure the config to allow adding a new key ---
            if model_type == 'vit' and 'layer_decay' in cfg.training:
                # Temporarily make the model config mutable
                OmegaConf.set_struct(cfg.model, False)
                # Now we can safely add the key
                cfg.model.layer_decay = cfg.training.layer_decay
                # Re-enable the struct flag to prevent accidental changes later
                OmegaConf.set_struct(cfg.model, True)
            
            return cfg
        except Exception as e:
            console.print(f"[red]Error creating config for {model_name} (fold {fold}): {e}[/red]")
            raise
        finally:
            GlobalHydra.instance().clear()

    def run_single_fold(self, cfg: DictConfig) -> Dict:
        model_type = 'vit' if 'vit' in cfg.model.name or 'swin' in cfg.model.name else 'cnn'
        pl.seed_everything(cfg.seed, workers=True)

        quality_report_path = Path(cfg.paths.data_dir).parent / 'reports' / 'quality_report.json'
        quality_path = quality_report_path if quality_report_path.exists() and cfg.model.get('quality_aware', True) else None
        
        train_transform = create_quality_aware_transform(target_size=cfg.dataset.image_size, quality_report_path=quality_path, augmentation_level='medium', split='train')
        val_transform = create_quality_aware_transform(target_size=cfg.dataset.image_size, quality_report_path=quality_path, augmentation_level='none', split='val')

        dataloaders = create_data_loaders(
            root_dir=cfg.dataset.path, batch_size=cfg.training.batch_size, num_workers=cfg.dataset.num_workers,
            transform_train=train_transform, transform_val=val_transform,
            target_size=cfg.dataset.image_size, normalize=False, fold=cfg.dataset.fold
        )
        
        test_dataset = CARSThyroidDataset(
             root_dir=cfg.dataset.path, split='test', transform=val_transform, 
             target_size=cfg.dataset.image_size, normalize=False, fold=cfg.dataset.fold
        )
        test_loader = DataLoader(
             test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.dataset.num_workers
        )
        
        model_class = ThyroidViTModule if model_type == 'vit' else ThyroidCNNModule
        model = model_class(cfg)
        
        checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, verbose=False)
        early_stop_callback = EarlyStopping(monitor='val_acc', patience=cfg.training.early_stopping.patience, mode='max', verbose=False)
        
        trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
        trainer = pl.Trainer(
            **trainer_config,
            callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar(leave=True)],
            logger=False
        )
        
        trainer.fit(model, train_dataloaders=dataloaders['train'], val_dataloaders=dataloaders['val'])
        test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best', verbose=False)
        
        return {'test_acc': test_results[0].get('test_acc', 0.0)}

    def run_kfold_sweep(self, models_to_run: Dict[str, str], k_folds: int, overrides: Dict, quick_test: bool):
        self.all_results = {}
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), console=console) as progress:
            model_task = progress.add_task("[cyan]Running Model Sweep...", total=len(models_to_run))
            for model_name, model_type in models_to_run.items():
                progress.update(model_task, description=f"[cyan]Testing Model: [bold]{model_name}[/bold]")
                fold_accuracies = []
                fold_task = progress.add_task(f"  Folds for {model_name}", total=k_folds)
                for k in range(1, k_folds + 1):
                    progress.update(fold_task, description=f"  Running fold {k}/{k_folds} for {model_name}...")
                    try:
                        fold_overrides = overrides.copy()
                        if quick_test:
                            fold_overrides.update({'trainer.max_epochs': 2, 'trainer.limit_train_batches': 5, 'trainer.limit_val_batches': 2})
                        cfg = self.create_experiment_config(model_name, model_type, k, fold_overrides)
                        result = self.run_single_fold(cfg)
                        fold_accuracies.append(result['test_acc'])
                        console.print(f"  [green]✓ Fold {k} Test Accuracy: {result['test_acc']:.4f}[/green]")
                    except Exception as e:
                        console.print(f"  [red]✗ Fold {k} for {model_name} failed:[/red]")
                        console.print(f"    {e}")
                        console.print(traceback.format_exc())
                        fold_accuracies.append(np.nan)
                    progress.advance(fold_task)
                self.all_results[model_name] = self._aggregate_fold_results(fold_accuracies)
                progress.remove_task(fold_task)
                progress.advance(model_task)
    
    def _aggregate_fold_results(self, fold_accuracies: List[float]) -> Dict:
        valid_accuracies = [acc for acc in fold_accuracies if not np.isnan(acc)]
        if not valid_accuracies:
            return {'mean_acc': 0, 'std_dev': 0, 'best_acc': 0, 'worst_acc': 0, 'best_fold': 'N/A', 'worst_fold': 'N/A', 'runs': f"0/{len(fold_accuracies)}"}
        accuracies_np = np.array(valid_accuracies)
        original_indices = np.arange(1, len(fold_accuracies) + 1)
        valid_indices = original_indices[[not np.isnan(acc) for acc in fold_accuracies]]
        return {'mean_acc': np.mean(accuracies_np), 'std_dev': np.std(accuracies_np), 'best_acc': np.max(accuracies_np), 'worst_acc': np.min(accuracies_np), 'best_fold': valid_indices[np.argmax(accuracies_np)], 'worst_fold': valid_indices[np.argmin(accuracies_np)], 'runs': f"{len(valid_accuracies)}/{len(fold_accuracies)}"}

    def display_results(self):
        if not self.all_results:
            console.print("[yellow]No results to display.[/yellow]")
            return
        console.print("\n\n")
        table = Table(title="[bold]K-Fold Cross-Validation Summary[/bold]", show_header=True, header_style="bold magenta")
        for col in ["Model", "Mean Accuracy", "Std Dev", "Best Acc", "Worst Acc", "Best Fold", "Worst Fold", "Successful Runs"]: table.add_column(col)
        sorted_results = sorted(self.all_results.items(), key=lambda x: x[1].get('mean_acc', 0), reverse=True)
        for model_name, stats in sorted_results:
            table.add_row(model_name, f"{stats['mean_acc']:.2%}", f"{stats['std_dev']:.2%}", f"{stats['best_acc']:.2%}", f"{stats['worst_acc']:.2%}", str(stats['best_fold']), str(stats['worst_fold']), stats['runs'])
        console.print(table)


def main():
    parser = argparse.ArgumentParser(description='K-Fold Experiment Runner for CARS Thyroid Classification')
    parser.add_argument('--models', nargs='+', default=['all'], help='List of models to run (e.g., resnet50 swin_tiny). Default is "all".')
    parser.add_argument('--k-folds', '-k', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with limited epochs/batches.')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"

    runner = KFoldExperimentRunner(config_dir)
    available_models = runner.get_available_models()
    
    models_to_run = {}
    if 'all' in args.models:
        for model_name in available_models['cnn']: models_to_run[model_name] = 'cnn'
        for model_name in available_models['vit']: models_to_run[model_name] = 'vit'
    else:
        for model_name in args.models:
            if model_name in available_models['cnn']: models_to_run[model_name] = 'cnn'
            elif model_name in available_models['vit']: models_to_run[model_name] = 'vit'
            else: console.print(f"[yellow]Warning: Model '{model_name}' not found in configs. Skipping.[/yellow]")

    if not models_to_run:
        console.print("[red]No valid models selected to run. Exiting.[/red]")
        return
    
    console.print(f"[cyan]Models to be tested:[/cyan] {', '.join(models_to_run.keys())}")
    
    runner.run_kfold_sweep(models_to_run, args.k_folds, overrides={}, quick_test=args.quick_test)
    runner.display_results()
    
    console.print("\n[bold green]✓ K-Fold Sweep Complete![/bold green]")

if __name__ == "__main__":
    main()