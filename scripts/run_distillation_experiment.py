#!/usr/bin/env python3
"""
Run knowledge distillation experiments for Vision Transformers.
Supports various teacher-student combinations and distillation strategies.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import project modules
from src.data.dataset import create_data_loaders
from src.data.quality_preprocessing import create_quality_aware_transform
from src.utils.device import get_device
from src.training.train_distillation import ThyroidDistillationModule
from src.utils.teacher_loader import load_best_teacher

console = Console()


class DistillationExperimentRunner:
    """Runner for knowledge distillation experiments."""
    
    def __init__(self, config_dir: Path):
        """Initialize the experiment runner."""
        self.config_dir = config_dir.absolute()
        self.results = {}
        
        # Clear Hydra instance
        GlobalHydra.instance().clear()
        
        console.print(Panel.fit(
            "[bold cyan]Knowledge Distillation Experiment Runner[/bold cyan]\n"
            "[dim]Train Vision Transformers with CNN/ViT teachers[/dim]",
            border_style="blue"
        ))
    
    def run_distillation_experiment(
        self,
        student_model: str,
        teacher_checkpoint: str,
        experiment_name: Optional[str] = None,
        config_overrides: Optional[Dict] = None,
        quick_test: bool = False
    ) -> Dict:
        """
        Run a single distillation experiment.
        
        Args:
            student_model: Name of student model (e.g., 'deit_tiny')
            teacher_checkpoint: Path to teacher checkpoint
            experiment_name: Custom experiment name
            config_overrides: Additional config overrides
            quick_test: Run quick test (2 epochs)
            
        Returns:
            Dictionary with experiment results
        """
        if experiment_name is None:
            teacher_name = Path(teacher_checkpoint).stem.split('-')[0]
            experiment_name = f"{student_model}_distill_{teacher_name}"
        
        console.print(f"\n[bold]Running experiment: {experiment_name}[/bold]")
        console.print(f"Student: {student_model}")
        console.print(f"Teacher: {teacher_checkpoint}")
        
        # Initialize Hydra
        with initialize_config_dir(config_dir=str(self.config_dir), version_base="1.2"):
            # Handle ViT models in subdirectory
            if student_model.startswith(('vit_', 'deit_', 'swin_')):
                model_override = f"model=vit/{student_model}"
            else:
                model_override = f"model={student_model}"
            
            # Compose configuration with all overrides at once
            overrides = [
                model_override,
                "training=distillation",
            ]
            
            if quick_test:
                overrides.extend([
                    "training.num_epochs=2",
                    "training.val_check_interval=1.0"
                ])
            
            # Compose the full config
            cfg = compose(config_name="config", overrides=overrides)
            
            # The distillation config might be in different places depending on how configs are structured
            # First, check if distillation config is already loaded from training config
            distillation_config = None
            
            # Check if it's at the top level
            if hasattr(cfg, 'distillation') and cfg.distillation is not None:
                distillation_config = cfg.distillation
            # Check if it's under model config
            elif hasattr(cfg.model, 'distillation') and cfg.model.distillation is not None:
                distillation_config = cfg.model.distillation
            # Check if it's under training config
            elif hasattr(cfg.training, 'distillation') and cfg.training.distillation is not None:
                distillation_config = cfg.training.distillation
            
            # If distillation config wasn't found anywhere, create it
            if distillation_config is None:
                # Temporarily disable struct mode to add new config
                OmegaConf.set_struct(cfg, False)
                cfg.distillation = OmegaConf.create({})
                distillation_config = cfg.distillation
                OmegaConf.set_struct(cfg, True)
            
            # Update distillation settings - disable struct mode temporarily
            struct_mode = OmegaConf.is_struct(cfg)
            if struct_mode:
                OmegaConf.set_struct(cfg, False)
            
            # Set distillation parameters
            distillation_config.enabled = True
            distillation_config.teacher_checkpoint = teacher_checkpoint
            distillation_config.teacher_model_type = 'cnn'
            distillation_config.freeze_teacher = True
            distillation_config.alpha = 0.7
            distillation_config.temperature = 3.0
            distillation_config.distillation_type = 'soft'
            distillation_config.teacher_eval_mode = True
            
            # If the distillation config is nested, ensure it's also at the top level for the training module
            if not hasattr(cfg, 'distillation'):
                cfg.distillation = distillation_config
            
            # Re-enable struct mode if it was enabled
            if struct_mode:
                OmegaConf.set_struct(cfg, True)
            
            # Fix paths - resolve interpolations manually
            if hasattr(cfg, 'paths'):
                OmegaConf.set_struct(cfg, False)
                cfg.paths.data_dir = str(Path.cwd() / "data")
                cfg.paths.checkpoint_dir = str(Path.cwd() / "checkpoints")
                cfg.paths.log_dir = str(Path.cwd() / "logs")
                OmegaConf.set_struct(cfg, True)
            
            # Update dataset path if it has interpolation
            if hasattr(cfg.dataset, 'path') and isinstance(cfg.dataset.path, str) and '${' in cfg.dataset.path:
                OmegaConf.set_struct(cfg.dataset, False)
                cfg.dataset.path = str(Path.cwd() / "data" / "raw")
                OmegaConf.set_struct(cfg.dataset, True)
            
            # Add custom overrides
            if config_overrides:
                # Temporarily disable struct mode for overrides
                struct_mode = OmegaConf.is_struct(cfg)
                if struct_mode:
                    OmegaConf.set_struct(cfg, False)
                
                for key, value in config_overrides.items():
                    # Special handling for distillation parameters
                    if key.startswith('distillation.'):
                        # Update the distillation config wherever it is
                        param = key.split('.', 1)[1]
                        if hasattr(cfg, 'distillation'):
                            setattr(cfg.distillation, param, value)
                        if hasattr(cfg.model, 'distillation'):
                            setattr(cfg.model.distillation, param, value)
                        if hasattr(cfg.training, 'distillation'):
                            setattr(cfg.training.distillation, param, value)
                    else:
                        # Regular override
                        parts = key.split('.')
                        if len(parts) == 2:
                            section, param = parts
                            if not hasattr(cfg, section):
                                setattr(cfg, section, OmegaConf.create({}))
                            setattr(getattr(cfg, section), param, value)
                        else:
                            # For nested keys, use OmegaConf.update
                            try:
                                OmegaConf.update(cfg, key, value, merge=False)
                            except Exception:
                                # If it fails, try to create the path
                                pass
                
                # Re-enable struct mode
                if struct_mode:
                    OmegaConf.set_struct(cfg, True)
            
            # Run experiment
            result = self._run_single_experiment(cfg, experiment_name)
            
        return result
    
    def _run_single_experiment(self, cfg: DictConfig, experiment_name: str) -> Dict:
        """Execute a single distillation experiment."""
        device = get_device()
        console.print(f"Device: {device}")
        
        try:
            # Create data loaders
            console.print("\n[cyan]Creating data loaders...[/cyan]")
            
            # Setup transforms
            train_transform = create_quality_aware_transform(
                target_size=cfg.dataset.image_size,
                quality_report_path=None,
                augmentation_level=cfg.training.get('augmentation_level', 'medium'),
                split='train'
            )
            
            val_transform = create_quality_aware_transform(
                target_size=cfg.dataset.image_size,
                quality_report_path=None,
                augmentation_level='none',
                split='val'
            )
            
            # Create data loaders
            data_loaders = create_data_loaders(
                root_dir=cfg.dataset.path,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.dataset.num_workers,
                transform_train=train_transform,
                transform_val=val_transform,
                target_size=cfg.dataset.image_size,
                normalize=False,
                patient_level_split=cfg.dataset.patient_level_split
            )
            
            console.print(f"[green]✓ Data loaders created[/green]")
            
            # Create distillation module
            console.print("\n[cyan]Creating distillation module...[/cyan]")
            model = ThyroidDistillationModule(cfg)
            
            # Print model info
            student_params = sum(p.numel() for p in model.student.parameters())
            console.print(f"[green]✓ Distillation module created[/green]")
            console.print(f"  Student parameters: {student_params:,}")
            
            # Setup callbacks
            callbacks = []
            
            # Model checkpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=cfg.paths.checkpoint_dir,
                filename=f"{experiment_name}-{{epoch:02d}}-{{val_acc:.4f}}",
                monitor='val_acc',
                mode='max',
                save_top_k=3,
                save_last=True
            )
            callbacks.append(checkpoint_callback)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_acc',
                patience=cfg.training.early_stopping.patience,
                mode='max',
                min_delta=cfg.training.early_stopping.min_delta
            )
            callbacks.append(early_stopping)
            
            # Progress bar
            callbacks.append(RichProgressBar())
            
            # Learning rate monitor
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))
            
            # Setup logger
            wandb_logger = None
            if cfg.get('use_wandb', True) and cfg.get('wandb', None):
                try:
                    import wandb
                    wandb_logger = WandbLogger(
                        project=cfg.wandb.get('project', 'thyroid-distillation'),
                        name=experiment_name,
                        tags=['distillation', cfg.model.name, 'phase3'],
                        config=OmegaConf.to_container(cfg, resolve=True)
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not initialize W&B logger: {e}[/yellow]")
                    wandb_logger = None
            
            # Create trainer
            trainer_config = {
                'max_epochs': cfg.training.num_epochs,
                'callbacks': callbacks,
                'logger': wandb_logger,
                'devices': 1,
                'log_every_n_steps': cfg.training.get('log_every_n_steps', 10),
                'val_check_interval': cfg.training.get('val_check_interval', 1.0),
                'gradient_clip_val': cfg.training.get('gradient_clip_val', 1.0),
                'accumulate_grad_batches': cfg.training.get('accumulate_grad_batches', 1),
                'deterministic': cfg.training.get('deterministic', False),
                'enable_model_summary': True,
                'enable_checkpointing': True,
            }
            
            # Device configuration
            if device.type == 'mps':
                trainer_config['accelerator'] = 'mps'
                trainer_config['precision'] = 32
            elif device.type == 'cuda':
                trainer_config['accelerator'] = 'gpu'
                trainer_config['precision'] = cfg.training.get('precision', '16-mixed')
            else:
                trainer_config['accelerator'] = 'cpu'
                trainer_config['precision'] = 32
            
            trainer = pl.Trainer(**trainer_config)
            
            # Train
            console.print("\n[cyan]Starting distillation training...[/cyan]")
            trainer.fit(
                model,
                train_dataloaders=data_loaders['train'],
                val_dataloaders=data_loaders['val']
            )
            
            # Get best validation accuracy from checkpoint callback
            best_val_acc = 0.0
            if hasattr(checkpoint_callback, 'best_model_score') and checkpoint_callback.best_model_score is not None:
                best_val_acc = float(checkpoint_callback.best_model_score)
                # Handle negative values (PyTorch Lightning sometimes stores negative for max metrics)
                if checkpoint_callback.mode == 'max' and best_val_acc < 0:
                    best_val_acc = -best_val_acc
            elif hasattr(checkpoint_callback, 'best_k_models') and checkpoint_callback.best_k_models:
                # Get the best score from the best_k_models dict
                scores = list(checkpoint_callback.best_k_models.values())
                if checkpoint_callback.mode == 'max':
                    # For max mode, scores might be stored as negative
                    best_val_acc = -min(scores) if all(s < 0 for s in scores) else max(scores)
                else:
                    best_val_acc = min(scores)
            else:
                # Try to get from callback metrics
                best_val_acc = float(trainer.callback_metrics.get('val_acc', 0.0))
            
            # Debug: print available metrics
            console.print(f"[dim]Available metrics: {list(trainer.callback_metrics.keys())}[/dim]")
            console.print(f"[dim]Best validation accuracy: {best_val_acc:.4f}[/dim]")
            
            # Get teacher agreement from last validation
            teacher_agreement = float(trainer.callback_metrics.get('val_teacher_agreement', 0.0))
            if teacher_agreement == 0.0:
                # Try regular teacher_agreement metric
                teacher_agreement = float(trainer.callback_metrics.get('teacher_agreement', 0.0))
            
            # Test
            console.print("\n[cyan]Running test evaluation...[/cyan]")
            test_results = trainer.test(
                model,
                dataloaders=data_loaders['test'],
                ckpt_path='best'
            )
            
            # Extract results
            best_epoch = trainer.current_epoch  # Default to last epoch
            if hasattr(checkpoint_callback, 'best_epoch'):
                best_epoch = checkpoint_callback.best_epoch
            elif hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
                # Try to extract epoch from checkpoint filename
                import re
                match = re.search(r'epoch=(\d+)', checkpoint_callback.best_model_path)
                if match:
                    best_epoch = int(match.group(1))
            
            result = {
                'experiment_name': experiment_name,
                'student_model': cfg.model.name,
                'teacher_checkpoint': cfg.distillation.teacher_checkpoint,
                'status': 'success',
                'val_acc': best_val_acc,
                'test_acc': float(test_results[0].get('test_acc', 0)),
                'best_epoch': best_epoch,
                'teacher_agreement': teacher_agreement,
                'final_alpha': model.get_current_alpha() if hasattr(model, 'get_current_alpha') else getattr(cfg.distillation, 'alpha', 0.5),
            }
            
            console.print(f"[green]✓ Experiment completed successfully[/green]")
            console.print(f"  Test accuracy: {result['test_acc']:.4f}")
            if result['teacher_agreement'] > 0:
                console.print(f"  Teacher agreement: {result['teacher_agreement']:.4f}")
            
            return result
            
        except Exception as e:
            console.print(f"[red]✗ Experiment failed: {str(e)}[/red]")
            import traceback
            traceback.print_exc()
            
            return {
                'experiment_name': experiment_name,
                'student_model': cfg.model.name,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_sweep(
        self,
        student_models: List[str],
        teacher_checkpoints: List[str],
        config_overrides: Optional[Dict] = None,
        quick_test: bool = False
    ) -> Dict:
        """
        Run a sweep of distillation experiments.
        
        Args:
            student_models: List of student model names
            teacher_checkpoints: List of teacher checkpoint paths
            config_overrides: Config overrides for all experiments
            quick_test: Run quick tests
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for student in student_models:
            for teacher_ckpt in teacher_checkpoints:
                result = self.run_distillation_experiment(
                    student_model=student,
                    teacher_checkpoint=teacher_ckpt,
                    config_overrides=config_overrides,
                    quick_test=quick_test
                )
                
                key = f"{student}_{Path(teacher_ckpt).stem}"
                results[key] = result
        
        return results
    
    def display_results(self, results: Dict):
        """Display experiment results in a table."""
        table = Table(title="Distillation Experiment Results")
        
        table.add_column("Experiment", style="cyan")
        table.add_column("Student", style="blue")
        table.add_column("Teacher", style="green")
        table.add_column("Val Acc", style="yellow")
        table.add_column("Test Acc", style="yellow")
        table.add_column("Agreement", style="magenta")
        table.add_column("Status", style="red")
        
        for exp_name, result in results.items():
            if result['status'] == 'success':
                table.add_row(
                    exp_name,
                    result.get('student_model', 'N/A'),
                    Path(result.get('teacher_checkpoint', 'N/A')).stem,
                    f"{result.get('val_acc', 0):.4f}",
                    f"{result.get('test_acc', 0):.4f}",
                    f"{result.get('teacher_agreement', 0):.4f}",
                    "[green]Success[/green]"
                )
            else:
                table.add_row(
                    exp_name,
                    result.get('student_model', 'N/A'),
                    'N/A',
                    'N/A',
                    'N/A',
                    'N/A',
                    "[red]Failed[/red]"
                )
        
        console.print(table)


def main():
    """Main entry point for distillation experiments."""
    parser = argparse.ArgumentParser(
        description="Run knowledge distillation experiments for Vision Transformers"
    )
    
    parser.add_argument(
        '--student',
        type=str,
        required=True,
        help='Student model name (e.g., deit_tiny, deit_small)'
    )
    
    parser.add_argument(
        '--teacher',
        type=str,
        required=True,
        help='Path to teacher checkpoint or model name for auto-discovery'
    )
    
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('configs'),
        help='Path to Hydra configs directory'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Custom experiment name'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        help='Distillation loss weight (0-1)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='Distillation temperature'
    )
    
    parser.add_argument(
        '--distillation-type',
        choices=['soft', 'hard'],
        help='Type of distillation'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test (2 epochs)'
    )
    
    parser.add_argument(
        '--ensemble-teachers',
        nargs='+',
        help='Use ensemble of teachers (provide multiple checkpoint paths)'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = DistillationExperimentRunner(args.config_dir)
    
    # Build config overrides
    config_overrides = {}
    
    if args.alpha is not None:
        config_overrides['distillation.alpha'] = args.alpha
    
    if args.temperature is not None:
        config_overrides['distillation.temperature'] = args.temperature
    
    if args.distillation_type is not None:
        config_overrides['distillation.distillation_type'] = args.distillation_type
    
    if args.batch_size is not None:
        config_overrides['training.batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config_overrides['optimizer.lr'] = args.learning_rate
    
    if args.epochs is not None:
        config_overrides['training.num_epochs'] = args.epochs
    
    # Handle teacher checkpoint
    teacher_checkpoint = args.teacher
    if not Path(teacher_checkpoint).exists():
        # Try to find checkpoint by model name
        console.print(f"[yellow]Teacher checkpoint not found, searching for {args.teacher}...[/yellow]")
        try:
            teacher_model, _ = load_best_teacher(
                args.teacher,
                checkpoint_dir=Path('checkpoints'),
                verbose=False
            )
            # Find the actual checkpoint path
            checkpoint_dir = Path('checkpoints')
            pattern = f"{args.teacher}*.ckpt"
            checkpoints = list(checkpoint_dir.glob(pattern))
            if checkpoints:
                teacher_checkpoint = str(checkpoints[0])
                console.print(f"[green]Found teacher checkpoint: {teacher_checkpoint}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to find teacher checkpoint: {e}[/red]")
            sys.exit(1)
    
    # Handle ensemble teachers
    if args.ensemble_teachers:
        # For ensemble, update the distillation config after runner initialization
        console.print(f"[yellow]Ensemble teachers specified: {args.ensemble_teachers}[/yellow]")
        console.print("[yellow]Note: You'll need to update the config manually for ensemble support[/yellow]")
    
    # Run experiment
    result = runner.run_distillation_experiment(
        student_model=args.student,
        teacher_checkpoint=teacher_checkpoint,
        experiment_name=args.experiment_name,
        config_overrides=config_overrides,
        quick_test=args.quick_test
    )
    
    # Display results
    runner.display_results({args.experiment_name or 'experiment': result})
    
    # Save results
    results_file = Path('distillation_results.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results[f"{args.student}_{datetime.now().isoformat()}"] = result
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    console.print(f"\n[green]Results saved to {results_file}[/green]")


if __name__ == "__main__":
    main()