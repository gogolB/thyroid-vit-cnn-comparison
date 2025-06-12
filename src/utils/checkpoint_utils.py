"""
Checkpoint utilities for managing model checkpoints.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich.console import Console

console = Console()


class BestCheckpointCallback(Callback):
    """
    Callback to copy the best checkpoint to a standardized location after training.
    
    This callback monitors the ModelCheckpoint callback and copies the best
    checkpoint to modelname-best.ckpt at the end of training.
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.best_model_path = None
        self.model_name = None
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Extract model name at training start."""
        # Try to get model name from config
        if hasattr(pl_module, 'config'):
            self.model_name = pl_module.config.model.name
        elif hasattr(pl_module, 'hparams') and 'model' in pl_module.hparams:
            self.model_name = pl_module.hparams.model.name
        else:
            # Fallback to class name
            self.model_name = pl_module.__class__.__name__.lower()
            
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Copy best checkpoint at the end of training."""
        # Find the ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break
                
        if checkpoint_callback is None:
            console.print("[yellow]Warning: No ModelCheckpoint callback found.[/yellow]")
            return
            
        # Get the best model path
        best_model_path = checkpoint_callback.best_model_path
        
        if not best_model_path or not os.path.exists(best_model_path):
            console.print("[yellow]Warning: No best model checkpoint found.[/yellow]")
            return
            
        # Create destination path
        dest_filename = f"{self.model_name}-best.ckpt"
        dest_path = self.checkpoint_dir / dest_filename
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the checkpoint
        try:
            shutil.copy2(best_model_path, dest_path)
            console.print(f"\n[green]✓ Best checkpoint copied to: {dest_path}[/green]")
            console.print(f"  Original: {best_model_path}")
            console.print(f"  Best score: {checkpoint_callback.best_model_score:.4f}")
            
            # Also create a symlink for easy access (Unix-like systems only)
            if os.name != 'nt':  # Not Windows
                latest_link = self.checkpoint_dir / f"{self.model_name}-latest.ckpt"
                if latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(dest_filename)
                console.print(f"  Symlink: {latest_link} -> {dest_filename}")
                
        except Exception as e:
            console.print(f"[red]Error copying best checkpoint: {e}[/red]")


def get_best_checkpoint(checkpoint_dir: Path, model_name: str) -> Optional[Path]:
    """
    Get the best checkpoint for a given model.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of the model
        
    Returns:
        Path to best checkpoint if found, None otherwise
    """
    best_ckpt_path = checkpoint_dir / f"{model_name}-best.ckpt"
    
    if best_ckpt_path.exists():
        return best_ckpt_path
        
    # Fallback: try to find latest checkpoint
    latest_ckpt_path = checkpoint_dir / f"{model_name}-latest.ckpt"
    if latest_ckpt_path.exists():
        return latest_ckpt_path
        
    # Last resort: find any checkpoint for this model
    pattern = f"{model_name}*.ckpt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if checkpoints:
        # Sort by modification time and return newest
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
        
    return None


def cleanup_old_checkpoints(checkpoint_dir: Path, model_name: str, keep_best: int = 3):
    """
    Clean up old checkpoints, keeping only the best N and the -best.ckpt file.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Name of the model
        keep_best: Number of best checkpoints to keep (excluding -best.ckpt)
    """
    # Find all checkpoints for this model
    pattern = f"{model_name}-epoch*.ckpt"
    checkpoints = list(checkpoint_dir.glob(pattern))
    
    if len(checkpoints) <= keep_best:
        return  # Nothing to clean up
        
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old checkpoints
    for ckpt in checkpoints[keep_best:]:
        try:
            ckpt.unlink()
            console.print(f"[dim]Removed old checkpoint: {ckpt.name}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove {ckpt.name}: {e}[/yellow]")