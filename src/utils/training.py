"""
Utility functions and classes related to model training processes.
This module will consolidate utilities for device management (previously in src/utils/device.py),
checkpoint handling (previously in src/utils/checkpoint_utils.py), and other training helpers.
"""

# Standard library imports
import platform
import os
import shutil
from pathlib import Path
from typing import Union, Optional

# Third-party imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich.console import Console
from rich.table import Table

# Project-specific imports (will be added when functions are moved)

console = Console()


def get_device(device_preference: str = 'auto') -> torch.device:
    """
    Get the best available device based on preference and availability.
    
    Args:
        device_preference: 'auto', 'cuda', 'mps', 'cpu', or specific device like 'cuda:0'
        
    Returns:
        torch.device object
    """
    if device_preference == 'auto':
        # Try CUDA first
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Then try MPS (Mac GPU)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        # Fallback to CPU
        else:
            return torch.device('cpu')
    
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            console.print("[yellow]CUDA not available, falling back to auto selection[/yellow]")
            return get_device('auto')
    
    elif device_preference == 'mps':
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        else:
            console.print("[yellow]MPS not available, falling back to auto selection[/yellow]")
            return get_device('auto')
    
    elif device_preference.startswith('cuda:'):
        # Specific CUDA device
        if torch.cuda.is_available():
            device_id = int(device_preference.split(':')[1])
            if device_id < torch.cuda.device_count():
                return torch.device(device_preference)
            else:
                console.print(f"[yellow]CUDA device {device_id} not found, using cuda:0[/yellow]")
                return torch.device('cuda:0')
        else:
            console.print("[yellow]CUDA not available, falling back to auto selection[/yellow]")
            return get_device('auto')
    
    else:
        # Default to CPU
        return torch.device('cpu')


def device_info() -> str:
    """
    Get detailed device information for debugging.
    
    Returns:
        String with device information formatted as a table
    """
    table = Table(title="Device Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # System info
    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    
    # CUDA info
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("CUDA Device Count", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            table.add_row(f"CUDA Device {i}", torch.cuda.get_device_name(i))
            table.add_row(f"CUDA Memory {i}", 
                         f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # MPS info
    table.add_row("MPS Available", str(torch.backends.mps.is_available()))
    table.add_row("MPS Built", str(torch.backends.mps.is_built()))
    
    # CPU info
    table.add_row("CPU Threads", str(torch.get_num_threads()))
    
    return table


def optimize_for_device(device: torch.device) -> dict:
    """
    Get device-specific optimization settings.
    
    Args:
        device: torch.device object
        
    Returns:
        Dictionary with optimization settings
    """
    settings = {
        'pin_memory': False,
        'num_workers': 4,
        'precision': 32,
        'use_amp': False,
        'cudnn_benchmark': False,
        'cudnn_deterministic': True
    }
    
    if device.type == 'cuda':
        settings.update({
            'pin_memory': True,
            'num_workers': 8,
            'precision': 16,
            'use_amp': True,
            'cudnn_benchmark': True,
            'cudnn_deterministic': False  # For better performance
        })
    elif device.type == 'mps':
        settings.update({
            'pin_memory': False,  # Not supported on MPS
            'num_workers': 4,     # MPS works better with fewer workers
            'precision': 32,      # MPS doesn't support fp16 yet
            'use_amp': False,     # Not supported on MPS
        })
    
    return settings


def move_to_device(data: Union[torch.Tensor, list, tuple, dict], device: torch.device):
    """
    Recursively move data to the specified device.
    
    Args:
        data: Data to move (tensor, list, tuple, or dict)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data


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


# Test function
if __name__ == "__main__":
    console.print("[bold cyan]Device Detection Test[/bold cyan]\n")
    
    # Test auto detection
    device = get_device('auto')
    console.print(f"Auto-detected device: [green]{device}[/green]")
    
    # Show device info
    console.print("\n")
    console.print(device_info())
    
    # Show optimization settings
    settings = optimize_for_device(device)
    console.print("\n[cyan]Optimization Settings:[/cyan]")
    for key, value in settings.items():
        console.print(f"  {key}: {value}")