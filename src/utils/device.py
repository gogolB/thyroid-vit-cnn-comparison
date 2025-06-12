"""
Device utilities for handling CUDA, MPS (Mac), and CPU selection.
"""

import torch
import platform
from typing import Union

from rich.console import Console
from rich.table import Table

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