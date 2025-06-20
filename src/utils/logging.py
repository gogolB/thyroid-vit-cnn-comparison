"""
Utility functions and classes related to logging and console output.
This module will consolidate utilities previously in src/utils/console.py
and potentially other logging-related code.
"""
# Standard library imports
import logging
from typing import Optional
from pathlib import Path # Added as it's used in the Logger class

# Third-party imports
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
import structlog

# Project-specific imports (will be added when functions are moved)


# Global console instance
console = Console()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with Rich formatting.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger with Rich handler
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    return logging.getLogger(name)


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a standardized progress bar for the project.
    
    Args:
        description: Task description
        
    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_data_summary(
    dataset_name: str,
    total_images: int,
    class_distribution: dict,
    image_size: tuple,
    split_info: Optional[dict] = None
):
    """
    Print a beautiful summary table for dataset information.
    
    Args:
        dataset_name: Name of the dataset
        total_images: Total number of images
        class_distribution: Dict mapping class names to counts
        image_size: Tuple of (height, width) or (height, width, channels)
        split_info: Optional dict with train/val/test counts
    """
    table = Table(title=f"Dataset Summary: {dataset_name}", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Basic info
    table.add_row("Total Images", str(total_images))
    table.add_row("Image Size", f"{' × '.join(map(str, image_size))}")
    
    # Class distribution
    table.add_section()
    table.add_row("[bold]Class Distribution[/bold]", "")
    for class_name, count in class_distribution.items():
        percentage = (count / total_images) * 100
        table.add_row(f"  {class_name}", f"{count} ({percentage:.1f}%)")
    
    # Split info if provided
    if split_info:
        table.add_section()
        table.add_row("[bold]Data Split[/bold]", "")
        for split_name, count in split_info.items():
            percentage = (count / total_images) * 100
            table.add_row(f"  {split_name}", f"{count} ({percentage:.1f}%)")
    
    console.print(table)


def print_training_config(config: dict):
    """
    Print training configuration in a nice panel.
    
    Args:
        config: Configuration dictionary
    """
    config_text = []
    
    for key, value in config.items():
        if isinstance(value, dict):
            config_text.append(f"[bold cyan]{key}:[/bold cyan]")
            for sub_key, sub_value in value.items():
                config_text.append(f"  {sub_key}: {sub_value}")
        else:
            config_text.append(f"[bold cyan]{key}:[/bold cyan] {value}")
    
    panel = Panel(
        "\n".join(config_text),
        title="[bold green]Training Configuration[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    
    console.print(panel)


def print_model_summary(
    model_name: str,
    total_params: int,
    trainable_params: int,
    model_size_mb: float
):
    """
    Print model summary information.
    
    Args:
        model_name: Name of the model
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
        model_size_mb: Model size in MB
    """
    table = Table(title=f"Model: {model_name}", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Frozen Parameters", f"{total_params - trainable_params:,}")
    table.add_row("Model Size", f"{model_size_mb:.2f} MB")
    
    console.print(table)

class Logger:
    """Unified logging with Rich display and structured logs."""
    
    def __init__(self, name: str, experiment_dir: Path = None):
        self.console = Console()
        self.logger = structlog.get_logger(name)
        if experiment_dir:
            self._setup_file_logging(name, experiment_dir)

    def _setup_file_logging(self, name: str, experiment_dir: Path):
        # Placeholder for actual file logging setup with structlog
        # e.g., structlog.configure([... processors ...])
        # For now, a simple print or pass is fine.
        print(f"Placeholder: File logging would be set up for {name} in {experiment_dir}")
        pass