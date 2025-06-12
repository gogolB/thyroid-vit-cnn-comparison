"""
Rich console utilities for beautiful logging throughout the project
"""

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
import logging
from typing import Optional


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


# Example usage function
def demo_rich_features():
    """Demonstrate Rich features for the project."""
    
    # Logger example
    logger = get_logger(__name__)
    logger.info("Starting thyroid classification project...")
    
    # Data summary example
    print_data_summary(
        dataset_name="CARS Thyroid",
        total_images=450,
        class_distribution={"normal": 225, "cancerous": 225},
        image_size=(512, 512, 1),
        split_info={"train": 315, "val": 68, "test": 67}
    )
    
    # Progress bar example
    import time
    with create_progress_bar("Loading images") as progress:
        task = progress.add_task("[cyan]Processing...", total=100)
        for i in range(100):
            time.sleep(0.01)
            progress.update(task, advance=1)
    
    # Training config example
    print_training_config({
        "model": "ResNet50",
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": {
            "type": "AdamW",
            "weight_decay": 0.01
        }
    })
    
    # Model summary example
    print_model_summary(
        model_name="ResNet50",
        total_params=25_557_032,
        trainable_params=23_514_432,
        model_size_mb=97.49
    )


if __name__ == "__main__":
    demo_rich_features()
