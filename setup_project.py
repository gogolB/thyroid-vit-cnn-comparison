#!/usr/bin/env python3
"""
Enhanced Project Setup Script with Rich Terminal UI
For Vision Transformer vs CNN Thyroid Classification
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Check if rich is installed, if not, use basic print
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich import print as rprint
    from rich.prompt import Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not installed. Using basic output.")
    print("Install with: pip install rich")
    rprint = print

# Initialize console
console = Console() if RICH_AVAILABLE else None


def create_directory_structure():
    """Create the project directory structure with progress tracking."""
    
    directories = [
        # Configuration directories
        "configs",
        "configs/model",
        "configs/model/cnn",
        "configs/model/vit",
        "configs/dataset",
        "configs/training",
        "configs/augmentation",
        
        # Source code directories
        "src",
        "src/data",
        "src/models",
        "src/models/cnn",
        "src/models/vit",
        "src/models/hybrid",
        "src/training",
        "src/evaluation",
        "src/utils",
        
        # Experiment and data directories
        "experiments",
        "data",
        "data/raw",
        "data/processed",
        "data/splits",
        
        # Other directories
        "notebooks",
        "tests",
        "tests/data",
        "tests/models",
        "tests/training",
        "docs",
        "scripts",
    ]
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Creating directory structure...", total=len(directories))
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py files in Python package directories
                if directory.startswith("src") and not directory.endswith("src"):
                    init_file = Path(directory) / "__init__.py"
                    if not init_file.exists():
                        init_file.touch()
                
                progress.update(task, advance=1)
                time.sleep(0.01)  # Small delay for visual effect
        
        console.print("[green]✓[/green] Directory structure created successfully")
    else:
        print("Creating directory structure...")
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            if directory.startswith("src") and not directory.endswith("src"):
                init_file = Path(directory) / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
        print("✓ Directory structure created successfully")


def show_directory_tree():
    """Display the created directory structure as a tree."""
    
    if not RICH_AVAILABLE:
        return
    
    tree = Tree("📁 thyroid-vit-cnn-comparison")
    
    # Add main branches
    configs = tree.add("📁 configs/")
    configs.add("📁 model/")
    configs.add("📁 dataset/")
    configs.add("📁 training/")
    configs.add("📁 augmentation/")
    
    src = tree.add("📁 src/")
    src.add("📁 data/")
    src.add("📁 models/")
    src.add("📁 training/")
    src.add("📁 evaluation/")
    src.add("📁 utils/")
    
    tree.add("📁 experiments/")
    tree.add("📁 data/")
    tree.add("📁 notebooks/")
    tree.add("📁 tests/")
    tree.add("📁 docs/")
    tree.add("📁 scripts/")
    
    console.print(tree)


def create_file_with_progress(filename, content, description):
    """Create a file with progress indication."""
    
    if RICH_AVAILABLE:
        with console.status(f"[bold green]{description}...") as status:
            with open(filename, "w") as f:
                f.write(content)
            time.sleep(0.1)  # Small delay for visual effect
        console.print(f"[green]✓[/green] {description} completed")
    else:
        print(f"{description}...")
        with open(filename, "w") as f:
            f.write(content)
        print(f"✓ {description} completed")


def create_gitignore():
    """Create .gitignore file."""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter Notebooks
.ipynb_checkpoints
*.ipynb_checkpoints

# Data files
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
*.tif
*.tiff
*.png
*.jpg
*.jpeg
*.npy
*.npz
*.h5
*.hdf5

# Model files
*.pth
*.pt
*.ckpt
*.onnx
*.pb

# Experiment outputs
experiments/*
!experiments/.gitkeep
wandb/
mlruns/
outputs/
logs/

# System files
.DS_Store
Thumbs.db

# Environment files
.env
.env.local

# Cache
.cache/
"""
    
    create_file_with_progress(".gitignore", gitignore_content, "Creating .gitignore")
    
    # Create .gitkeep files
    gitkeep_dirs = ["data/raw", "data/processed", "experiments"]
    for dir_path in gitkeep_dirs:
        Path(dir_path, ".gitkeep").touch()


def create_requirements():
    """Create requirements.txt."""
    
    requirements_content = """# Core dependencies
torch>=2.1.0
torchvision>=0.16.0
pytorch-lightning>=2.1.0
torchmetrics>=1.2.0

# Configuration and experiment tracking
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.16.0
pyyaml>=6.0.1

# Medical imaging
monai>=1.3.0
torchio>=0.19.0

# Image processing
opencv-python>=4.8.0
scikit-image>=0.22.0
albumentations>=1.3.0
Pillow>=10.0.0

# Data handling
numpy>=1.24.0
pandas>=2.0.0
h5py>=3.9.0
tifffile>=2023.8.0

# Visualization and CLI
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
rich>=13.7.0  # Beautiful terminal formatting and progress bars

# ML utilities
scikit-learn>=1.3.0
tqdm>=4.66.0  # Alternative progress bars (keeping for compatibility)

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Documentation
jupyterlab>=4.0.0
ipywidgets>=8.1.0

# Model architectures
timm>=0.9.0  # PyTorch Image Models for pretrained models
einops>=0.7.0  # For ViT implementations

# Optional but recommended
tensorboard>=2.14.0
optuna>=3.3.0  # Hyperparameter optimization
"""
    
    create_file_with_progress("requirements.txt", requirements_content, "Creating requirements.txt")


def write_yaml_manually(filepath, data, indent=0):
    """Write YAML-formatted data manually."""
    
    def write_value(f, key, value, indent_level):
        indent_str = "  " * indent_level
        
        if isinstance(value, dict):
            f.write(f"{indent_str}{key}:\n")
            for k, v in value.items():
                write_value(f, k, v, indent_level + 1)
        elif isinstance(value, list):
            f.write(f"{indent_str}{key}:\n")
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k == "_self_":
                            f.write(f"{indent_str}  - {k}\n")
                        else:
                            f.write(f"{indent_str}  - {k}: {v}\n")
                else:
                    f.write(f"{indent_str}  - {item}\n")
        elif isinstance(value, bool):
            f.write(f"{indent_str}{key}: {str(value).lower()}\n")
        elif isinstance(value, (int, float)):
            f.write(f"{indent_str}{key}: {value}\n")
        elif value is None:
            f.write(f"{indent_str}{key}: null\n")
        else:
            if any(c in str(value) for c in [':', '{', '}', '[', ']', ',', '&', '*', '#', '?', '|', '-', '<', '>', '=', '!', '%', '@', '"', "'"]):
                f.write(f'{indent_str}{key}: "{value}"\n')
            else:
                f.write(f"{indent_str}{key}: {value}\n")
    
    with open(filepath, "w") as f:
        if isinstance(data, dict):
            for key, value in data.items():
                write_value(f, key, value, indent)
        else:
            f.write(str(data))


def create_hydra_configs():
    """Create Hydra configuration files."""
    
    configs = {
        "configs/config.yaml": {
            "defaults": [
                {"model": "resnet50"},
                {"dataset": "cars"},
                {"training": "standard"},
                {"augmentation": "basic"},
                "_self_"
            ],
            "experiment_name": "${model.name}_${dataset.name}_${now:%Y-%m-%d_%H-%M-%S}",
            "seed": 42,
            "device": "cuda",
            "paths": {
                "data_dir": "${hydra:runtime.cwd}/data",
                "log_dir": "${hydra:runtime.cwd}/logs",
                "checkpoint_dir": "${hydra:runtime.cwd}/checkpoints"
            },
            "wandb": {
                "project": "thyroid-classification",
                "entity": None,
                "tags": ["${model.architecture}", "${dataset.name}"],
                "mode": "online"
            }
        },
        "configs/dataset/cars.yaml": {
            "name": "CARS_Thyroid",
            "path": "${paths.data_dir}/raw",
            "image_size": 256,
            "patch_size": 256,
            "patch_overlap": 0.1,
            "num_classes": 2,
            "class_names": ["normal", "cancerous"],
            "split_ratio": [0.7, 0.15, 0.15],
            "patient_level_split": True,
            "normalize": True,
            "cache": False,
            "num_workers": 4,
            "pin_memory": True
        },
        "configs/training/standard.yaml": {
            "batch_size": 32,
            "num_epochs": 100,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 0.001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999]
            },
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 100,
                "eta_min": 1e-6
            },
            "loss": {
                "_target_": "torch.nn.CrossEntropyLoss",
                "label_smoothing": 0.1
            },
            "early_stopping": {
                "patience": 15,
                "min_delta": 0.001,
                "mode": "max",
                "monitor": "val/accuracy"
            },
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "precision": "16-mixed",
            "deterministic": False
        }
    }
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Creating Hydra configs...", total=len(configs))
            
            for path, config in configs.items():
                write_yaml_manually(path, config)
                progress.update(task, advance=1)
                time.sleep(0.05)
        
        console.print("[green]✓[/green] Hydra configs created")
    else:
        print("Creating Hydra configuration files...")
        for path, config in configs.items():
            write_yaml_manually(path, config)
        print("✓ Hydra configs created")


def create_additional_files():
    """Create README, project log, and other documentation."""
    
    # Simplified versions for brevity
    files = {
        "README.md": "# Vision Transformer vs CNN Thyroid Classification\n\nProject for comparing ViTs and CNNs on thyroid tissue classification.",
        "project_log.md": f"# Project Log\n\n**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Status\n- [x] Project initialized\n- [ ] Data loader implementation",
        "pyproject.toml": '[project]\nname = "thyroid-vit-cnn-comparison"\nversion = "0.1.0"'
    }
    
    for filename, content in files.items():
        create_file_with_progress(filename, content, f"Creating {filename}")


def show_summary():
    """Show a summary of what was created."""
    
    if RICH_AVAILABLE:
        # Create a summary table
        table = Table(title="Project Setup Summary", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("Description")
        
        table.add_row("Directory Structure", "✓ Created", "Complete project hierarchy with Python packages")
        table.add_row("Configuration", "✓ Created", "Hydra configs for experiments")
        table.add_row("Dependencies", "✓ Created", "requirements.txt with all packages")
        table.add_row("Documentation", "✓ Created", "README, project log, and metadata")
        table.add_row("Git Setup", "✓ Created", ".gitignore with ML-specific patterns")
        
        console.print(table)
        
        # Next steps panel
        next_steps = """[bold cyan]Next Steps:[/bold cyan]

1. [yellow]Install dependencies:[/yellow]
   pip install -r requirements.txt

2. [yellow]Configure Weights & Biases:[/yellow]
   wandb login

3. [yellow]Place your CARS images:[/yellow]
   Copy images to data/raw/

4. [yellow]Run data preparation:[/yellow]
   Coming next with rich progress bars!
"""
        
        console.print(Panel(next_steps, title="[bold green]Setup Complete![/bold green]", border_style="green"))
    else:
        print("\n" + "="*60)
        print("✓ Project setup completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure W&B: wandb login")
        print("3. Place CARS images in data/raw/")
        print("4. Run data preparation script")


def main():
    """Main setup function."""
    
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]Vision Transformer vs CNN Thyroid Classification[/bold cyan]\n"
            "[dim]Project Setup with Rich Terminal UI[/dim]",
            border_style="blue"
        ))
    else:
        print("="*60)
        print("Vision Transformer vs CNN Thyroid Classification")
        print("Project Setup Script")
        print("="*60)
    
    # Check directory
    if os.path.basename(os.getcwd()) != "thyroid-vit-cnn-comparison":
        if RICH_AVAILABLE:
            if not Confirm.ask("[yellow]Not in 'thyroid-vit-cnn-comparison' directory. Continue?[/yellow]"):
                console.print("[red]Setup cancelled.[/red]")
                return
        else:
            response = input("Not in 'thyroid-vit-cnn-comparison' directory. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Setup cancelled.")
                return
    
    # Run setup
    create_directory_structure()
    if RICH_AVAILABLE:
        show_directory_tree()
    
    create_gitignore()
    create_requirements()
    create_hydra_configs()
    create_additional_files()
    
    show_summary()


if __name__ == "__main__":
    main()
