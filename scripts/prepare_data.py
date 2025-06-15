#!/usr/bin/env python3
"""
Data preparation script for CARS thyroid dataset.
This script helps organize, split, and validate the dataset.
"""

import os
import sys
from pathlib import Path
import shutil
import numpy as np
from typing import Dict, List, Tuple
import argparse
import json
from sklearn.model_selection import train_test_split, StratifiedKFold


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt

# Import our modules
from src.data.dataset import CARSThyroidDataset, create_data_loaders
from src.data.transforms import print_augmentation_summary
from src.data.visualize import visualize_dataset

console = Console()


def scan_data_directory(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan the data directory for images.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary mapping class names to lists of image paths
    """
    
    supported_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    class_images = {}
    
    console.print(f"[cyan]Scanning directory:[/cyan] {data_dir}")
    
    # Check for class subdirectories
    for class_name in ['normal', 'cancerous']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            images = []
            for fmt in supported_formats:
                images.extend(list(class_dir.glob(f'*{fmt}')))
            class_images[class_name] = images
            console.print(f"[green]Found {len(images)} images in {class_name}/[/green]")
    
    # If no class directories, scan root
    if not class_images:
        console.print("[yellow]No class directories found. Scanning root directory...[/yellow]")
        all_images = []
        for fmt in supported_formats:
            all_images.extend(list(data_dir.glob(f'*{fmt}')))
        
        if all_images:
            console.print(f"[yellow]Found {len(all_images)} images in root directory.[/yellow]")
            console.print("[yellow]Images should be organized into 'normal' and 'cancerous' subdirectories.[/yellow]")
            return {'unorganized': all_images}
    
    return class_images


def organize_images(data_dir: Path, class_images: Dict[str, List[Path]]):
    """
    Organize images into proper directory structure.
    
    Args:
        data_dir: Path to data directory
        class_images: Dictionary of images by class
    """
    
    if 'unorganized' in class_images:
        console.print("\n[yellow]Images need to be organized into class directories.[/yellow]")
        console.print("Please organize your images into:")
        console.print("  - data/raw/normal/")
        console.print("  - data/raw/cancerous/")
        
        if Confirm.ask("\nWould you like to see an example organization script?"):
            example_script = """
# Example Python script to organize images:

from pathlib import Path
import shutil

data_dir = Path('data/raw')

# Create class directories
(data_dir / 'normal').mkdir(exist_ok=True)
(data_dir / 'cancerous').mkdir(exist_ok=True)

# Move images (example - adjust based on your naming convention)
for img in data_dir.glob('*.tif'):
    if 'normal' in img.name.lower() or 'healthy' in img.name.lower():
        shutil.move(str(img), str(data_dir / 'normal' / img.name))
    elif 'cancer' in img.name.lower() or 'tumor' in img.name.lower():
        shutil.move(str(img), str(data_dir / 'cancerous' / img.name))
    else:
        print(f"Could not classify: {img.name}")
"""
            console.print(Panel(example_script, title="Example Organization Script", 
                              border_style="blue"))


def validate_image_properties(class_images: Dict[str, List[Path]], sample_size: int = 5):
    """
    Validate image properties (size, channels, bit depth).
    
    Args:
        class_images: Dictionary of images by class
        sample_size: Number of images to sample per class
    """
    
    console.print("\n[cyan]Validating image properties...[/cyan]")
    
    import cv2
    import tifffile
    
    for class_name, images in class_images.items():
        if class_name == 'unorganized':
            continue
            
        console.print(f"\n[yellow]Class: {class_name}[/yellow]")
        
        # Sample images
        sample_images = np.random.choice(images, 
                                       min(sample_size, len(images)), 
                                       replace=False)
        
        properties = []
        for img_path in sample_images:
            try:
                # Try tifffile first for TIFF images
                if img_path.suffix.lower() in ['.tif', '.tiff']:
                    img = tifffile.imread(str(img_path))
                else:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                
                properties.append({
                    'name': img_path.name,
                    'shape': img.shape,
                    'dtype': str(img.dtype),
                    'min': np.min(img),
                    'max': np.max(img),
                    'mean': np.mean(img)
                })
            except Exception as e:
                console.print(f"[red]Error reading {img_path.name}: {e}[/red]")
        
        # Display properties table
        if properties:
            table = Table(title=f"{class_name.capitalize()} Image Properties", 
                         show_header=True, header_style="bold magenta")
            table.add_column("Image", style="cyan", no_wrap=True)
            table.add_column("Shape", style="white")
            table.add_column("Type", style="yellow")
            table.add_column("Range", style="green")
            table.add_column("Mean", style="blue")
            
            for prop in properties:
                table.add_row(
                    prop['name'][:20] + '...' if len(prop['name']) > 20 else prop['name'],
                    str(prop['shape']),
                    prop['dtype'],
                    f"[{prop['min']}, {prop['max']}]",
                    f"{prop['mean']:.1f}"
                )
            
            console.print(table)


def check_patient_ids(class_images: Dict[str, List[Path]]):
    """
    Check if patient IDs can be extracted from filenames.
    
    Args:
        class_images: Dictionary of images by class
    """
    
    console.print("\n[cyan]Checking for patient IDs in filenames...[/cyan]")
    
    # Check filename patterns
    sample_names = []
    for class_name, images in class_images.items():
        if class_name == 'unorganized':
            continue
        for img in images[:5]:  # Sample first 5
            sample_names.append((class_name, img.name))
    
    console.print("\n[dim]Sample filenames:[/dim]")
    for class_name, name in sample_names[:6]:
        console.print(f"  {class_name}: {name}")
    
    # Check if filenames suggest patient IDs
    has_patient_ids = False
    
    # Common patterns for patient IDs
    patterns_found = []
    if any('patient' in name[1].lower() for name in sample_names):
        patterns_found.append("Contains 'patient' keyword")
        has_patient_ids = True
    if any('pt' in name[1].lower() for name in sample_names):
        patterns_found.append("Contains 'pt' abbreviation")
        has_patient_ids = True
    if any(len([c for c in name[1] if c.isdigit()]) > 3 for name in sample_names):
        # Check for long number sequences that might be patient IDs
        patterns_found.append("Contains long number sequences")
    
    if patterns_found:
        console.print(f"\n[yellow]Patterns found: {', '.join(patterns_found)}[/yellow]")
    
    if not has_patient_ids:
        console.print("\n[yellow]⚠ No clear patient IDs found in filenames.[/yellow]")
        console.print("[green]✓ Using image-level splitting (each image treated independently)[/green]")
        console.print("\n[dim]Note: If your images come from different patients, consider:[/dim]")
        console.print("[dim]  - Renaming files to include patient IDs (e.g., patient001_image1.tif)[/dim]")
        console.print("[dim]  - This prevents data leakage between train/test sets[/dim]")
    else:
        console.print("\n[green]✓ Patient ID patterns detected[/green]")
        console.print("[yellow]Consider enabling patient-level splitting to prevent data leakage[/yellow]")


def create_dataset_summary(data_dir: Path):
    """
    Create a summary of the dataset.
    
    Args:
        data_dir: Path to data directory
    """
    
    # Try to create dataset
    try:
        dataset = CARSThyroidDataset(
            root_dir=data_dir,
            split='all',
            target_size=256,
            normalize=True
        )
        
        # Create summary
        summary = {
            'total_images': len(dataset),
            'class_distribution': {
                'normal': int((dataset.labels == 0).sum()),
                'cancerous': int((dataset.labels == 1).sum())
            },
            'unique_patients': len(np.unique(dataset.patient_ids)),
            'image_paths': [str(p) for p in dataset.image_paths[:5]]  # Sample paths
        }
        
        # Save summary
        summary_path = data_dir.parent / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"\n[green]✓ Dataset summary saved to {summary_path}[/green]")
        
        # Display summary
        table = Table(title="Dataset Summary", show_header=True, 
                     header_style="bold magenta")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        
        table.add_row("Total Images", str(summary['total_images']))
        table.add_row("Normal Images", str(summary['class_distribution']['normal']))
        table.add_row("Cancerous Images", str(summary['class_distribution']['cancerous']))
        table.add_row("Unique Patients", str(summary['unique_patients']))
        
        console.print(table)
        
        return dataset
        
    except Exception as e:
        console.print(f"[red]Error creating dataset: {e}[/red]")
        return None

# --- K-FOLD ENHANCEMENT: New function to generate all split files ---
def generate_kfold_splits(
    data_dir: Path,
    k: int,
    test_size: float = 0.15,
    random_state: int = 42
):
    """Generates a held-out test set and k-fold train/validation splits."""
    console.print(Panel(f"[bold green]Generating {k}-Fold Cross-Validation Splits[/bold green]", border_style="green"))
    splits_dir = data_dir.parent / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the dataset class to load all file paths and labels
    full_dataset = CARSThyroidDataset(root_dir=data_dir, split='all')
    indices = np.arange(len(full_dataset.image_paths))
    labels = full_dataset.labels

    # 1. Create and save the held-out test set
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )
    test_split_path = splits_dir / 'test_split.json'
    with open(test_split_path, 'w') as f:
        json.dump({'test': test_indices.tolist()}, f, indent=2)
    console.print(f"✓ Saved held-out test set ({len(test_indices)} images) to [cyan]{test_split_path}[/cyan]")

    # 2. Use StratifiedKFold on the remaining data
    train_val_labels = labels[train_val_indices]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    for i, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
        fold_num = i + 1
        train_global_indices = train_val_indices[train_fold_idx]
        val_global_indices = train_val_indices[val_fold_idx]
        
        fold_split_path = splits_dir / f'split_fold_{fold_num}.json'
        with open(fold_split_path, 'w') as f:
            json.dump({'train': train_global_indices.tolist(), 'val': val_global_indices.tolist()}, f, indent=2)
        console.print(f"✓ Saved Fold {fold_num} ({len(train_global_indices)} train, {len(val_global_indices)} val) to [cyan]{fold_split_path}[/cyan]")

def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Prepare CARS thyroid dataset for training")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--visualize", action="store_true", help="Run visualization after preparation")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation steps")
    # --- K-FOLD ENHANCEMENT: New argument ---
    parser.add_argument("--k-folds", type=int, default=None, help="If specified, generates k-fold splits and a held-out test set.")
    
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]CARS Thyroid Dataset Preparation[/bold cyan]", border_style="blue"))
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        console.print(f"[red]Error: Directory {data_dir} does not exist![/red]")
        return
        
    # --- K-FOLD ENHANCEMENT: Main logic branch ---
    if args.k_folds:
        if args.k_folds < 2:
            console.print("[red]Error: --k-folds must be 2 or greater.[/red]")
            return
        generate_kfold_splits(data_dir, args.k_folds)
        console.print("\n[bold green]✓ K-fold split generation complete![/bold green]")
        return # End the script after generating splits

    # --- BACKWARD COMPATIBILITY: Original script flow ---
    console.print("\n[bold]Step 1: Scanning for images[/bold]")
    class_images = scan_data_directory(data_dir)
    if not class_images or 'unorganized' in class_images: return

    console.print("\n[bold]Step 2: Checking data organization[/bold]")
    organize_images(data_dir, class_images)

    if not args.skip_validation:
        console.print("\n[bold]Step 3: Validating images[/bold]")
        validate_image_properties(class_images)
        console.print("\n[bold]Step 4: Checking patient IDs[/bold]")
        check_patient_ids(class_images)
    
    console.print("\n[bold]Step 5: Creating dataset summary & default split[/bold]")
    dataset = create_dataset_summary(data_dir) # This uses the CARSThyroidDataset, which will create split_info.json if needed
    
    # The rest of your original main() continues here...
    if args.visualize and dataset:
        console.print("\n[bold]Step 6: Running visualization[/bold]")
        if Confirm.ask("Would you like to visualize the dataset?"):
            visualize_dataset(data_dir, Path("visualization_outputs"))
    
    console.print("\n[bold green]✓ Data preparation check complete![/bold green]")

if __name__ == "__main__":
    main()