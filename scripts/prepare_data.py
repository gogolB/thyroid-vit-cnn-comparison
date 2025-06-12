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


def main():
    """Main data preparation function."""
    
    parser = argparse.ArgumentParser(
        description="Prepare CARS thyroid dataset for training"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/raw",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Run visualization after preparation"
    )
    parser.add_argument(
        "--skip-validation", 
        action="store_true",
        help="Skip validation steps"
    )
    
    args = parser.parse_args()
    
    # Banner
    console.print(Panel.fit(
        "[bold cyan]CARS Thyroid Dataset Preparation[/bold cyan]\n"
        "[dim]Organizing and validating your microscopy images[/dim]",
        border_style="blue"
    ))
    
    data_dir = Path(args.data_dir)
    
    # Step 1: Check if directory exists
    if not data_dir.exists():
        console.print(f"[red]Error: Directory {data_dir} does not exist![/red]")
        console.print("[yellow]Please create the directory and add your images:[/yellow]")
        console.print("  mkdir -p data/raw/normal")
        console.print("  mkdir -p data/raw/cancerous")
        console.print("  # Copy your images to the appropriate directories")
        return
    
    # Step 2: Scan directory
    console.print("\n[bold]Step 1: Scanning for images[/bold]")
    class_images = scan_data_directory(data_dir)
    
    if not class_images:
        console.print("[red]No images found! Please add images to the directory.[/red]")
        return
    
    # Step 3: Check organization
    console.print("\n[bold]Step 2: Checking data organization[/bold]")
    organize_images(data_dir, class_images)
    
    if 'unorganized' in class_images:
        return  # Need to organize first
    
    # Step 4: Validate images
    if not args.skip_validation:
        console.print("\n[bold]Step 3: Validating images[/bold]")
        validate_image_properties(class_images)
        
        # Step 5: Check patient IDs
        console.print("\n[bold]Step 4: Checking patient IDs[/bold]")
        check_patient_ids(class_images)
    
    # Step 6: Create dataset and splits
    console.print("\n[bold]Step 5: Creating dataset splits[/bold]")
    dataset = create_dataset_summary(data_dir)
    
    if dataset:
        # Step 7: Create data loaders
        console.print("\n[bold]Step 6: Testing data loaders[/bold]")
        try:
            dataloaders = create_data_loaders(
                root_dir=data_dir,
                batch_size=32,
                num_workers=4,
                target_size=256,
                normalize=True,
                patient_level_split=False  # Disable patient-level splitting
            )
            console.print("[green]✓ Data loaders created successfully![/green]")
            
            # Test loading a batch
            for split, loader in dataloaders.items():
                images, labels = next(iter(loader))
                console.print(f"[green]✓ {split} loader: batch shape = {images.shape}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error creating data loaders: {e}[/red]")
    
    # Step 8: Show augmentation options
    console.print("\n[bold]Step 7: Available augmentation levels[/bold]")
    for level in ['light', 'medium', 'heavy']:
        console.print(f"\n[yellow]{level.upper()} augmentation:[/yellow]")
        print_augmentation_summary(level)
    
    # Step 9: Visualize if requested
    if args.visualize and dataset:
        console.print("\n[bold]Step 8: Running visualization[/bold]")
        if Confirm.ask("Would you like to visualize the dataset?"):
            output_dir = Path("visualization_outputs")
            visualize_dataset(data_dir, output_dir)
    
    # Final summary
    console.print("\n" + "="*60)
    console.print("[bold green]✓ Data preparation complete![/bold green]")
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("1. Review the generated splits in data/splits/split_info.json")
    console.print("2. Check visualization outputs (if generated)")
    console.print("3. Start training with: python train.py")
    console.print("\n[dim]Your data is ready for training![/dim]")


if __name__ == "__main__":
    main()