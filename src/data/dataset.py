"""
CARS Microscopy Image Dataset for Thyroid Classification
Handles 512x512 single-channel uint16 images
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import hashlib
import json

# Rich imports for beautiful progress bars
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

# Import our console utilities
try:
    from src.utils.console import console, create_progress_bar, print_data_summary
except ImportError:
    # Fallback if running standalone
    from rich.console import Console
    console = Console()
    
    def create_progress_bar(description: str = "Processing"):
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )


class CARSThyroidDataset(Dataset):
    """
    PyTorch Dataset for CARS Thyroid Microscopy Images
    
    Args:
        root_dir: Root directory containing the images
        split: One of 'train', 'val', 'test', or 'all'
        transform: Optional transform to be applied on images
        target_size: Target size for images (default: 256)
        normalize: Whether to normalize uint16 to [0, 1]
        cache_images: Whether to cache images in memory
        patient_level_split: Whether to split by patient ID
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: int = 256,
        normalize: bool = True,
        cache_images: bool = False,
        patient_level_split: bool = False,  # Changed default to False
        split_info_file: Optional[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.cache_images = cache_images
        self.patient_level_split = patient_level_split
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        # Load or create split information
        self.split_info_file = split_info_file or self.root_dir.parent / 'splits' / 'split_info.json'
        
        # Initialize dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset and create/load splits."""
        console.print(f"[cyan]Loading CARS dataset from:[/cyan] {self.root_dir}")
        
        # Find all image files
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        
        # Expected directory structure: root_dir/class_name/images
        for class_idx, class_name in enumerate(['normal', 'cancerous']):
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                console.print(f"[yellow]Warning:[/yellow] Directory {class_dir} not found")
                continue
            
            # Support multiple image formats
            supported_formats = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
            class_images = []
            
            for fmt in supported_formats:
                class_images.extend(list(class_dir.glob(fmt)))
            
            console.print(f"[green]Found {len(class_images)} images for class '{class_name}'[/green]")
            
            for img_path in class_images:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                
                # Extract patient ID from filename
                # For files like normal_150.tif or cancer_220.tif, use the number as unique ID
                if '_' in img_path.stem:
                    # Extract the number after underscore as unique identifier
                    parts = img_path.stem.split('_')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        patient_id = f"{class_name}_{parts[-1]}"
                    else:
                        patient_id = img_path.stem
                else:
                    patient_id = img_path.stem
                self.patient_ids.append(patient_id)
        
        # Convert to numpy arrays
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        self.patient_ids = np.array(self.patient_ids)
        
        # Create or load splits
        self._create_splits()
        
    def _create_splits(self):
        """Create train/val/test splits, ensuring patient-level splitting if requested."""
        
        if self.split == 'all':
            self.indices = np.arange(len(self.image_paths))
            return
        
        # Check if split file exists
        if self.split_info_file.exists():
            console.print(f"[cyan]Loading existing splits from:[/cyan] {self.split_info_file}")
            with open(self.split_info_file, 'r') as f:
                split_data = json.load(f)
            
            self.indices = np.array(split_data[self.split])
        else:
            console.print("[yellow]Creating new train/val/test splits...[/yellow]")
            self._generate_splits()
    
    def _generate_splits(self):
        """Generate new train/val/test splits."""
        
        # Create directory for split info
        self.split_info_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.patient_level_split:
            # Split by patient to avoid data leakage
            unique_patients = np.unique(self.patient_ids)
            patient_labels = {}
            
            # Get majority class for each patient
            for patient in unique_patients:
                patient_mask = self.patient_ids == patient
                patient_label = np.bincount(self.labels[patient_mask]).argmax()
                patient_labels[patient] = patient_label
            
            # Split patients
            patients = list(patient_labels.keys())
            labels = list(patient_labels.values())
            
            # First split: train+val vs test (85% vs 15%)
            train_val_patients, test_patients = train_test_split(
                patients, test_size=0.15, stratify=labels, random_state=42
            )
            
            # Second split: train vs val (70% vs 15% of total)
            train_val_labels = [patient_labels[p] for p in train_val_patients]
            train_patients, val_patients = train_test_split(
                train_val_patients, test_size=0.176, stratify=train_val_labels, random_state=42
            )
            
            # Get indices for each split
            train_indices = [i for i, p in enumerate(self.patient_ids) if p in train_patients]
            val_indices = [i for i, p in enumerate(self.patient_ids) if p in val_patients]
            test_indices = [i for i, p in enumerate(self.patient_ids) if p in test_patients]
            
        else:
            # Standard random split
            indices = np.arange(len(self.image_paths))
            
            # Stratified split
            train_val_indices, test_indices = train_test_split(
                indices, test_size=0.15, stratify=self.labels, random_state=42
            )
            
            train_indices, val_indices = train_test_split(
                train_val_indices, test_size=0.176, stratify=self.labels[train_val_indices], random_state=42
            )
        
        # Save split information
        split_data = {
            'train': train_indices.tolist() if isinstance(train_indices, np.ndarray) else train_indices,
            'val': val_indices.tolist() if isinstance(val_indices, np.ndarray) else val_indices,
            'test': test_indices.tolist() if isinstance(test_indices, np.ndarray) else test_indices,
            'metadata': {
                'total_images': len(self.image_paths),
                'patient_level_split': self.patient_level_split,
                'split_ratios': {
                    'train': len(train_indices) / len(self.image_paths),
                    'val': len(val_indices) / len(self.image_paths),
                    'test': len(test_indices) / len(self.image_paths)
                }
            }
        }
        
        with open(self.split_info_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Set indices for current split
        if self.split == 'train':
            self.indices = np.array(train_indices)
        elif self.split == 'val':
            self.indices = np.array(val_indices)
        elif self.split == 'test':
            self.indices = np.array(test_indices)
        
        # Print split summary
        self._print_split_summary(train_indices, val_indices, test_indices)
    
    def _print_split_summary(self, train_indices, val_indices, test_indices):
        """Print a summary of the data splits."""
        
        splits = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        table = Table(title="Data Split Summary", show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan", no_wrap=True)
        table.add_column("Total", style="white")
        table.add_column("Normal", style="green")
        table.add_column("Cancerous", style="red")
        table.add_column("Percentage", style="yellow")
        
        for split_name, indices in splits.items():
            split_labels = self.labels[indices]
            normal_count = (split_labels == 0).sum()
            cancer_count = (split_labels == 1).sum()
            total = len(indices)
            percentage = (total / len(self.image_paths)) * 100
            
            table.add_row(
                split_name.capitalize(),
                str(total),
                str(normal_count),
                str(cancer_count),
                f"{percentage:.1f}%"
            )
        
        console.print(table)
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load a single image."""
        
        # Check cache first
        if self.cache_images and idx in self.image_cache:
            return self.image_cache[idx]
        
        img_path = self.image_paths[self.indices[idx]]
        
        # Load image based on format
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            # Load TIFF (preserving uint16)
            img = tifffile.imread(str(img_path))
        else:
            # Load other formats
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                # Try with PIL
                img = np.array(Image.open(img_path))
        
        # Ensure single channel
        if len(img.shape) == 3:
            # Convert to grayscale if needed
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = img[:, :, 0]
        
        # Ensure uint16
        if img.dtype != np.uint16:
            if img.dtype == np.uint8:
                img = img.astype(np.uint16) * 257  # Scale to uint16 range
            else:
                img = img.astype(np.uint16)
        
        # Cache if requested
        if self.cache_images:
            self.image_cache[idx] = img
        
        return img
    
    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess image: resize and normalize."""
        
        # Resize if needed
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            img = img.astype(np.float32) / 65535.0  # Normalize uint16 to [0, 1]
        else:
            img = img.astype(np.float32)
        
        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: [1, H, W]
        
        return img_tensor
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample."""
        
        # Load image
        img = self._load_image(idx)
        
        # Preprocess
        img_tensor = self._preprocess_image(img)
        
        # Apply transforms if any
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Get label
        label = self.labels[self.indices[idx]]
        
        return img_tensor, label
    
    def get_sample_batch(self, n_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of random samples for visualization."""
        indices = np.random.choice(len(self), n_samples, replace=False)
        
        images = []
        labels = []
        
        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels.append(label)
        
        return torch.stack(images), torch.tensor(labels)
    
    def cache_all_images(self):
        """Pre-cache all images in memory."""
        if not self.cache_images:
            console.print("[yellow]Warning: cache_images is False. Enabling caching.[/yellow]")
            self.cache_images = True
            self.image_cache = {}
        
        with create_progress_bar("Caching images") as progress:
            task = progress.add_task("[cyan]Loading images into memory...", total=len(self))
            
            for i in range(len(self)):
                _ = self._load_image(i)
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Cached {len(self.image_cache)} images in memory[/green]")


def create_data_loaders(
    root_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        root_dir: Root directory containing the images
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        transform_train: Transform for training data
        transform_val: Transform for validation/test data
        **dataset_kwargs: Additional arguments for CARSThyroidDataset
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    
    console.print(Panel.fit(
        "[bold cyan]Creating CARS Thyroid Data Loaders[/bold cyan]",
        border_style="blue"
    ))
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        transform = transform_train if split == 'train' else transform_val
        
        dataset = CARSThyroidDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            **dataset_kwargs
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')
        )
    
    # Print summary
    total_images = len(dataloaders['train'].dataset) + len(dataloaders['val'].dataset) + len(dataloaders['test'].dataset)
    console.print(f"\n[bold]Dataset Summary:[/bold]")
    console.print(f"Total images: {total_images}")
    console.print(f"Image size: ({dataset_kwargs.get('target_size', 256)}, {dataset_kwargs.get('target_size', 256)}, 1)")
    console.print(f"Splits:")
    console.print(f"  Train: {len(dataloaders['train'].dataset)} images")
    console.print(f"  Val: {len(dataloaders['val'].dataset)} images")
    console.print(f"  Test: {len(dataloaders['test'].dataset)} images")
    
    return dataloaders


# Demo function
def demo_dataset():
    """Demo the dataset functionality."""
    
    # This is for testing - update with your actual data path
    root_dir = Path("data/raw")
    
    # Create sample data structure if it doesn't exist
    if not root_dir.exists():
        console.print("[yellow]Creating sample data structure...[/yellow]")
        for class_name in ['normal', 'cancerous']:
            (root_dir / class_name).mkdir(parents=True, exist_ok=True)
        console.print("[green]✓ Created sample directories[/green]")
        console.print("[yellow]Please add your CARS images to:[/yellow]")
        console.print(f"  - {root_dir}/normal/")
        console.print(f"  - {root_dir}/cancerous/")
        return
    
    # Create dataset
    dataset = CARSThyroidDataset(
        root_dir=root_dir,
        split='train',
        target_size=256,
        normalize=True,
        cache_images=False,
        patient_level_split=False
    )
    
    if len(dataset) > 0:
        # Get a sample
        img, label = dataset[0]
        console.print(f"\n[cyan]Sample image shape:[/cyan] {img.shape}")
        console.print(f"[cyan]Label:[/cyan] {label} ({'normal' if label == 0 else 'cancerous'})")
        console.print(f"[cyan]Data type:[/cyan] {img.dtype}")
        console.print(f"[cyan]Value range:[/cyan] [{img.min():.3f}, {img.max():.3f}]")
    
    # Create data loaders
    dataloaders = create_data_loaders(
        root_dir=root_dir,
        batch_size=32,
        num_workers=4,
        target_size=256,
        normalize=True,
        patient_level_split=False
    )