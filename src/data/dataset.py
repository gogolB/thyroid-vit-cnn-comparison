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
        fold: (Optional) The specific fold to load for cross-validation (e.g., 1 to k).
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: int = 256,
        normalize: bool = True,
        cache_images: bool = False,
        patient_level_split: bool = False,
        fold: Optional[int] = None,
        split_info_file: Optional[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.normalize = normalize
        self.cache_images = cache_images
        self.patient_level_split = patient_level_split
        self.fold = fold
        
        self.image_cache = {} if cache_images else None
        
        self.splits_dir = self.root_dir.parent / 'splits'
        
        # --- DEFINITIVE SPLIT FILE LOGIC ---
        if self.fold is not None:
            # K-fold mode: Load a specific train/val fold
            self.split_info_file = self.splits_dir / f'split_fold_{self.fold}.json'
        else:
            # Backward-compatible single-split mode
            self.split_info_file = split_info_file or self.splits_dir / 'split_info.json'
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset and create/load splits."""
        console.print(f"[cyan]Loading CARS dataset from:[/cyan] {self.root_dir}")
        
        self.image_paths = []
        self.labels = []
        self.patient_ids = []
        
        for class_idx, class_name in enumerate(['normal', 'cancerous']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists(): continue
            
            supported_formats = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
            class_images = [p for fmt in supported_formats for p in class_dir.glob(fmt)]
            console.print(f"[green]Found {len(class_images)} images for class '{class_name}'[/green]")
            
            for img_path in class_images:
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                if '_' in img_path.stem and img_path.stem.split('_')[-1].isdigit():
                    self.patient_ids.append(f"{class_name}_{img_path.stem.split('_')[-1]}")
                else:
                    self.patient_ids.append(img_path.stem)
        
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
        self.patient_ids = np.array(self.patient_ids)
        
        self._create_splits()
        
    def _create_splits(self):
        """Load splits from the appropriate file."""
        if self.split == 'all':
            self.indices = np.arange(len(self.image_paths))
            return
        
        if self.split_info_file.exists():
            console.print(f"[cyan]Loading splits from:[/cyan] {self.split_info_file}")
            with open(self.split_info_file, 'r') as f:
                split_data = json.load(f)
            self.indices = np.array(split_data[self.split])
        else:
            if self.fold is not None or self.split == 'test':
                raise FileNotFoundError(
                    f"Required split file not found: {self.split_info_file}\n"
                    f"Please run 'python scripts/prepare_data.py --k-folds N' to generate all necessary split files."
                )
            
            console.print("[yellow]Default split file not found. Generating new single train/val/test splits...[/yellow]")
            self._generate_splits()
    
    def _generate_splits(self):
        """Generate a single default train/val/test split (original functionality)."""
        self.split_info_file.parent.mkdir(parents=True, exist_ok=True)
        
        indices = np.arange(len(self.image_paths))
        train_val_indices, test_indices = train_test_split(indices, test_size=0.15, stratify=self.labels, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.176, stratify=self.labels[train_val_indices], random_state=42)
        
        split_data = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist(),
        }
        
        with open(self.split_info_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        self.indices = np.array(split_data[self.split])
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
    fold: Optional[int] = None,
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
            fold=fold,
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