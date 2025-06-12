"""
Visualization utilities for CARS microscopy images and augmentations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Union
import cv2

from rich.console import Console
from rich.progress import track
from rich.panel import Panel

# Try to import dataset and transforms
try:
    from src.data.dataset import CARSThyroidDataset
    from src.data.transforms import get_training_transforms, get_validation_transforms
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.data.dataset import CARSThyroidDataset
    from src.data.transforms import get_training_transforms, get_validation_transforms

console = Console()

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_image_statistics(dataset: CARSThyroidDataset, n_samples: int = 100):
    """
    Visualize statistics of the dataset images.
    
    Args:
        dataset: CARSThyroidDataset instance
        n_samples: Number of samples to analyze
    """
    
    console.print(f"[cyan]Analyzing {n_samples} images for statistics...[/cyan]")
    
    # Collect statistics
    means = []
    stds = []
    mins = []
    maxs = []
    
    for i in track(range(min(n_samples, len(dataset))), description="Processing images"):
        img = dataset._load_image(i)
        means.append(np.mean(img))
        stds.append(np.std(img))
        mins.append(np.min(img))
        maxs.append(np.max(img))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Image Statistics', fontsize=16)
    
    # Plot distributions
    axes[0, 0].hist(means, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Mean Pixel Values')
    axes[0, 0].set_xlabel('Mean Value')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(stds, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Standard Deviations')
    axes[0, 1].set_xlabel('Std Dev')
    axes[0, 1].set_ylabel('Frequency')
    
    axes[1, 0].hist(mins, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Minimum Values')
    axes[1, 0].set_xlabel('Min Value')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(maxs, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Maximum Values')
    axes[1, 1].set_xlabel('Max Value')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Print summary statistics
    console.print("\n[bold cyan]Image Statistics Summary:[/bold cyan]")
    console.print(f"Mean pixel value: {np.mean(means):.2f} ± {np.std(means):.2f}")
    console.print(f"Mean std dev: {np.mean(stds):.2f} ± {np.std(stds):.2f}")
    console.print(f"Min value range: [{np.min(mins):.0f}, {np.max(mins):.0f}]")
    console.print(f"Max value range: [{np.min(maxs):.0f}, {np.max(maxs):.0f}]")
    
    return fig


def visualize_samples(
    dataset: CARSThyroidDataset,
    n_samples: int = 8,
    save_path: Optional[Path] = None
):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset: CARSThyroidDataset instance
        n_samples: Number of samples to visualize
        save_path: Optional path to save the figure
    """
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    # Calculate grid size
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{dataset.split.capitalize()} Set Samples', fontsize=16)
    
    for idx, (ax_idx, data_idx) in enumerate(zip(np.ndindex(n_rows, n_cols), indices)):
        ax = axes[ax_idx]
        
        # Load and process image
        img, label = dataset[data_idx]
        
        # Convert to numpy for visualization
        if isinstance(img, torch.Tensor):
            img_np = img.squeeze().numpy()
        else:
            img_np = img
        
        # Display image
        im = ax.imshow(img_np, cmap='gray')
        ax.set_title(f'{"Normal" if label == 0 else "Cancerous"}', 
                     color='green' if label == 0 else 'red')
        ax.axis('off')
        
        # Add colorbar for first image
        if idx == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for idx in range(len(indices), n_rows * n_cols):
        ax_idx = (idx // n_cols, idx % n_cols)
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]✓ Saved figure to {save_path}[/green]")
    
    return fig


def visualize_augmentations(
    dataset: CARSThyroidDataset,
    sample_idx: int = 0,
    augmentation_levels: List[str] = ['none', 'light', 'medium', 'heavy']
):
    """
    Visualize the effect of different augmentation levels on a single image.
    
    Args:
        dataset: CARSThyroidDataset instance
        sample_idx: Index of sample to augment
        augmentation_levels: List of augmentation levels to show
    """
    
    # Load original image
    original_img = dataset._load_image(sample_idx)
    label = dataset.labels[dataset.indices[sample_idx]]
    
    fig, axes = plt.subplots(1, len(augmentation_levels), figsize=(4 * len(augmentation_levels), 4))
    if len(augmentation_levels) == 1:
        axes = [axes]
    
    fig.suptitle(f'Augmentation Levels on {"Normal" if label == 0 else "Cancerous"} Sample', 
                 fontsize=16)
    
    for idx, (ax, level) in enumerate(zip(axes, augmentation_levels)):
        # Get transform
        if level == 'none':
            transform = get_validation_transforms(target_size=256, normalize=True)
        else:
            transform = get_training_transforms(
                target_size=256, 
                normalize=True, 
                augmentation_level=level
            )
        
        # Apply transform
        img_tensor = torch.from_numpy(original_img).unsqueeze(0).float()
        augmented = transform(img_tensor)
        
        # Display
        img_np = augmented.squeeze().numpy()
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'{level.capitalize()}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_batch_diversity(
    dataset: CARSThyroidDataset,
    batch_size: int = 16,
    augmentation_level: str = 'medium'
):
    """
    Visualize a batch showing diversity of augmentations.
    
    Args:
        dataset: CARSThyroidDataset instance
        batch_size: Number of images in batch
        augmentation_level: Augmentation level to use
    """
    
    # Create temporary dataset with augmentation
    transform = get_training_transforms(
        target_size=256,
        normalize=True,
        augmentation_level=augmentation_level
    )
    
    dataset.transform = transform
    
    # Get batch
    images, labels = dataset.get_sample_batch(batch_size)
    
    # Calculate grid
    n_cols = int(np.sqrt(batch_size))
    n_rows = (batch_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = axes.flatten()
    
    fig.suptitle(f'Augmented Batch (Level: {augmentation_level})', fontsize=16)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[idx]
        img_np = img.squeeze().numpy()
        
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'{"N" if label == 0 else "C"}', 
                     color='green' if label == 0 else 'red',
                     fontsize=10)
        ax.axis('off')
    
    # Hide extra axes
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_augmentation_comparison_grid(
    dataset: CARSThyroidDataset,
    n_originals: int = 3,
    n_augmentations: int = 4
):
    """
    Create a grid showing original images and their augmented versions.
    
    Args:
        dataset: CARSThyroidDataset instance
        n_originals: Number of original images to show
        n_augmentations: Number of augmentations per image
    """
    
    # Get transform
    transform = get_training_transforms(
        target_size=256,
        normalize=True,
        augmentation_level='medium'
    )
    
    fig, axes = plt.subplots(n_originals, n_augmentations + 1, 
                            figsize=((n_augmentations + 1) * 3, n_originals * 3))
    
    fig.suptitle('Original vs Augmented Images', fontsize=16)
    
    # Get random samples
    indices = np.random.choice(len(dataset), n_originals, replace=False)
    
    for row_idx, data_idx in enumerate(indices):
        # Load original
        original_img = dataset._load_image(data_idx)
        label = dataset.labels[dataset.indices[data_idx]]
        
        # Preprocess original
        img_tensor = torch.from_numpy(original_img).unsqueeze(0).float()
        processed = dataset._preprocess_image(original_img)
        
        # Show original
        axes[row_idx, 0].imshow(processed.squeeze().numpy(), cmap='gray')
        axes[row_idx, 0].set_title('Original' if row_idx == 0 else '')
        axes[row_idx, 0].set_ylabel(f'{"Normal" if label == 0 else "Cancer"}',
                                   color='green' if label == 0 else 'red')
        axes[row_idx, 0].axis('off')
        
        # Show augmentations
        for col_idx in range(1, n_augmentations + 1):
            augmented = transform(img_tensor)
            axes[row_idx, col_idx].imshow(augmented.squeeze().numpy(), cmap='gray')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'Aug {col_idx}')
            axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution(dataset: CARSThyroidDataset):
    """
    Plot the class distribution in the dataset.
    
    Args:
        dataset: CARSThyroidDataset instance
    """
    
    # Count classes
    labels = dataset.labels[dataset.indices]
    unique, counts = np.unique(labels, return_counts=True)
    
    # Create pie chart and bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    class_names = ['Normal', 'Cancerous']
    
    wedges, texts, autotexts = ax1.pie(counts, labels=class_names, colors=colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{dataset.split.capitalize()} Set Class Distribution')
    
    # Bar chart
    bars = ax2.bar(class_names, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Sample Counts by Class')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Print summary
    console.print(f"\n[bold cyan]{dataset.split.capitalize()} Set Distribution:[/bold cyan]")
    for name, count in zip(class_names, counts):
        percentage = (count / len(dataset)) * 100
        console.print(f"{name}: {count} ({percentage:.1f}%)")
    
    return fig


# Main visualization function
def visualize_dataset(
    root_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
):
    """
    Run all visualizations on the dataset.
    
    Args:
        root_dir: Root directory of the dataset
        output_dir: Optional directory to save figures
    """
    
    console.print(Panel.fit(
        "[bold cyan]CARS Thyroid Dataset Visualization[/bold cyan]",
        border_style="blue"
    ))
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets for each split
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = CARSThyroidDataset(
                root_dir=root_dir,
                split=split,
                target_size=256,
                normalize=False  # We'll normalize in transforms
            )
            console.print(f"[green]✓ Loaded {split} dataset with {len(datasets[split])} images[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to load {split} dataset: {e}[/red]")
    
    if not datasets:
        console.print("[red]No datasets could be loaded. Please check your data directory.[/red]")
        return
    
    # Run visualizations
    figs = {}
    
    # 1. Dataset statistics
    if 'train' in datasets:
        console.print("\n[yellow]1. Generating dataset statistics...[/yellow]")
        figs['statistics'] = visualize_image_statistics(datasets['train'], n_samples=100)
        if output_dir:
            plt.savefig(output_dir / 'dataset_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 2. Sample images from each split
    for split, dataset in datasets.items():
        console.print(f"\n[yellow]2. Visualizing {split} samples...[/yellow]")
        figs[f'samples_{split}'] = visualize_samples(dataset, n_samples=8)
        if output_dir:
            plt.savefig(output_dir / f'samples_{split}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 3. Augmentation effects
    if 'train' in datasets:
        console.print("\n[yellow]3. Visualizing augmentation levels...[/yellow]")
        figs['augmentations'] = visualize_augmentations(datasets['train'])
        if output_dir:
            plt.savefig(output_dir / 'augmentation_levels.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Augmentation comparison grid
        console.print("\n[yellow]4. Creating augmentation comparison grid...[/yellow]")
        figs['aug_grid'] = create_augmentation_comparison_grid(datasets['train'])
        if output_dir:
            plt.savefig(output_dir / 'augmentation_grid.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 5. Augmented batch
        console.print("\n[yellow]5. Visualizing augmented batch...[/yellow]")
        figs['batch'] = visualize_batch_diversity(datasets['train'])
        if output_dir:
            plt.savefig(output_dir / 'augmented_batch.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 6. Class distributions
    for split, dataset in datasets.items():
        console.print(f"\n[yellow]6. Plotting {split} class distribution...[/yellow]")
        figs[f'dist_{split}'] = plot_class_distribution(dataset)
        if output_dir:
            plt.savefig(output_dir / f'class_distribution_{split}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    console.print("\n[green]✓ Visualization complete![/green]")
    if output_dir:
        console.print(f"[green]✓ Figures saved to {output_dir}[/green]")


if __name__ == "__main__":
    # Run visualization
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CARS thyroid dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw", 
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="visualization_outputs",
                       help="Directory to save visualization outputs")
    
    args = parser.parse_args()
    
    visualize_dataset(args.data_dir, args.output_dir)
