#!/usr/bin/env python3
"""
Generate sample CARS images for presentation.
Creates visualizations showing normal vs cancerous tissue examples.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CARSThyroidDataset
from rich.console import Console
import json

console = Console()


def create_sample_grid(dataset, n_samples=6, save_path=None):
    """Create a grid of sample images from each class."""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, n_samples, figure=fig, hspace=0.3, wspace=0.2)
    
    # Collect samples from each class
    normal_indices = []
    cancer_indices = []
    
    for idx in range(len(dataset)):
        label = dataset.labels[dataset.indices[idx]]
        if label == 0 and len(normal_indices) < n_samples:
            normal_indices.append(idx)
        elif label == 1 and len(cancer_indices) < n_samples:
            cancer_indices.append(idx)
        
        if len(normal_indices) >= n_samples and len(cancer_indices) >= n_samples:
            break
    
    # Plot normal samples
    for i, idx in enumerate(normal_indices):
        ax = fig.add_subplot(gs[0, i])
        img = dataset._load_image(idx)
        
        # Display image
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'Normal #{i+1}', fontsize=12, color='green', fontweight='bold')
        ax.axis('off')
        
        # Add scale bar on first image
        if i == 0:
            # Add 50μm scale bar (assuming 0.5μm/pixel)
            scalebar_length = 100  # pixels for 50μm
            ax.plot([10, 10+scalebar_length], [img.shape[0]-20, img.shape[0]-20],
                   'w-', linewidth=3)
            ax.text(10+scalebar_length/2, img.shape[0]-30, '50μm',
                   ha='center', va='top', color='white', fontsize=10)
    
    # Plot cancerous samples
    for i, idx in enumerate(cancer_indices):
        ax = fig.add_subplot(gs[1, i])
        img = dataset._load_image(idx)
        
        # Display image
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'Cancerous #{i+1}', fontsize=12, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('CARS Microscopy: Normal vs Cancerous Thyroid Tissue', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved sample grid to {save_path}[/green]")
    
    plt.close()


def create_feature_comparison(dataset, save_path=None):
    """Create a detailed comparison showing key visual differences."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get one good example from each class
    normal_idx = None
    cancer_idx = None
    
    # Find high-quality examples
    for idx in range(len(dataset)):
        img = dataset._load_image(idx)
        label = dataset.labels[dataset.indices[idx]]
        
        # Check if image has good contrast
        if np.std(img) > 100 and np.std(img) < 500:
            if label == 0 and normal_idx is None:
                normal_idx = idx
            elif label == 1 and cancer_idx is None:
                cancer_idx = idx
        
        if normal_idx is not None and cancer_idx is not None:
            break
    
    # Load images
    normal_img = dataset._load_image(normal_idx)
    cancer_img = dataset._load_image(cancer_idx)
    
    # Original images
    axes[0, 0].imshow(normal_img, cmap='gray')
    axes[0, 0].set_title('Normal Tissue', fontsize=14, color='green', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(cancer_img, cmap='gray')
    axes[1, 0].set_title('Cancerous Tissue', fontsize=14, color='red', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Zoomed regions (center crop)
    h, w = normal_img.shape
    crop_size = 128
    y, x = h//2 - crop_size//2, w//2 - crop_size//2
    
    normal_crop = normal_img[y:y+crop_size, x:x+crop_size]
    cancer_crop = cancer_img[y:y+crop_size, x:x+crop_size]
    
    axes[0, 1].imshow(normal_crop, cmap='gray')
    axes[0, 1].set_title('Zoomed Region', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(cancer_crop, cmap='gray')
    axes[1, 1].set_title('Zoomed Region', fontsize=12)
    axes[1, 1].axis('off')
    
    # Add rectangles to show zoom area
    rect = Rectangle((x, y), crop_size, crop_size, linewidth=2, 
                     edgecolor='yellow', facecolor='none')
    axes[0, 0].add_patch(rect)
    rect2 = Rectangle((x, y), crop_size, crop_size, linewidth=2, 
                      edgecolor='yellow', facecolor='none')
    axes[1, 0].add_patch(rect2)
    
    # Edge detection to show structure
    normal_edges = cv2.Canny(normal_crop.astype(np.uint8), 50, 150)
    cancer_edges = cv2.Canny(cancer_crop.astype(np.uint8), 50, 150)
    
    axes[0, 2].imshow(normal_edges, cmap='hot')
    axes[0, 2].set_title('Tissue Structure', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(cancer_edges, cmap='hot')
    axes[1, 2].set_title('Tissue Structure', fontsize=12)
    axes[1, 2].axis('off')
    
    # Add annotations
    fig.text(0.5, 0.95, 'Key Visual Features in CARS Imaging', 
             ha='center', fontsize=16, fontweight='bold')
    
    # Add text boxes with features
    normal_features = "• Organized follicular structure\n• Uniform intensity\n• Clear boundaries"
    cancer_features = "• Disrupted architecture\n• Heterogeneous intensity\n• Irregular patterns"
    
    fig.text(0.05, 0.45, normal_features, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    fig.text(0.05, 0.05, cancer_features, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved feature comparison to {save_path}[/green]")
    
    plt.close()


def create_intensity_distribution_plot(dataset, save_path=None):
    """Create intensity distribution plots for normal vs cancerous."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Collect intensity values
    normal_intensities = []
    cancer_intensities = []
    
    console.print("[cyan]Analyzing intensity distributions...[/cyan]")
    
    for idx in tqdm(range(min(50, len(dataset))), desc="Processing images"):
        img = dataset._load_image(idx)
        label = dataset.labels[dataset.indices[idx]]
        
        if label == 0:
            normal_intensities.extend(img.flatten())
        else:
            cancer_intensities.extend(img.flatten())
    
    # Subsample for plotting efficiency
    normal_sample = np.random.choice(normal_intensities, size=min(100000, len(normal_intensities)))
    cancer_sample = np.random.choice(cancer_intensities, size=min(100000, len(cancer_intensities)))
    
    # Histogram
    ax1.hist(normal_sample, bins=50, alpha=0.7, color='green', label='Normal', density=True)
    ax1.hist(cancer_sample, bins=50, alpha=0.7, color='red', label='Cancerous', density=True)
    ax1.set_xlabel('Pixel Intensity', fontsize=12)
    ax1.set_ylabel('Normalized Frequency', fontsize=12)
    ax1.set_title('Intensity Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [normal_sample, cancer_sample]
    bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Cancerous'], patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    
    ax2.set_ylabel('Pixel Intensity', fontsize=12)
    ax2.set_title('Intensity Range Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Normal: μ={np.mean(normal_sample):.1f}, σ={np.std(normal_sample):.1f}\n"
    stats_text += f"Cancer: μ={np.mean(cancer_sample):.1f}, σ={np.std(cancer_sample):.1f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('CARS Image Intensity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved intensity distribution to {save_path}[/green]")
    
    plt.close()


def create_dataset_overview(save_path=None):
    """Create an overview figure showing dataset statistics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dataset composition
    sizes = [225, 225]
    labels = ['Normal\n(225)', 'Cancerous\n(225)']
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 12})
    ax1.set_title('Dataset Composition', fontsize=14, fontweight='bold')
    
    # Data splits
    splits = [315, 68, 67]
    split_labels = ['Train\n(70%)', 'Val\n(15%)', 'Test\n(15%)']
    split_colors = ['#3498db', '#f39c12', '#9b59b6']
    
    wedges2, texts2, autotexts2 = ax2.pie(splits, labels=split_labels, colors=split_colors,
                                          autopct=lambda pct: f'{int(pct/100*450)}',
                                          startangle=45)
    ax2.set_title('Train/Val/Test Split', fontsize=14, fontweight='bold')
    
    # Augmentation impact
    original = 450
    augmented = 4500
    ax3.bar(['Original', 'Augmented'], [original, augmented], 
            color=['#34495e', '#2ecc71'], alpha=0.8)
    ax3.set_ylabel('Number of Images', fontsize=12)
    ax3.set_title('Data Augmentation Impact', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    ax3.text(0, original + 50, str(original), ha='center', fontsize=12, fontweight='bold')
    ax3.text(1, augmented + 50, str(augmented), ha='center', fontsize=12, fontweight='bold')
    ax3.text(1, augmented/2, '10×', ha='center', fontsize=20, fontweight='bold', color='white')
    
    # Key specifications
    specs_text = """
    Image Properties:
    • Format: CARS Microscopy
    • Original Size: 512×512 pixels
    • Bit Depth: 16-bit grayscale
    • Processed Size: 224×224 pixels
    
    Augmentations Applied:
    • Rotation: ±15°
    • Horizontal Flip
    • Zoom: ±20%
    • Intensity Variations
    
    Quality Distribution:
    • High Quality: 71%
    • Low Contrast: 9.1%
    • Artifacts: 14.2%
    • Extreme Dark: 5.8%
    """
    
    ax4.text(0.1, 0.9, specs_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Dataset Specifications', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('CARS Thyroid Dataset Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved dataset overview to {save_path}[/green]")
    
    plt.close()


def main():
    """Generate all sample visualizations."""
    # Create output directory
    output_dir = Path('visualizations/ppt-report')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Generating Sample CARS Images for Presentation[/bold cyan]\n")
    
    # Load dataset
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',  # Use test set for visualization
        target_size=512,  # Keep original size for visualization
        normalize=False,  # Keep original intensity values
        patient_level_split=False
    )
    
    console.print(f"[green]Loaded dataset with {len(dataset)} images[/green]\n")
    
    # Generate visualizations
    visualizations = [
        ('sample_grid.png', lambda: create_sample_grid(dataset, n_samples=6)),
        ('feature_comparison.png', lambda: create_feature_comparison(dataset)),
        ('intensity_distribution.png', lambda: create_intensity_distribution_plot(dataset)),
        ('dataset_overview.png', lambda: create_dataset_overview())
    ]
    
    for filename, func in visualizations:
        console.print(f"[cyan]Creating {filename}...[/cyan]")
        func(save_path=output_dir / filename)
    
    console.print("\n[bold green]✓ All sample visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


if __name__ == "__main__":
    main()
