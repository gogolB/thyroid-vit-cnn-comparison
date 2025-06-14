#!/usr/bin/env python3
"""
Generate sample CARS images for presentation.
Creates visualizations showing normal vs cancerous tissue examples with quality-aware preprocessing.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import cv2
from tqdm import tqdm
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CARSThyroidDataset
from src.data.quality_preprocessing import create_quality_aware_transform
from rich.console import Console
import json

console = Console()


def create_sample_grid(dataset_raw, dataset_processed, n_samples=6, save_path=None):
    """Create a grid of sample images from each class showing raw and processed."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(4, n_samples, figure=fig, hspace=0.3, wspace=0.2)
    
    # Collect samples from each class
    normal_indices = []
    cancer_indices = []
    
    for idx in range(len(dataset_raw)):
        label = dataset_raw.labels[dataset_raw.indices[idx]]
        if label == 0 and len(normal_indices) < n_samples:
            normal_indices.append(idx)
        elif label == 1 and len(cancer_indices) < n_samples:
            cancer_indices.append(idx)
        
        if len(normal_indices) >= n_samples and len(cancer_indices) >= n_samples:
            break
    
    # Plot normal samples - raw
    for i, idx in enumerate(normal_indices):
        ax = fig.add_subplot(gs[0, i])
        img = dataset_raw._load_image(idx)
        
        # Display raw image
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'Normal #{i+1}\n(Raw)', fontsize=11, color='green', fontweight='bold')
        ax.axis('off')
        
        # Add scale bar on first image
        if i == 0:
            scalebar_length = 100  # pixels for 50μm
            ax.plot([10, 10+scalebar_length], [img.shape[0]-20, img.shape[0]-20],
                   'w-', linewidth=3)
            ax.text(10+scalebar_length/2, img.shape[0]-30, '50μm',
                   ha='center', va='top', color='white', fontsize=10)
    
    # Plot normal samples - processed
    for i, idx in enumerate(normal_indices):
        ax = fig.add_subplot(gs[1, i])
        processed_img, _ = dataset_processed[idx]
        
        if isinstance(processed_img, torch.Tensor):
            processed_img = processed_img.squeeze().numpy()
        
        # Apply contrast adjustment for visualization
        if processed_img.max() <= 1:
            p2, p98 = np.percentile(processed_img, (2, 98))
            processed_display = np.clip((processed_img - p2) / (p98 - p2), 0, 1)
        else:
            processed_display = processed_img
        
        im = ax.imshow(processed_display, cmap='gray')
        ax.set_title(f'Normal #{i+1}\n(Processed)', fontsize=11, color='green')
        ax.axis('off')
    
    # Plot cancerous samples - raw
    for i, idx in enumerate(cancer_indices):
        ax = fig.add_subplot(gs[2, i])
        img = dataset_raw._load_image(idx)
        
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=65535)
        ax.set_title(f'Cancerous #{i+1}\n(Raw)', fontsize=11, color='red', fontweight='bold')
        ax.axis('off')
    
    # Plot cancerous samples - processed
    for i, idx in enumerate(cancer_indices):
        ax = fig.add_subplot(gs[3, i])
        processed_img, _ = dataset_processed[idx]
        
        if isinstance(processed_img, torch.Tensor):
            processed_img = processed_img.squeeze().numpy()
        
        # Apply contrast adjustment for visualization
        if processed_img.max() <= 1:
            p2, p98 = np.percentile(processed_img, (2, 98))
            processed_display = np.clip((processed_img - p2) / (p98 - p2), 0, 1)
        else:
            processed_display = processed_img
        
        im = ax.imshow(processed_display, cmap='gray')
        ax.set_title(f'Cancerous #{i+1}\n(Processed)', fontsize=11, color='red')
        ax.axis('off')
    
    plt.suptitle('CARS Microscopy: Raw vs Quality-Aware Preprocessed Images', 
                 fontsize=16, fontweight='bold')
    
    # Add description
    fig.text(0.5, 0.02, 'Quality-aware preprocessing includes: CLAHE enhancement, adaptive denoising, and percentile normalization',
             ha='center', fontsize=10, style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved sample grid to {save_path}[/green]")
    
    plt.close()


def create_feature_comparison(dataset_raw, dataset_processed, save_path=None):
    """Create a detailed comparison showing key visual differences with preprocessing."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Get one sample from each class
    normal_idx = None
    cancer_idx = None
    
    for idx in range(len(dataset_raw)):
        label = dataset_raw.labels[dataset_raw.indices[idx]]
        if label == 0 and normal_idx is None:
            normal_idx = idx
        elif label == 1 and cancer_idx is None:
            cancer_idx = idx
        
        if normal_idx is not None and cancer_idx is not None:
            break
    
    # Load raw images
    normal_raw = dataset_raw._load_image(normal_idx)
    cancer_raw = dataset_raw._load_image(cancer_idx)
    
    # Load processed images
    normal_proc, _ = dataset_processed[normal_idx]
    cancer_proc, _ = dataset_processed[cancer_idx]
    
    if isinstance(normal_proc, torch.Tensor):
        normal_proc = normal_proc.squeeze().numpy()
    if isinstance(cancer_proc, torch.Tensor):
        cancer_proc = cancer_proc.squeeze().numpy()
    
    # Apply contrast adjustment for processed images
    if normal_proc.max() <= 1:
        p2, p98 = np.percentile(normal_proc, (2, 98))
        normal_proc_display = np.clip((normal_proc - p2) / (p98 - p2), 0, 1)
    else:
        normal_proc_display = normal_proc
        
    if cancer_proc.max() <= 1:
        p2, p98 = np.percentile(cancer_proc, (2, 98))
        cancer_proc_display = np.clip((cancer_proc - p2) / (p98 - p2), 0, 1)
    else:
        cancer_proc_display = cancer_proc
    
    # Row 0: Raw images
    axes[0, 0].imshow(normal_raw, cmap='gray', vmin=0, vmax=65535)
    axes[0, 0].set_title('Normal Tissue\n(Raw)', fontsize=12, color='green', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cancer_raw, cmap='gray', vmin=0, vmax=65535)
    axes[0, 1].set_title('Cancerous Tissue\n(Raw)', fontsize=12, color='red', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 0: Processed images
    axes[0, 2].imshow(normal_proc_display, cmap='gray')
    axes[0, 2].set_title('Normal Tissue\n(Processed)', fontsize=12, color='green')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(cancer_proc_display, cmap='gray')
    axes[0, 3].set_title('Cancerous Tissue\n(Processed)', fontsize=12, color='red')
    axes[0, 3].axis('off')
    
    # Row 1: Zoomed regions
    zoom_size = 128
    center = normal_raw.shape[0] // 2
    
    normal_zoom_raw = normal_raw[center-zoom_size//2:center+zoom_size//2,
                                 center-zoom_size//2:center+zoom_size//2]
    cancer_zoom_raw = cancer_raw[center-zoom_size//2:center+zoom_size//2,
                                 center-zoom_size//2:center+zoom_size//2]
    
    # For processed images, we need to handle the smaller size
    if normal_proc_display.shape[0] < normal_raw.shape[0]:
        # Images are already smaller, adjust zoom accordingly
        zoom_size_proc = zoom_size * normal_proc_display.shape[0] // normal_raw.shape[0]
        center_proc = normal_proc_display.shape[0] // 2
        normal_zoom_proc = normal_proc_display[center_proc-zoom_size_proc//2:center_proc+zoom_size_proc//2,
                                              center_proc-zoom_size_proc//2:center_proc+zoom_size_proc//2]
        cancer_zoom_proc = cancer_proc_display[center_proc-zoom_size_proc//2:center_proc+zoom_size_proc//2,
                                              center_proc-zoom_size_proc//2:center_proc+zoom_size_proc//2]
    else:
        normal_zoom_proc = normal_proc_display[center-zoom_size//2:center+zoom_size//2,
                                              center-zoom_size//2:center+zoom_size//2]
        cancer_zoom_proc = cancer_proc_display[center-zoom_size//2:center+zoom_size//2,
                                              center-zoom_size//2:center+zoom_size//2]
    
    axes[1, 0].imshow(normal_zoom_raw, cmap='gray', vmin=0, vmax=65535)
    axes[1, 0].set_title('Zoomed Normal (Raw)', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cancer_zoom_raw, cmap='gray', vmin=0, vmax=65535)
    axes[1, 1].set_title('Zoomed Cancer (Raw)', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(normal_zoom_proc, cmap='gray')
    axes[1, 2].set_title('Zoomed Normal (Proc)', fontsize=11)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(cancer_zoom_proc, cmap='gray')
    axes[1, 3].set_title('Zoomed Cancer (Proc)', fontsize=11)
    axes[1, 3].axis('off')
    
    # Row 2: Edge detection on processed images
    normal_edges = cv2.Canny((normal_proc_display * 255).astype(np.uint8), 50, 150)
    cancer_edges = cv2.Canny((cancer_proc_display * 255).astype(np.uint8), 50, 150)
    
    axes[2, 0].imshow(normal_edges, cmap='hot')
    axes[2, 0].set_title('Normal Structure', fontsize=11)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(cancer_edges, cmap='hot')
    axes[2, 1].set_title('Cancer Structure', fontsize=11)
    axes[2, 1].axis('off')
    
    # Hide unused subplots
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')
    
    # Add text boxes with features
    normal_features = "Normal:\n• Organized follicles\n• Uniform intensity\n• Clear boundaries"
    cancer_features = "Cancer:\n• Disrupted structure\n• Heterogeneous\n• Irregular patterns"
    
    axes[2, 2].text(0.1, 0.5, normal_features, fontsize=11, color='green',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                    transform=axes[2, 2].transAxes)
    
    axes[2, 3].text(0.1, 0.5, cancer_features, fontsize=11, color='red',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
                    transform=axes[2, 3].transAxes)
    
    fig.suptitle('Visual Feature Analysis: Raw vs Quality-Aware Preprocessed', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved feature comparison to {save_path}[/green]")
    
    plt.close()


def create_intensity_distribution_plot(dataset_raw, dataset_processed=None, save_path=None):
    """Create intensity distribution plots for normal vs cancerous (raw and processed)."""
    if dataset_processed is None:
        # Single dataset mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        # Dual dataset mode
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Collect intensity values from raw dataset
    normal_intensities = []
    cancer_intensities = []
    
    console.print("[cyan]Analyzing raw intensity distributions...[/cyan]")
    
    for idx in tqdm(range(min(50, len(dataset_raw))), desc="Processing raw images"):
        img = dataset_raw._load_image(idx)
        label = dataset_raw.labels[dataset_raw.indices[idx]]
        
        if label == 0:
            normal_intensities.extend(img.flatten())
        else:
            cancer_intensities.extend(img.flatten())
    
    # Subsample for plotting efficiency
    normal_sample = np.random.choice(normal_intensities, size=min(100000, len(normal_intensities)))
    cancer_sample = np.random.choice(cancer_intensities, size=min(100000, len(cancer_intensities)))
    
    # Raw histogram
    ax1.hist(normal_sample, bins=50, alpha=0.7, color='green', label='Normal', density=True)
    ax1.hist(cancer_sample, bins=50, alpha=0.7, color='red', label='Cancerous', density=True)
    ax1.set_xlabel('Pixel Intensity', fontsize=12)
    ax1.set_ylabel('Normalized Frequency', fontsize=12)
    ax1.set_title('Raw Image Intensity Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Raw box plot
    data_to_plot = [normal_sample, cancer_sample]
    bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Cancerous'], patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.7)
    
    ax2.set_ylabel('Pixel Intensity', fontsize=12)
    ax2.set_title('Raw Intensity Range Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Normal: μ={np.mean(normal_sample):.1f}, σ={np.std(normal_sample):.1f}\n"
    stats_text += f"Cancer: μ={np.mean(cancer_sample):.1f}, σ={np.std(cancer_sample):.1f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # If we have processed dataset, show processed distributions too
    if dataset_processed is not None:
        normal_proc = []
        cancer_proc = []
        
        console.print("[cyan]Analyzing processed intensity distributions...[/cyan]")
        
        for idx in tqdm(range(min(50, len(dataset_processed))), desc="Processing preprocessed images"):
            img, label = dataset_processed[idx]
            if isinstance(img, torch.Tensor):
                img = img.squeeze().numpy()
            
            if label == 0:
                normal_proc.extend(img.flatten())
            else:
                cancer_proc.extend(img.flatten())
        
        # Subsample
        normal_proc_sample = np.random.choice(normal_proc, size=min(100000, len(normal_proc)))
        cancer_proc_sample = np.random.choice(cancer_proc, size=min(100000, len(cancer_proc)))
        
        # Processed histogram
        ax3.hist(normal_proc_sample, bins=50, alpha=0.7, color='green', label='Normal', density=True)
        ax3.hist(cancer_proc_sample, bins=50, alpha=0.7, color='red', label='Cancerous', density=True)
        ax3.set_xlabel('Normalized Intensity', fontsize=12)
        ax3.set_ylabel('Normalized Frequency', fontsize=12)
        ax3.set_title('Processed Image Intensity Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Processed box plot
        data_proc = [normal_proc_sample, cancer_proc_sample]
        bp2 = ax4.boxplot(data_proc, labels=['Normal', 'Cancerous'], patch_artist=True)
        
        bp2['boxes'][0].set_facecolor('green')
        bp2['boxes'][0].set_alpha(0.7)
        bp2['boxes'][1].set_facecolor('red')
        bp2['boxes'][1].set_alpha(0.7)
        
        ax4.set_ylabel('Normalized Intensity', fontsize=12)
        ax4.set_title('Processed Intensity Range Comparison', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        stats_text2 = f"Normal: μ={np.mean(normal_proc_sample):.3f}, σ={np.std(normal_proc_sample):.3f}\n"
        stats_text2 += f"Cancer: μ={np.mean(cancer_proc_sample):.3f}, σ={np.std(cancer_proc_sample):.3f}"
        ax4.text(0.02, 0.98, stats_text2, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('CARS Image Intensity Analysis: Impact of Quality-Aware Preprocessing', 
                 fontsize=16, fontweight='bold')
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
    • Model Input Size: 224×224 pixels
    
    Quality-Aware Preprocessing:
    • Issue Detection & Correction
    • CLAHE Enhancement
    • Adaptive Denoising
    • Percentile Normalization (1-99%)
    
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
    
    # Import quality-aware preprocessing
    from src.data.quality_preprocessing import create_quality_aware_transform
    
    # Create transform for visualization
    transform = create_quality_aware_transform(
        target_size=224,  # Use the actual model input size
        quality_report_path=None,
        augmentation_level='none',
        split='val'
    )
    
    # Load dataset with quality-aware preprocessing
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',  # Use test set for visualization
        target_size=224,  # Model input size
        transform=transform,  # Apply quality-aware preprocessing
        normalize=False,  # Keep values in original range for better visualization
        patient_level_split=False
    )
    
    # Also create a raw dataset for comparison
    dataset_raw = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',
        target_size=512,  # Keep original size
        transform=None,
        normalize=False,
        patient_level_split=False
    )
    
    console.print(f"[green]Loaded dataset with {len(dataset)} images[/green]")
    console.print(f"[cyan]Using quality-aware preprocessing for visualization[/cyan]\n")
    
    # Generate visualizations
    visualizations = [
        ('sample_grid.png', lambda save_path: create_sample_grid(dataset_raw, dataset, n_samples=6, save_path=save_path)),
        ('preprocessing_comparison.png', lambda save_path: create_preprocessing_comparison(dataset_raw, dataset, save_path=save_path)),
        ('feature_comparison.png', lambda save_path: create_feature_comparison(dataset_raw, dataset, save_path=save_path)),
        ('intensity_distribution.png', lambda save_path: create_intensity_distribution_plot(dataset_raw, dataset, save_path=save_path)),
        ('dataset_overview.png', lambda save_path: create_dataset_overview(save_path=save_path))
    ]
    
    for filename, func in visualizations:
        console.print(f"[cyan]Creating {filename}...[/cyan]")
        func(save_path=output_dir / filename)
    
    console.print("\n[bold green]✓ All sample visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


def create_preprocessing_comparison(dataset_raw, dataset_processed, save_path=None):
    """Compare raw vs preprocessed images showing quality-aware corrections."""
    from src.data.quality_preprocessing import QualityAwarePreprocessor
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Create preprocessor to identify issues
    preprocessor = QualityAwarePreprocessor()
    
    # Find examples of different quality issues
    examples = {
        'high_quality': None,
        'extreme_dark': None,
        'low_contrast': None,
        'artifacts': None
    }
    
    console.print("[cyan]Finding examples of different quality issues...[/cyan]")
    
    for idx in range(min(100, len(dataset_raw))):
        if all(v is not None for v in examples.values()):
            break
            
        img = dataset_raw._load_image(idx)
        issues = preprocessor.identify_quality_issues(img)
        
        if not issues and examples['high_quality'] is None:
            examples['high_quality'] = idx
        elif 'extreme_dark' in issues and examples['extreme_dark'] is None:
            examples['extreme_dark'] = idx
        elif 'low_contrast' in issues and examples['low_contrast'] is None:
            examples['low_contrast'] = idx
        elif 'potential_artifacts' in issues and examples['artifacts'] is None:
            examples['artifacts'] = idx
    
    # Plot examples
    categories = [
        ('high_quality', 'High Quality', 0),
        ('extreme_dark', 'Extreme Dark', 1),
        ('low_contrast', 'Low Contrast', 2),
        ('artifacts', 'With Artifacts', 3)
    ]
    
    for cat_key, cat_name, col in categories:
        idx = examples.get(cat_key)
        if idx is None:
            # Hide empty columns
            for row in range(3):
                axes[row, col].axis('off')
            continue
        
        # Raw image
        raw_img = dataset_raw._load_image(idx)
        axes[0, col].imshow(raw_img, cmap='gray', vmin=0, vmax=65535)
        axes[0, col].set_title(f'{cat_name}\n(Raw)', fontsize=11, fontweight='bold')
        axes[0, col].axis('off')
        
        # Processed image
        processed_img, _ = dataset_processed[idx]
        if isinstance(processed_img, torch.Tensor):
            processed_img = processed_img.squeeze().numpy()
        
        # Apply contrast adjustment for visualization
        if processed_img.max() <= 1:
            p2, p98 = np.percentile(processed_img, (2, 98))
            processed_display = np.clip((processed_img - p2) / (p98 - p2), 0, 1)
        else:
            processed_display = processed_img
        
        axes[1, col].imshow(processed_display, cmap='gray')
        axes[1, col].set_title('Processed', fontsize=11)
        axes[1, col].axis('off')
        
        # Intensity histogram comparison
        ax_hist = axes[2, col]
        
        # Raw histogram
        hist_raw, bins_raw = np.histogram(raw_img.flatten(), bins=50, range=(0, 65535))
        # Processed histogram (scale to match)
        hist_proc, bins_proc = np.histogram(processed_img.flatten(), bins=50, range=(0, 1))
        
        # Normalize for comparison
        hist_raw = hist_raw / hist_raw.max()
        hist_proc = hist_proc / hist_proc.max()
        
        ax_hist.plot(bins_raw[:-1] / 65535, hist_raw, 'b-', alpha=0.7, label='Raw', linewidth=2)
        ax_hist.plot(bins_proc[:-1], hist_proc, 'r-', alpha=0.7, label='Processed', linewidth=2)
        ax_hist.set_xlabel('Intensity', fontsize=9)
        ax_hist.set_ylabel('Frequency', fontsize=9)
        ax_hist.set_title('Distribution', fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        
        # Add statistics
        raw_mean = np.mean(raw_img)
        raw_std = np.std(raw_img)
        proc_mean = np.mean(processed_img)
        proc_std = np.std(processed_img)
        
        stats_text = f'Raw: μ={raw_mean:.0f}, σ={raw_std:.0f}\n'
        stats_text += f'Proc: μ={proc_mean:.3f}, σ={proc_std:.3f}'
        ax_hist.text(0.02, 0.98, stats_text, transform=ax_hist.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Quality-Aware Preprocessing: Handling Different Image Issues', 
                 fontsize=14, fontweight='bold')
    
    # Add explanation
    fig.text(0.5, 0.01, 
             'Preprocessing pipeline: Issue detection → CLAHE enhancement → Adaptive denoising → Percentile normalization',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved preprocessing comparison to {save_path}[/green]")
    
    plt.close()


if __name__ == "__main__":
    main()