#!/usr/bin/env python3
"""
Visualize and analyze images with quality issues identified in the data quality report.
This script helps understand artifacts, low contrast, and extreme dark images.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from src.data.dataset import CARSThyroidDataset
import torch

console = Console()


def load_quality_report(report_path: Path = Path("reports/quality_report.json")):
    """Load the quality report JSON."""
    with open(report_path, 'r') as f:
        return json.load(f)


def visualize_problematic_images(dataset, indices, issue_type, n_samples=16):
    """Visualize a grid of problematic images."""
    
    n_samples = min(n_samples, len(indices))
    if n_samples == 0:
        console.print(f"[yellow]No {issue_type} images found in this split[/yellow]")
        return
    
    # Create grid
    cols = min(4, n_samples)
    rows = (n_samples + cols - 1) // cols
    
    fig = plt.figure(figsize=(cols * 3, rows * 3.5))
    fig.suptitle(f'{issue_type} Images (n={len(indices)})', fontsize=16)
    
    # Create gridspec for better subplot control
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx in range(n_samples):
        ax = fig.add_subplot(gs[idx])
        
        # Load image
        img_idx = indices[idx]
        img = dataset._load_image(img_idx)
        label = dataset.labels[dataset.indices[img_idx]]
        
        # Calculate statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Display image
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=65535)
        
        # Add colorbar for first image
        if idx == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        # Title with stats
        class_name = "Normal" if label == 0 else "Cancer"
        ax.set_title(f'{class_name} #{img_idx}', fontsize=10, fontweight='bold')
        
        # Add statistics as text
        stats_text = f'μ={mean_val:.0f}, σ={std_val:.0f}\n[{min_val}, {max_val}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_samples, rows * cols):
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def analyze_intensity_patterns(dataset, indices, issue_type):
    """Analyze intensity patterns of problematic images."""
    
    if len(indices) == 0:
        return None
    
    # Collect intensity histograms
    all_intensities = []
    labels = []
    
    for idx in track(indices[:50], description=f"Analyzing {issue_type}"):  # Limit to 50 for speed
        img = dataset._load_image(idx)
        all_intensities.append(img.flatten())
        labels.append(dataset.labels[dataset.indices[idx]])
    
    # Create analysis figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{issue_type} - Intensity Analysis', fontsize=16)
    
    # 1. Combined histogram
    ax = axes[0, 0]
    for intensities, label in zip(all_intensities, labels):
        color = 'green' if label == 0 else 'red'
        alpha = 0.1
        ax.hist(intensities, bins=50, alpha=alpha, color=color, density=True)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Intensity Distributions')
    ax.set_xlim(0, 1000)  # Focus on lower range
    
    # 2. Mean vs Std scatter
    ax = axes[0, 1]
    means = [np.mean(img) for img in all_intensities]
    stds = [np.std(img) for img in all_intensities]
    colors = ['green' if l == 0 else 'red' for l in labels]
    ax.scatter(means, stds, c=colors, alpha=0.6)
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Std Deviation')
    ax.set_title('Mean vs Std Distribution')
    
    # 3. Intensity range plot
    ax = axes[1, 0]
    ranges = [(np.min(img), np.max(img)) for img in all_intensities]
    for i, (min_val, max_val) in enumerate(ranges):
        color = 'green' if labels[i] == 0 else 'red'
        ax.plot([i, i], [min_val, max_val], color=color, alpha=0.5, linewidth=2)
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Intensity Range')
    ax.set_title('Min-Max Ranges')
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    normal_means = [m for m, l in zip(means, labels) if l == 0]
    cancer_means = [m for m, l in zip(means, labels) if l == 1]
    
    stats_text = f"""Summary Statistics:
    
Normal Images (n={len(normal_means)}):
  Mean: {np.mean(normal_means):.1f} ± {np.std(normal_means):.1f}
  Range: [{np.min(normal_means):.1f}, {np.max(normal_means):.1f}]

Cancerous Images (n={len(cancer_means)}):
  Mean: {np.mean(cancer_means):.1f} ± {np.std(cancer_means):.1f}
  Range: [{np.min(cancer_means):.1f}, {np.max(cancer_means):.1f}]
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def create_quality_aware_splits(quality_report):
    """Create indices for different quality tiers."""
    
    quality_tiers = {
        'high_quality': [],
        'extreme_dark': [],
        'low_contrast': [],
        'artifacts': [],
        'multiple_issues': []
    }
    
    for split in ['train', 'val', 'test']:
        issues = quality_report['dataset_stats'][split]['metrics']['quality_issues']
        
        # Get all indices
        all_indices = set(range(len(quality_report['dataset_stats'][split]['intensities']['normal']['means']) + 
                               len(quality_report['dataset_stats'][split]['intensities']['cancerous']['means'])))
        
        # Categorize
        extreme_dark = set(issues['extreme_dark'])
        low_contrast = set(issues['low_contrast'])
        artifacts = set(issues['potential_artifacts'])
        
        # Multiple issues
        multiple = (extreme_dark & low_contrast) | (extreme_dark & artifacts) | (low_contrast & artifacts)
        
        # High quality = no issues
        problematic = extreme_dark | low_contrast | artifacts
        high_quality = all_indices - problematic
        
        quality_tiers['high_quality'].extend([(split, idx) for idx in high_quality])
        quality_tiers['extreme_dark'].extend([(split, idx) for idx in extreme_dark - multiple])
        quality_tiers['low_contrast'].extend([(split, idx) for idx in low_contrast - multiple])
        quality_tiers['artifacts'].extend([(split, idx) for idx in artifacts - multiple])
        quality_tiers['multiple_issues'].extend([(split, idx) for idx in multiple])
    
    return quality_tiers


def main():
    """Main visualization function."""
    
    console.print(Panel.fit(
        "[bold cyan]Quality Issues Visualization[/bold cyan]\n"
        "[dim]Analyzing problematic images from quality report[/dim]",
        border_style="blue"
    ))
    
    # Load quality report
    quality_report = load_quality_report()
    
    # Create output directory
    output_dir = Path("visualization_outputs/quality_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        console.print(f"\n[bold yellow]Analyzing {split} set...[/bold yellow]")
        
        # Load dataset
        dataset = CARSThyroidDataset(
            root_dir='data/raw',
            split=split,
            target_size=256,
            normalize=False,  # Keep original values
            patient_level_split=False
        )
        
        # Get quality issues
        issues = quality_report['dataset_stats'][split]['metrics']['quality_issues']
        
        # Visualize each issue type
        for issue_type, indices in issues.items():
            if issue_type == 'extreme_bright' and len(indices) == 0:
                continue
                
            console.print(f"\n[cyan]{issue_type}: {len(indices)} images[/cyan]")
            
            # Visualize sample images
            fig = visualize_problematic_images(dataset, indices, issue_type)
            if fig:
                fig.savefig(output_dir / f'{split}_{issue_type}_samples.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Analyze patterns
            if len(indices) > 5:  # Only analyze if we have enough samples
                fig = analyze_intensity_patterns(dataset, indices, issue_type)
                if fig:
                    fig.savefig(output_dir / f'{split}_{issue_type}_analysis.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
    
    # Create quality tier summary
    console.print("\n[bold cyan]Creating quality tier summary...[/bold cyan]")
    quality_tiers = create_quality_aware_splits(quality_report)
    
    # Summary table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary data
    summary_data = []
    for tier, items in quality_tiers.items():
        train_count = sum(1 for split, _ in items if split == 'train')
        val_count = sum(1 for split, _ in items if split == 'val')
        test_count = sum(1 for split, _ in items if split == 'test')
        total = len(items)
        summary_data.append([tier.replace('_', ' ').title(), train_count, val_count, test_count, total])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Quality Tier', 'Train', 'Val', 'Test', 'Total'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
    
    plt.title('Quality Tier Distribution Across Splits', fontsize=16, pad=20)
    plt.savefig(output_dir / 'quality_tier_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"\n[green]✓ Analysis complete! Results saved to {output_dir}/[/green]")
    
    # Recommendations
    console.print("\n[bold]Recommendations based on analysis:[/bold]")
    console.print("1. [yellow]Extreme Dark Images[/yellow]: Consider adaptive histogram equalization")
    console.print("2. [yellow]Low Contrast Images[/yellow]: Apply CLAHE with appropriate clip limit")
    console.print("3. [yellow]Artifacts[/yellow]: May need manual review or weighted loss during training")
    console.print("4. [yellow]Multiple Issues[/yellow]: Consider excluding from initial training")


if __name__ == "__main__":
    main()
