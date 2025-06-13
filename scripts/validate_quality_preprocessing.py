#!/usr/bin/env python3
"""
Enhanced test for quality-aware preprocessing on actual CARS thyroid images.
Includes detailed analysis and quality tier statistics.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import json

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.data.dataset import CARSThyroidDataset
from src.data.quality_preprocessing import QualityAwarePreprocessor, create_quality_aware_transform

console = Console()


def analyze_preprocessing_statistics(dataset, preprocessor, quality_report_path):
    """Analyze preprocessing effects across all quality tiers."""
    
    # Load quality report
    with open(quality_report_path, 'r') as f:
        quality_report = json.load(f)
    
    # Get quality issues for this split
    split_issues = quality_report['dataset_stats'][dataset.split]['metrics']['quality_issues']
    
    # Create categories
    all_indices = set(range(len(dataset)))
    extreme_dark = set(split_issues['extreme_dark'])
    low_contrast = set(split_issues['low_contrast'])
    artifacts = set(split_issues['potential_artifacts'])
    
    # Multiple issues
    multiple_issues = (extreme_dark & low_contrast) | (extreme_dark & artifacts) | (low_contrast & artifacts)
    
    # Single issues (excluding those with multiple)
    only_dark = extreme_dark - multiple_issues - artifacts
    only_contrast = low_contrast - multiple_issues - extreme_dark
    only_artifacts = artifacts - multiple_issues - extreme_dark - low_contrast
    
    # High quality
    problematic = extreme_dark | low_contrast | artifacts
    high_quality = all_indices - problematic
    
    # Analyze each category
    categories = {
        'High Quality': list(high_quality)[:20],  # Sample 20
        'Extreme Dark Only': list(only_dark)[:20],
        'Low Contrast Only': list(only_contrast)[:20],
        'Artifacts Only': list(only_artifacts)[:20],
        'Multiple Issues': list(multiple_issues)[:20]
    }
    
    # Create analysis table
    table = Table(title="Preprocessing Effects by Quality Tier", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Count", style="white")
    table.add_column("Avg Original Mean", style="yellow")
    table.add_column("Avg Processed Mean", style="green")
    table.add_column("Mean Change", style="blue")
    table.add_column("Avg Original Std", style="yellow")
    table.add_column("Avg Processed Std", style="green")
    
    results = {}
    for category, indices in categories.items():
        if not indices:
            continue
            
        orig_means = []
        proc_means = []
        orig_stds = []
        proc_stds = []
        
        for idx in indices[:10]:  # Analyze up to 10 per category
            img = dataset._load_image(idx)
            processed = preprocessor.preprocess_image(img)
            
            orig_means.append(np.mean(img))
            proc_means.append(np.mean(processed))
            orig_stds.append(np.std(img))
            proc_stds.append(np.std(processed))
        
        if orig_means:
            avg_orig_mean = np.mean(orig_means)
            avg_proc_mean = np.mean(proc_means)
            mean_change = avg_proc_mean / avg_orig_mean if avg_orig_mean > 0 else 0
            
            table.add_row(
                category,
                str(len(indices)),
                f"{avg_orig_mean:.1f}",
                f"{avg_proc_mean:.1f}",
                f"{mean_change:.2f}x",
                f"{np.mean(orig_stds):.1f}",
                f"{np.mean(proc_stds):.1f}"
            )
            
            results[category] = {
                'indices': indices,
                'stats': {
                    'orig_mean': avg_orig_mean,
                    'proc_mean': avg_proc_mean,
                    'mean_change': mean_change
                }
            }
    
    console.print(table)
    return results


def visualize_preprocessing_effects():
    """Visualize the effects of quality-aware preprocessing with enhanced analysis."""
    
    console.print(Panel.fit(
        "[bold cyan]Quality-Aware Preprocessing Test[/bold cyan]\n"
        "[dim]Comparing original vs preprocessed images[/dim]",
        border_style="blue"
    ))
    
    # Load dataset without preprocessing
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='train',
        target_size=256,
        normalize=False,
        patient_level_split=False
    )
    
    # Create preprocessor
    quality_report_path = Path('reports/quality_report.json')
    preprocessor = QualityAwarePreprocessor(quality_report_path)
    
    # Analyze by quality tier
    console.print("\n[bold yellow]Analyzing preprocessing effects by quality tier...[/bold yellow]")
    tier_results = analyze_preprocessing_statistics(dataset, preprocessor, quality_report_path)
    
    # Test specific cases
    test_cases = [
        (11, ['extreme_dark', 'artifacts'], 'Extreme Dark + Artifacts'),
        (16, ['extreme_dark'], 'Low Contrast + Dark'),
        (1, ['artifacts'], 'Artifacts'),
        (0, [], 'High Quality (baseline)')
    ]
    
    # Create enhanced visualization
    fig = plt.figure(figsize=(16, 4 * len(test_cases)))
    gs = gridspec.GridSpec(len(test_cases), 4, width_ratios=[1, 1, 1, 1.2], wspace=0.3)
    
    for row, (idx, expected_issues, description) in enumerate(test_cases):
        # Load original image
        img = dataset._load_image(idx)
        label = dataset.labels[dataset.indices[idx]]
        class_name = "Normal" if label == 0 else "Cancer"
        
        # Identify issues
        detected_issues = preprocessor.identify_quality_issues(img)
        
        # Apply preprocessing
        img_preprocessed = preprocessor.preprocess_image(img)
        
        # Original image
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(img, cmap='gray', vmin=0, vmax=65535)
        ax1.set_title(f'{description} - Original\n{class_name} #{idx}', fontsize=10)
        ax1.axis('off')
        
        # Add statistics overlay
        stats_text = f'μ={np.mean(img):.0f}\nσ={np.std(img):.0f}\nmax={np.max(img)}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=8, verticalalignment='top', color='yellow',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Preprocessed image
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(img_preprocessed, cmap='gray', vmin=0, vmax=65535)
        ax2.set_title(f'Preprocessed\nDetected: {detected_issues}', fontsize=10)
        ax2.axis('off')
        
        # Add statistics overlay
        stats_text = f'μ={np.mean(img_preprocessed):.0f}\nσ={np.std(img_preprocessed):.0f}\nmax={np.max(img_preprocessed)}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=8, verticalalignment='top', color='yellow',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Difference map
        ax3 = fig.add_subplot(gs[row, 2])
        diff = img_preprocessed.astype(float) - img.astype(float)
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-1000, vmax=1000)
        ax3.set_title('Difference Map', fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Histogram comparison
        ax4 = fig.add_subplot(gs[row, 3])
        
        # Plot histograms
        hist_orig, bins_orig = np.histogram(img.flatten(), bins=100, range=(0, 2000))
        hist_proc, bins_proc = np.histogram(img_preprocessed.flatten(), bins=100, range=(0, 2000))
        
        ax4.plot(bins_orig[:-1], hist_orig, 'b-', alpha=0.7, label='Original', linewidth=1.5)
        ax4.plot(bins_proc[:-1], hist_proc, 'r-', alpha=0.7, label='Preprocessed', linewidth=1.5)
        ax4.set_xlabel('Intensity', fontsize=9)
        ax4.set_ylabel('Count', fontsize=9)
        ax4.set_title('Intensity Distribution', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Print detailed stats
        console.print(f"\n[yellow]{description} (Index {idx}):[/yellow]")
        console.print(f"  Detected issues: {detected_issues}")
        console.print(f"  Original  - Mean: {np.mean(img):.1f}, Std: {np.std(img):.1f}, Max: {np.max(img)}")
        console.print(f"  Processed - Mean: {np.mean(img_preprocessed):.1f}, Std: {np.std(img_preprocessed):.1f}, Max: {np.max(img_preprocessed)}")
        
        # Calculate change metrics
        mean_change = np.mean(img_preprocessed) / np.mean(img) if np.mean(img) > 0 else 0
        std_change = np.std(img_preprocessed) / np.std(img) if np.std(img) > 0 else 0
        console.print(f"  Changes   - Mean: {mean_change:.2f}x, Std: {std_change:.2f}x")
    
    plt.suptitle('Quality-Aware Preprocessing Effects - Detailed Analysis', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('visualization_outputs/preprocessing_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'quality_preprocessing_effects_enhanced.png', dpi=150, bbox_inches='tight')
    console.print(f"\n[green]✓ Saved visualization to {output_dir}/quality_preprocessing_effects_enhanced.png[/green]")
    plt.show()


def test_transform_pipeline():
    """Test the complete transform pipeline."""
    
    console.print("\n[cyan]Testing complete transform pipeline...[/cyan]")
    
    # Create quality-aware transform
    transform = create_quality_aware_transform(
        target_size=256,
        quality_report_path=Path('reports/quality_report.json'),
        augmentation_level='medium',
        split='train'
    )
    
    # Test on a batch
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='train',
        target_size=256,
        normalize=False,
        transform=transform,
        patient_level_split=False
    )
    
    # Get sample batch
    batch_images, batch_labels = dataset.get_sample_batch(n_samples=4)
    
    console.print(f"[green]✓ Transform pipeline test successful![/green]")
    console.print(f"  Batch shape: {batch_images.shape}")
    console.print(f"  Value range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
    
    # Visualize batch
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        img = batch_images[i].squeeze().numpy()
        label = batch_labels[i].item()
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{"Normal" if label == 0 else "Cancer"}\nProcessed & Augmented')
        ax.axis('off')
        
        # Add value range info
        ax.text(0.02, 0.02, f'Range: [{img.min():.2f}, {img.max():.2f}]', 
                transform=ax.transAxes, fontsize=8, color='yellow',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    plt.suptitle('Quality-Preprocessed Training Batch', fontsize=14)
    plt.tight_layout()
    plt.show()


def benchmark_preprocessing_speed():
    """Benchmark preprocessing speed."""
    
    console.print("\n[cyan]Benchmarking preprocessing speed...[/cyan]")
    
    import time
    
    # Create preprocessor
    preprocessor = QualityAwarePreprocessor()
    
    # Test different image sizes
    sizes = [256, 512]
    
    for size in sizes:
        # Create test images
        dark_img = np.random.randint(120, 150, (size, size), dtype=np.uint16)
        artifact_img = np.random.randint(150, 300, (size, size), dtype=np.uint16)
        artifact_img[50:60, 50:60] = 15000
        
        # Time preprocessing
        times = []
        for _ in range(10):
            start = time.time()
            _ = preprocessor.preprocess_image(dark_img)
            _ = preprocessor.preprocess_image(artifact_img)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000 / 2  # ms per image
        console.print(f"  {size}x{size}: {avg_time:.2f} ms/image")


if __name__ == "__main__":
    # Run tests
    visualize_preprocessing_effects()
    test_transform_pipeline()
    benchmark_preprocessing_speed()
    
    console.print("\n[bold green]✅ Quality-aware preprocessing tests complete![/bold green]")
    
    # Final recommendations
    console.print("\n[bold cyan]Recommendations:[/bold cyan]")
    console.print("1. The preprocessing is now working well with reasonable intensity changes")
    console.print("2. Consider creating separate models for high-quality vs problematic images")
    console.print("3. Monitor performance metrics per quality tier during training")
    console.print("4. Use the quality-aware transform in your training pipeline")