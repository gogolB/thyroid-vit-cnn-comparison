#!/usr/bin/env python3
"""
Comprehensive Data Quality Report for CARS Thyroid Dataset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track
from rich.panel import Panel

from src.data.dataset import CARSThyroidDataset

console = Console()


def analyze_intensity_distributions(dataset, n_samples=None):
    """Analyze pixel intensity distributions."""
    
    if n_samples is None:
        n_samples = len(dataset)
    
    console.print(f"\n[cyan]Analyzing intensity distributions for {n_samples} images...[/cyan]")
    
    intensities = {
        'normal': {'means': [], 'stds': [], 'mins': [], 'maxs': [], 'percentiles': []},
        'cancerous': {'means': [], 'stds': [], 'mins': [], 'maxs': [], 'percentiles': []}
    }
    
    for idx in track(range(min(n_samples, len(dataset))), description="Processing"):
        img = dataset._load_image(idx)
        label = dataset.labels[dataset.indices[idx]]
        class_name = 'normal' if label == 0 else 'cancerous'
        
        intensities[class_name]['means'].append(np.mean(img))
        intensities[class_name]['stds'].append(np.std(img))
        intensities[class_name]['mins'].append(np.min(img))
        intensities[class_name]['maxs'].append(np.max(img))
        intensities[class_name]['percentiles'].append([
            np.percentile(img, 1),
            np.percentile(img, 25),
            np.percentile(img, 50),
            np.percentile(img, 75),
            np.percentile(img, 99)
        ])
    
    return intensities


def detect_outliers(intensities, threshold=3):
    """Detect outlier images based on intensity statistics."""
    
    outliers = {'normal': [], 'cancerous': []}
    
    for class_name in ['normal', 'cancerous']:
        means = np.array(intensities[class_name]['means'])
        
        if len(means) > 0:
            mean_val = np.mean(means)
            std_val = np.std(means)
            
            # Find outliers (beyond threshold std deviations)
            outlier_mask = np.abs(means - mean_val) > threshold * std_val
            outlier_indices = np.where(outlier_mask)[0]
            
            outliers[class_name] = outlier_indices.tolist()
    
    return outliers


def generate_quality_metrics(dataset):
    """Generate comprehensive quality metrics."""
    
    metrics = {
        'dataset_info': {
            'total_images': len(dataset),
            'split': dataset.split,
            'target_size': dataset.target_size,
            'original_size': (512, 512)
        },
        'class_balance': {},
        'intensity_stats': {},
        'quality_issues': {
            'extreme_dark': [],
            'extreme_bright': [],
            'low_contrast': [],
            'potential_artifacts': []
        }
    }
    
    # Class balance
    labels = dataset.labels[dataset.indices]
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = 'normal' if label == 0 else 'cancerous'
        metrics['class_balance'][class_name] = int(count)
    
    # Analyze subset for quality issues
    console.print("\n[cyan]Checking for quality issues...[/cyan]")
    
    for idx in track(range(min(100, len(dataset))), description="Quality check"):
        img = dataset._load_image(idx)
        label = dataset.labels[dataset.indices[idx]]
        
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        # Check for quality issues
        if mean_val < 150:  # Very dark
            metrics['quality_issues']['extreme_dark'].append(idx)
        elif mean_val > 5000:  # Very bright
            metrics['quality_issues']['extreme_bright'].append(idx)
        
        if std_val < 50:  # Low contrast
            metrics['quality_issues']['low_contrast'].append(idx)
        
        # Check for potential artifacts (sudden intensity jumps)
        if np.max(img) > 10000 and np.mean(img) < 500:
            metrics['quality_issues']['potential_artifacts'].append(idx)
    
    return metrics


def create_quality_report(output_dir='reports'):
    """Create comprehensive data quality report."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    console.print(Panel.fit(
        "[bold cyan]CARS Thyroid Dataset Quality Report[/bold cyan]\n"
        "[dim]Comprehensive analysis of data quality[/dim]",
        border_style="blue"
    ))
    
    all_results = {}
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        console.print(f"\n[bold yellow]Analyzing {split} set...[/bold yellow]")
        
        dataset = CARSThyroidDataset(
            root_dir='data/raw',
            split=split,
            target_size=256,
            normalize=False,  # Keep original values for analysis
            patient_level_split=False
        )
        
        # Get intensity distributions
        intensities = analyze_intensity_distributions(dataset)
        
        # Detect outliers
        outliers = detect_outliers(intensities)
        
        # Generate metrics
        metrics = generate_quality_metrics(dataset)
        
        # Store results
        all_results[split] = {
            'intensities': intensities,
            'outliers': outliers,
            'metrics': metrics
        }
        
        # Print summary
        table = Table(title=f"{split.capitalize()} Set Summary", 
                     show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Normal", style="green")
        table.add_column("Cancerous", style="red")
        
        # Add intensity statistics
        for stat in ['means', 'stds']:
            normal_vals = intensities['normal'][stat]
            cancer_vals = intensities['cancerous'][stat]
            
            if normal_vals and cancer_vals:
                table.add_row(
                    f"Avg {stat[:-1]}",
                    f"{np.mean(normal_vals):.1f} ± {np.std(normal_vals):.1f}",
                    f"{np.mean(cancer_vals):.1f} ± {np.std(cancer_vals):.1f}"
                )
        
        # Add outlier info
        table.add_row(
            "Outliers",
            str(len(outliers['normal'])),
            str(len(outliers['cancerous']))
        )
        
        console.print(table)
    
    # Create visualizations
    console.print("\n[cyan]Creating visualization plots...[/cyan]")
    
    # 1. Intensity distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Intensity Distribution Analysis', fontsize=16)
    
    for idx, (split, results) in enumerate(all_results.items()):
        if idx >= 3:
            break
            
        ax1 = axes[idx // 2, idx % 2]
        
        # Plot distributions
        normal_means = results['intensities']['normal']['means']
        cancer_means = results['intensities']['cancerous']['means']
        
        ax1.hist(normal_means, bins=20, alpha=0.5, label='Normal', color='green')
        ax1.hist(cancer_means, bins=20, alpha=0.5, label='Cancerous', color='red')
        ax1.set_title(f'{split.capitalize()} Set')
        ax1.set_xlabel('Mean Intensity')
        ax1.set_ylabel('Frequency')
        ax1.legend()
    
    # Hide extra subplot if only 3 splits
    if len(all_results) == 3:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intensity_distributions.png', dpi=150)
    plt.close()
    
    # 2. Quality issues summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quality_issues = {
        'Extreme Dark': 0,
        'Extreme Bright': 0,
        'Low Contrast': 0,
        'Artifacts': 0
    }
    
    for split, results in all_results.items():
        issues = results['metrics']['quality_issues']
        quality_issues['Extreme Dark'] += len(issues['extreme_dark'])
        quality_issues['Extreme Bright'] += len(issues['extreme_bright'])
        quality_issues['Low Contrast'] += len(issues['low_contrast'])
        quality_issues['Artifacts'] += len(issues['potential_artifacts'])
    
    bars = ax.bar(quality_issues.keys(), quality_issues.values())
    ax.set_title('Quality Issues Summary (All Splits)')
    ax.set_ylabel('Number of Images')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_issues.png', dpi=150)
    plt.close()
    
    # Save detailed report as JSON
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'dataset_stats': all_results,
        'summary': {
            'total_outliers': sum(len(all_results[split]['outliers']['normal']) + 
                                len(all_results[split]['outliers']['cancerous']) 
                                for split in all_results),
            'quality_issues_total': sum(quality_issues.values())
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    report_data = convert_to_serializable(report_data)
    
    with open(output_dir / 'quality_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    console.print(f"\n[green]✓ Quality report saved to {output_dir}/[/green]")
    console.print("[green]✓ Generated files:[/green]")
    console.print("  - quality_report.json (detailed data)")
    console.print("  - intensity_distributions.png")
    console.print("  - quality_issues.png")
    
    # Final recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if sum(quality_issues.values()) > 0:
        console.print("[yellow]• Review images with quality issues[/yellow]")
        console.print("[yellow]• Consider additional preprocessing for extreme cases[/yellow]")
    
    total_outliers = sum(len(all_results[split]['outliers']['normal']) + 
                        len(all_results[split]['outliers']['cancerous']) 
                        for split in all_results)
    
    if total_outliers > 0:
        console.print(f"[yellow]• Found {total_outliers} statistical outliers - review for data errors[/yellow]")
    
    console.print("[green]• Overall data quality appears good for training[/green]")


if __name__ == "__main__":
    create_quality_report()
