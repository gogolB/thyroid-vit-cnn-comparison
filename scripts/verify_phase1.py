#!/usr/bin/env python3
"""
Phase 1 Verification Script
Checks that all Phase 1 components are properly set up
"""

import os
import sys
from pathlib import Path
import subprocess
import json

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()


def check_git_setup():
    """Check if git repository is initialized."""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            # Check for remote
            remote_result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
            has_remote = bool(remote_result.stdout.strip())
            
            # Check for commits
            commit_result = subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True)
            has_commits = bool(commit_result.stdout.strip())
            
            return True, {
                'initialized': True,
                'has_remote': has_remote,
                'has_commits': has_commits
            }
        else:
            return False, {'initialized': False}
    except FileNotFoundError:
        return False, {'error': 'Git not installed'}


def check_wandb_setup():
    """Check if Weights & Biases is configured."""
    try:
        import wandb
        
        # Check if logged in
        api_key = wandb.api.api_key
        is_configured = api_key is not None
        
        return is_configured, {
            'installed': True,
            'configured': is_configured
        }
    except ImportError:
        return False, {'installed': False}
    except Exception as e:
        return False, {'error': str(e)}


def check_data_structure():
    """Check if data is properly organized."""
    data_dir = Path('data/raw')
    
    checks = {
        'data_dir_exists': data_dir.exists(),
        'normal_dir_exists': (data_dir / 'normal').exists(),
        'cancerous_dir_exists': (data_dir / 'cancerous').exists(),
        'normal_images': 0,
        'cancerous_images': 0,
        'splits_exist': Path('data/splits/split_info.json').exists()
    }
    
    if checks['normal_dir_exists']:
        checks['normal_images'] = len(list((data_dir / 'normal').glob('*.tif')))
    
    if checks['cancerous_dir_exists']:
        checks['cancerous_images'] = len(list((data_dir / 'cancerous').glob('*.tif')))
    
    all_good = all([
        checks['data_dir_exists'],
        checks['normal_dir_exists'],
        checks['cancerous_dir_exists'],
        checks['normal_images'] > 0,
        checks['cancerous_images'] > 0,
        checks['splits_exist']
    ])
    
    return all_good, checks


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'pytorch_lightning',
        'hydra',
        'wandb',
        'monai',
        'rich',
        'tifffile',
        'albumentations'
    ]
    
    missing = []
    versions = {}
    
    for package in required_packages:
        try:
            module = __import__(package.replace('-', '_'))
            versions[package] = getattr(module, '__version__', 'installed')
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, {
        'missing': missing,
        'versions': versions
    }


def check_code_components():
    """Check if all code components are in place."""
    components = {
        'dataset': Path('src/data/dataset.py'),
        'transforms': Path('src/data/transforms.py'),
        'visualize': Path('src/data/visualize.py'),
        'console_utils': Path('src/utils/console.py'),
        'prepare_script': Path('scripts/prepare_data.py'),
        'configs': Path('configs/config.yaml'),
        'dataset_config': Path('configs/dataset/cars.yaml'),
        'training_config': Path('configs/training/standard.yaml'),
    }
    
    status = {}
    for name, path in components.items():
        status[name] = path.exists()
    
    return all(status.values()), status


def check_visualization_outputs():
    """Check if visualizations have been generated."""
    viz_dir = Path('visualization_outputs')
    
    expected_files = [
        'dataset_statistics.png',
        'samples_train.png',
        'samples_val.png',
        'samples_test.png',
        'augmentation_levels.png',
        'class_distribution_train.png'
    ]
    
    found_files = []
    if viz_dir.exists():
        found_files = [f.name for f in viz_dir.glob('*.png')]
    
    return viz_dir.exists() and len(found_files) > 0, {
        'directory_exists': viz_dir.exists(),
        'files_found': len(found_files),
        'expected_files': expected_files,
        'actual_files': found_files
    }


def test_data_loading():
    """Test if data loads correctly."""
    try:
        from src.data.dataset import CARSThyroidDataset
        
        dataset = CARSThyroidDataset(
            root_dir='data/raw',
            split='train',
            target_size=256,
            normalize=True,
            patient_level_split=False
        )
        
        if len(dataset) > 0:
            img, label = dataset[0]
            return True, {
                'dataset_size': len(dataset),
                'image_shape': str(img.shape),
                'image_range': f"[{img.min():.3f}, {img.max():.3f}]"
            }
        else:
            return False, {'error': 'Empty dataset'}
            
    except Exception as e:
        return False, {'error': str(e)}


def main():
    """Run all checks and display results."""
    
    console.print(Panel.fit(
        "[bold cyan]Phase 1 Completion Checklist[/bold cyan]\n"
        "[dim]Verifying all components are ready[/dim]",
        border_style="blue"
    ))
    
    # Run all checks
    checks = {
        'Git Repository': check_git_setup(),
        'Weights & Biases': check_wandb_setup(),
        'Data Structure': check_data_structure(),
        'Dependencies': check_dependencies(),
        'Code Components': check_code_components(),
        'Visualizations': check_visualization_outputs(),
        'Data Loading': test_data_loading()
    }
    
    # Display results
    table = Table(title="Phase 1 Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")
    
    all_passed = True
    
    for component, (passed, details) in checks.items():
        status = "[green]✓ Pass[/green]" if passed else "[red]✗ Fail[/red]"
        
        # Format details
        if component == 'Git Repository':
            if passed:
                detail_str = f"Commits: {'Yes' if details.get('has_commits') else 'No'}, Remote: {'Yes' if details.get('has_remote') else 'No'}"
            else:
                detail_str = "Not initialized"
                
        elif component == 'Weights & Biases':
            if details.get('installed'):
                detail_str = "Configured" if details.get('configured') else "Not configured"
            else:
                detail_str = "Not installed"
                
        elif component == 'Data Structure':
            detail_str = f"Normal: {details.get('normal_images', 0)}, Cancer: {details.get('cancerous_images', 0)}"
            
        elif component == 'Dependencies':
            if details.get('missing'):
                detail_str = f"Missing: {', '.join(details['missing'])}"
            else:
                detail_str = f"{len(details.get('versions', {}))} packages installed"
                
        elif component == 'Visualizations':
            detail_str = f"{details.get('files_found', 0)} files generated"
            
        elif component == 'Data Loading':
            if passed:
                detail_str = f"Size: {details.get('dataset_size', 0)}, Shape: {details.get('image_shape', 'N/A')}"
            else:
                detail_str = details.get('error', 'Failed')[:50]
                
        else:
            detail_str = str(details)[:50]
        
        table.add_row(component, status, detail_str)
        
        if not passed:
            all_passed = False
    
    console.print("\n")
    console.print(table)
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    if not checks['Git Repository'][0]:
        console.print("1. [yellow]Initialize git repository:[/yellow]")
        console.print("   git init && git add . && git commit -m 'Initial commit'")
    
    if not checks['Weights & Biases'][0] or not checks['Weights & Biases'][1].get('configured'):
        console.print("2. [yellow]Configure Weights & Biases:[/yellow]")
        console.print("   wandb login")
    
    if not checks['Visualizations'][0] or checks['Visualizations'][1].get('files_found', 0) == 0:
        console.print("3. [yellow]Generate visualizations:[/yellow]")
        console.print("   python scripts/prepare_data.py --visualize")
    
    if checks['Dependencies'][1].get('missing'):
        console.print("4. [yellow]Install missing packages:[/yellow]")
        console.print("   pip install -r requirements.txt")
    
    # Overall status
    if all_passed:
        console.print("\n[bold green]✅ Phase 1 Complete! Ready for Phase 2 (CNN Implementation)[/bold green]")
    else:
        console.print("\n[bold yellow]⚠️  Some components need attention before proceeding to Phase 2[/bold yellow]")
    
    # Additional checks to consider
    console.print("\n[bold]Additional Considerations:[/bold]")
    console.print("• Test on Linux workstation with GPU")
    console.print("• Consider implementing patch extraction (currently using resize)")
    console.print("• Test caching functionality for faster data loading")
    console.print("• Generate comprehensive data quality report")


if __name__ == "__main__":
    main()
