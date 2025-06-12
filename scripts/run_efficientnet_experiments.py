#!/usr/bin/env python3
"""
Run all EfficientNet experiments for CARS Thyroid Classification
Tests B0, B1, B2, and B3 variants with optimized configurations
Version 2: With real-time output and fixed Hydra config path
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
import time
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EfficientNet experiments')
    
    parser.add_argument('--models', nargs='+', 
                       default=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3'],
                       help='EfficientNet models to test')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--wandb-mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases logging mode')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 2 epochs')
    parser.add_argument('--batch-size-b3', type=int, default=16,
                       help='Batch size for B3 model (may need to be smaller)')
    parser.add_argument('--show-output', action='store_true', default=True,
                       help='Show training output in real-time')
    
    return parser.parse_args()


def run_single_experiment(model_name: str, args, results: dict):
    """Run a single EfficientNet experiment with real-time output."""
    
    console.print(f"\n[bold cyan]Starting experiment: {model_name}[/bold cyan]")
    console.print("[dim]━" * 80 + "[/dim]")
    start_time = time.time()
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Build command (remove conflicting Hydra args)
    cmd = [
        sys.executable,
        "src/training/train_cnn.py",
        f"model=cnn/{model_name}",
        f"training=efficientnet",  # Use EfficientNet-specific training config
        f"device={args.device}",
        f"wandb.mode={args.wandb_mode}",
    ]
    
    # Add batch size override for B3
    if model_name == 'efficientnet_b3':
        cmd.append(f"training.batch_size={args.batch_size_b3}")
    
    # Quick test mode
    if args.quick_test:
        cmd.extend([
            "trainer.max_epochs=2",
            "trainer.limit_train_batches=10",
            "trainer.limit_val_batches=5",
            "training.early_stopping.patience=1"
        ])
    
    # Run experiment
    try:
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
        console.print(f"[dim]Working dir: {project_root}[/dim]\n")
        
        # Set environment variables to help with Hydra
        env = os.environ.copy()
        env['HYDRA_FULL_ERROR'] = '1'
        env['PYTHONPATH'] = str(project_root)
        
        if args.show_output:
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            test_acc = None
            val_acc = None
            last_val_acc = None
            
            # Read output line by line
            for line in process.stdout:
                # Print the line (remove extra newline)
                print(line.rstrip())
                
                # Extract metrics from output
                if "Test Accuracy:" in line:
                    match = re.search(r"Test Accuracy:\s*([0-9.]+)", line)
                    if match:
                        test_acc = float(match.group(1))
                
                # Look for validation accuracy in the progress bar or logs
                if "val_acc" in line:
                    match = re.search(r"val_acc[:\s=]+([0-9.]+)", line)
                    if match:
                        last_val_acc = float(match.group(1))
            
            # Wait for process to complete
            process.wait()
            return_code = process.returncode
            
            # Use the last validation accuracy we saw
            val_acc = last_val_acc
            
        else:
            # Run without real-time output (original method)
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                env=env
            )
            return_code = result.returncode
            
            if return_code == 0:
                # Parse output for metrics
                output_lines = result.stdout.split('\n')
                test_acc = None
                val_acc = None
                
                for line in output_lines:
                    if "Test Accuracy:" in line:
                        match = re.search(r"Test Accuracy:\s*([0-9.]+)", line)
                        if match:
                            test_acc = float(match.group(1))
        
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            results[model_name] = {
                'status': 'success',
                'test_acc': test_acc,
                'val_acc': val_acc,
                'time_seconds': elapsed_time,
                'time_formatted': f"{elapsed_time/60:.1f} min"
            }
            
            console.print(f"\n[green]✓ {model_name} completed successfully![/green]")
            if test_acc:
                console.print(f"  Test Accuracy: {test_acc:.4f}")
            
        else:
            results[model_name] = {
                'status': 'failed',
                'error': f'Process exited with code {return_code}',
                'time_seconds': elapsed_time
            }
            console.print(f"\n[red]✗ {model_name} failed with exit code {return_code}![/red]")
            
    except Exception as e:
        results[model_name] = {
            'status': 'error',
            'error': str(e),
            'time_seconds': time.time() - start_time
        }
        console.print(f"\n[red]✗ Error running {model_name}: {e}[/red]")
    
    console.print("[dim]━" * 80 + "[/dim]\n")
    return results


def display_results(results: dict):
    """Display experiment results in a nice table."""
    
    # Create results table
    table = Table(title="EfficientNet Experiment Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Test Acc", style="green")
    table.add_column("Val Acc", style="yellow")
    table.add_column("Time", style="blue")
    
    # Add baseline ResNet18 for comparison
    table.add_row(
        "ResNet18 (baseline)",
        "[green]✓[/green]",
        "0.8530",
        "0.8680",
        "38 epochs"
    )
    table.add_row("", "", "", "", "")  # Separator
    
    # Add EfficientNet results
    for model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']:
        if model_name in results:
            result = results[model_name]
            
            status = "[green]✓[/green]" if result['status'] == 'success' else "[red]✗[/red]"
            test_acc = f"{result.get('test_acc', 0):.4f}" if result.get('test_acc') else "N/A"
            val_acc = f"{result.get('val_acc', 0):.4f}" if result.get('val_acc') else "N/A"
            time_str = result.get('time_formatted', f"{result['time_seconds']/60:.1f} min")
            
            table.add_row(model_name, status, test_acc, val_acc, time_str)
    
    console.print("\n")
    console.print(table)
    
    # Save results to file
    results_file = Path("experiments/efficientnet_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {results_file}[/green]")


def update_project_log(results: dict):
    """Update the project log with EfficientNet results."""
    
    log_path = Path("project_log.md")
    
    # Read current log
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Add update to log
    update_text = f"""

## EfficientNet Testing Update ({datetime.now().strftime('%Y-%m-%d %H:%M')})

Completed testing of EfficientNet variants with quality-aware preprocessing.

### Results Summary:
"""
    
    for model_name, result in results.items():
        if result['status'] == 'success':
            update_text += f"- **{model_name}**: Test Acc = {result.get('test_acc', 'N/A')}, Time = {result['time_formatted']}\n"
        else:
            update_text += f"- **{model_name}**: Failed - {result.get('error', 'Unknown error')[:100]}...\n"
    
    # Find where to insert (before "## Next Immediate Tasks")
    if "## Next Immediate Tasks" in content:
        insert_pos = content.find("## Next Immediate Tasks")
        content = content[:insert_pos] + update_text + "\n" + content[insert_pos:]
    else:
        content += update_text
    
    # Write updated log
    with open(log_path, 'w') as f:
        f.write(content)
    
    console.print("[green]✓ Project log updated[/green]")


def main():
    """Main experiment runner."""
    args = parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]EfficientNet Experiment Runner[/bold cyan]\n"
        f"[dim]Testing models: {', '.join(args.models)}[/dim]",
        border_style="blue"
    ))
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    
    if not config_dir.exists():
        console.print(f"[red]Error: Config directory not found at {config_dir}[/red]")
        console.print("[red]Please run this script from the project root directory[/red]")
        sys.exit(1)
    
    console.print(f"[green]✓ Config directory found: {config_dir}[/green]")
    
    # Confirm settings
    console.print("\n[bold]Experiment Settings:[/bold]")
    console.print(f"  Device: {args.device}")
    console.print(f"  W&B Mode: {args.wandb_mode}")
    console.print(f"  Quick Test: {args.quick_test}")
    console.print(f"  B3 Batch Size: {args.batch_size_b3}")
    console.print(f"  Show Output: {args.show_output}")
    
    # Important note about training time
    if not args.quick_test:
        console.print("\n[yellow]⚠ Note: Each model may take 30-60+ minutes to train with 150 epochs![/yellow]")
        console.print("[yellow]  Use --quick-test for a faster test run (2 epochs only)[/yellow]")
    
    # Run experiments
    results = {}
    
    for model_name in args.models:
        run_single_experiment(model_name, args, results)
    
    # Display results
    display_results(results)
    
    # Update project log
    if not args.quick_test:
        update_project_log(results)
    
    console.print("\n[bold green]✓ All experiments complete![/bold green]")
    
    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    
    # Find best model
    best_model = None
    best_acc = 0
    for model_name, result in results.items():
        if result['status'] == 'success' and result.get('test_acc', 0) > best_acc:
            best_acc = result['test_acc']
            best_model = model_name
    
    if best_model:
        console.print(f"[green]• Best performing model: {best_model} ({best_acc:.4f} test accuracy)[/green]")
        
        if best_acc > 0.853:  # Better than ResNet18
            console.print("[green]• EfficientNet outperforms ResNet18 baseline![/green]")
        
    console.print("• Consider ensemble of top performing models")
    console.print("• Test with different augmentation levels")
    console.print("• Analyze per-quality-tier performance")


if __name__ == "__main__":
    main()