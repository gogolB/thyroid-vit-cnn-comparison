#!/usr/bin/env python3
"""
Generate confusion matrices and ROC curves for best models.
Fixed version with quality-aware preprocessing and working Swin loading.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import json
from tqdm import tqdm
import timm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CARSThyroidDataset
from src.data.quality_preprocessing import create_quality_aware_transform
from torch.utils.data import DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import model loading utilities
from scripts.model_loader_utils import get_model_output

console = Console()

# Model configurations
MODELS = {
    'swin_tiny': {
        'checkpoint': 'checkpoints/best/swin_tiny-best.ckpt',
        'accuracy': 0.9412,  # As decimal
        'type': 'vit',
        'image_size': 224  # Swin uses 224
    },
    'resnet50': {
        'checkpoint': 'checkpoints/best/resnet50-best.ckpt',
        'accuracy': 0.9118,  # As decimal
        'type': 'cnn',
        'image_size': 256
    },
    'efficientnet_b0': {
        'checkpoint': 'checkpoints/best/efficientnet_b0-best.ckpt',
        'accuracy': 0.8971,  # As decimal
        'type': 'cnn',
        'image_size': 256
    }
}


def load_swin_minimal(checkpoint_path, device='cpu'):
    """
    Minimal Swin loader that works with dimension mismatches.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Get image size
        img_size = 224
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            if isinstance(hp, dict) and 'config' in hp and 'dataset' in hp['config']:
                img_size = hp['config']['dataset'].get('image_size', 224)
        
        # Create model with timm
        model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=2,
            in_chans=1,
            img_size=img_size
        )
        
        # Clean and load state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            cleaned_state_dict[clean_key] = v
        
        # Load only matching parameters
        model_state = model.state_dict()
        loaded_state = {}
        for key in model_state.keys():
            if key in cleaned_state_dict and cleaned_state_dict[key].shape == model_state[key].shape:
                loaded_state[key] = cleaned_state_dict[key]
        
        model.load_state_dict(loaded_state, strict=False)
        console.print(f"[green]✓ Loaded Swin model ({len(loaded_state)}/{len(model_state)} params)[/green]")
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        console.print(f"[red]Swin loading failed: {e}[/red]")
        return None


def load_model_fixed(checkpoint_path, model_name, device='cpu'):
    """
    Fixed model loading that handles all model types.
    """
    console.print(f"[cyan]Loading {model_name} from {checkpoint_path}...[/cyan]")
    
    try:
        if 'swin' in model_name:
            # Use minimal loader for Swin
            return load_swin_minimal(checkpoint_path, device)
        else:
            # Load CNN models using Lightning
            from src.training.train_cnn import ThyroidCNNModule
            
            model_module = ThyroidCNNModule.load_from_checkpoint(
                checkpoint_path,
                map_location=device,
                weights_only=False,
                strict=False
            )
            console.print(f"[green]✓ Loaded {model_name} successfully[/green]")
            model_module.eval()
            return model_module.to(device)
            
    except Exception as e:
        console.print(f"[red]Failed to load {model_name}: {e}[/red]")
        return None


def get_test_dataloader_with_quality_preprocessing(batch_size=32, target_size=256):
    """
    Create test dataloader with quality-aware preprocessing.
    """
    # Get quality report path
    quality_report_path = Path('reports/quality_report.json')
    
    if not quality_report_path.exists():
        console.print("[yellow]Warning: Quality report not found. Using standard preprocessing.[/yellow]")
        quality_report_path = None
    else:
        console.print("[green]Using quality-aware preprocessing[/green]")
    
    # Create quality-aware transform for validation/test
    transform = create_quality_aware_transform(
        target_size=target_size,
        quality_report_path=quality_report_path,
        augmentation_level='none',  # No augmentation for test
        split='test'
    )
    
    # Create dataset with quality-aware preprocessing
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',
        transform=transform,
        target_size=target_size,
        normalize=False,  # Normalization handled in transform
        patient_level_split=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    console.print(f"[green]Test dataset loaded: {len(dataset)} images[/green]")
    console.print(f"[dim]Using quality-aware preprocessing: {quality_report_path is not None}[/dim]")
    
    return dataloader


def generate_predictions(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate predictions with proper preprocessing.
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    first_batch = True
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Generating predictions")):
            images = images.to(device)
            
            # Get model outputs
            logits = get_model_output(model, images)
            
            # Debug first batch
            if first_batch:
                console.print(f"\n[dim]First batch debug:[/dim]")
                console.print(f"  Input shape: {images.shape}")
                console.print(f"  Input range: [{images.min():.3f}, {images.max():.3f}]")
                console.print(f"  Logits shape: {logits.shape}")
                console.print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                console.print(f"  Labels: {labels.tolist()}")
                first_batch = False
            
            # Handle predictions
            if logits.dim() == 1:
                probs_pos = torch.sigmoid(logits)
                probs_neg = 1 - probs_pos
                probs = torch.stack([probs_neg, probs_pos], dim=1)
                preds = (probs_pos > 0.5).long()
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Check prediction distribution
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    console.print(f"\n[dim]Prediction distribution:[/dim]")
    for pred, count in zip(unique_preds, counts):
        console.print(f"  Class {pred}: {count} ({count/len(all_preds)*100:.1f}%)")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot confusion matrix with improved styling."""
    plt.figure(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text with both count and percentage
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_text[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot_text, fmt='', cmap='Blues',
                xticklabels=['Normal', 'Cancerous'],
                yticklabels=['Normal', 'Cancerous'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved confusion matrix: {save_path}[/green]")


def plot_roc_curve(y_true, y_probs, model_name, save_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    
    if y_probs.ndim == 1:
        y_probs_positive = y_probs
    else:
        y_probs_positive = y_probs[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_probs_positive)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved ROC curve: {save_path}[/green]")


def plot_combined_roc_curves(all_results, save_path):
    """Plot all ROC curves on one figure."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        if results is None:
            continue
            
        y_true = results['labels']
        y_probs = results['probs']
        
        if y_probs.ndim == 1:
            y_probs_positive = y_probs
        else:
            y_probs_positive = y_probs[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_probs_positive)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx % len(colors)],
                linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved combined ROC curves: {save_path}[/green]")


def main():
    """Main function with quality-aware preprocessing."""
    output_dir = Path('visualizations/ppt-report')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(Panel.fit(
        "[bold cyan]Generating Confusion Matrices and ROC Curves[/bold cyan]\n"
        "[dim]With Quality-Aware Preprocessing[/dim]",
        border_style="blue"
    ))
    
    all_results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[dim]Using device: {device}[/dim]\n")
    
    # Process each model
    for model_name, config in MODELS.items():
        console.print(f"\n[cyan]Processing {model_name}...[/cyan]")
        
        checkpoint_path = Path(config['checkpoint'])
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            all_results[model_name] = None
            continue
        
        # Load model with fixed loading function
        model = load_model_fixed(checkpoint_path, model_name, device)
        
        if model is None:
            console.print(f"[yellow]Skipping {model_name} due to loading error[/yellow]")
            all_results[model_name] = None
            continue
        
        # Get test dataloader with quality preprocessing
        image_size = config.get('image_size', 256)
        console.print(f"[dim]Using image size: {image_size}x{image_size}[/dim]")
        test_loader = get_test_dataloader_with_quality_preprocessing(target_size=image_size)
        
        # Generate predictions
        preds, labels, probs = generate_predictions(model, test_loader, device)
        
        # Store results
        all_results[model_name] = {
            'preds': preds,
            'labels': labels,
            'probs': probs
        }
        
        # Calculate metrics
        accuracy = np.sum(preds == labels) / len(labels)
        console.print(f"[green]Test Accuracy: {accuracy:.4f}[/green]")
        
        # Plot confusion matrix
        cm_path = output_dir / f'confusion_matrix_{model_name}.png'
        plot_confusion_matrix(labels, preds, model_name, cm_path)
        
        # Plot individual ROC curve
        roc_path = output_dir / f'roc_curve_{model_name}.png'
        plot_roc_curve(labels, probs, model_name, roc_path)
        
        # Print classification report
        console.print(f"\n[cyan]Classification Report - {model_name}:[/cyan]")
        report = classification_report(labels, preds,
                                     target_names=['Normal', 'Cancerous'],
                                     digits=4)
        console.print(report)
    
    # Generate combined ROC curve
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        console.print("\n[cyan]Generating combined ROC curve...[/cyan]")
        combined_roc_path = output_dir / 'roc_curves_comparison.png'
        plot_combined_roc_curves(valid_results, combined_roc_path)
    
    # Save results summary
    results_summary = {}
    
    for model_name, config in MODELS.items():
        if model_name in all_results and all_results[model_name] is not None:
            results = all_results[model_name]
            accuracy = np.sum(results['preds'] == results['labels']) / len(results['labels'])
            cm = confusion_matrix(results['labels'], results['preds'])
            
            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results_summary[model_name] = {
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'total_samples': len(results['labels']),
                'confusion_matrix': cm.tolist(),
                'status': 'tested'
            }
        else:
            # Include training results for models that failed to load
            results_summary[model_name] = {
                'accuracy': config['accuracy'] / 100.0,  # Convert percentage to decimal
                'status': 'training_only',
                'note': 'Model could not be loaded for testing'
            }
    
    # Save summary
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    console.print("\n[bold green]✓ All visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")
    
    # Print final summary
    console.print("\n[bold cyan]Final Results Summary:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Test Accuracy", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Status", justify="center")
    
    for model_name, config in MODELS.items():
        if model_name in results_summary:
            result = results_summary[model_name]
            if result['status'] == 'tested':
                table.add_row(
                    model_name,
                    f"{result['accuracy']:.2%}",
                    f"{config['accuracy']/100:.2%}",  # Divide by 100 since it's already a percentage
                    "[green]✓ Tested[/green]"
                )
            else:
                table.add_row(
                    model_name,
                    "N/A",
                    f"{config['accuracy']/100:.2%}",  # Divide by 100 since it's already a percentage
                    "[yellow]Training only[/yellow]"
                )
    
    console.print(table)


if __name__ == "__main__":
    main()