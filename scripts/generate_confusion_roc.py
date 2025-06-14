#!/usr/bin/env python3
"""
Generate confusion matrices and ROC curves for best models.
Saves visualizations to visualizations/ppt-report/

Fixed version that:
- Uses Lightning module loading (train_cnn.py and train_vit.py)
- Properly handles model outputs from Lightning modules
- Adds extensive debugging to diagnose prediction issues
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CARSThyroidDataset
from src.data.transforms import get_validation_transforms
from torch.utils.data import DataLoader
from rich.console import Console

# Import our model loading utilities
from scripts.model_loader_utils import load_model_from_checkpoint, get_model_output, debug_model_output

console = Console()

# Model configurations
MODELS = {
    'swin_tiny': {
        'checkpoint': 'checkpoints/best/swin_tiny-best.ckpt',
        'accuracy': 94.12,
        'type': 'vit',
        'image_size': 224  # Swin uses 224
    },
    'resnet50': {
        'checkpoint': 'checkpoints/best/resnet50-best.ckpt',
        'accuracy': 91.18,
        'type': 'cnn',
        'image_size': 256
    },
    'efficientnet_b0': {
        'checkpoint': 'checkpoints/best/efficientnet_b0-best.ckpt',
        'accuracy': 89.71,
        'type': 'cnn',
        'image_size': 256
    }
}


def get_test_dataloader(batch_size=32, target_size=256):
    """Create test dataloader with specified target size."""
    transform = get_validation_transforms(target_size=target_size, normalize=True)
    
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',
        transform=transform,
        target_size=target_size,
        normalize=True,
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
    return dataloader


def generate_predictions(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate predictions for the entire test set.
    Handles both Lightning modules and raw models properly.
    """
    # Move model to device (handles both Lightning modules and raw models)
    model = model.to(device)
    model.eval()  # Ensure eval mode
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Check first batch to debug
    first_batch = True
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Generating predictions"):
            images = images.to(device)
            
            # Get model outputs using helper function
            logits = get_model_output(model, images)
            
            # Debug first batch
            if first_batch:
                console.print(f"\n[dim]First batch debug:[/dim]")
                console.print(f"  Input shape: {images.shape}")
                console.print(f"  Logits shape: {logits.shape}")
                console.print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                console.print(f"  Labels: {labels.tolist()}")
                first_batch = False
            
            # Ensure logits are 2D [batch_size, num_classes]
            if logits.dim() == 1:
                # Single output neuron for binary classification
                # Convert to two-class format
                probs_pos = torch.sigmoid(logits)
                probs_neg = 1 - probs_pos
                probs = torch.stack([probs_neg, probs_pos], dim=1)
                preds = (probs_pos > 0.5).long()
            else:
                # Standard multi-class output
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Debug predictions
                if len(all_preds) == 0:
                    console.print(f"  Predictions: {preds.tolist()}")
                    console.print(f"  Probabilities: {probs[0].tolist()}")
            
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
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Cancerous'],
                yticklabels=['Normal', 'Cancerous'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy text
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved confusion matrix: {save_path}[/green]")


def plot_roc_curve(y_true, y_probs, model_name, save_path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    
    # Ensure y_probs is 2D
    if y_probs.ndim == 1:
        # Single probability output
        y_probs_positive = y_probs
    else:
        # Use probability of positive class (cancerous)
        y_probs_positive = y_probs[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probs_positive)
    roc_auc = auc(fpr, tpr)
    
    # Plot
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
        
        # Handle different probability formats
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
    """Main function to generate confusion matrices and ROC curves."""
    # Create output directory
    output_dir = Path('visualizations/ppt-report')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Generating Confusion Matrices and ROC Curves[/bold cyan]\n")
    
    # Store results for combined plot
    all_results = {}
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    console.print(f"[dim]Using device: {device}[/dim]\n")
    
    # Process each model
    for model_name, config in MODELS.items():
        console.print(f"\n[cyan]Processing {model_name}...[/cyan]")
        
        # Check if checkpoint exists
        checkpoint_path = Path(config['checkpoint'])
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            console.print("[yellow]Skipping this model[/yellow]")
            all_results[model_name] = None
            continue
        
        # Load model using our improved utility
        model = load_model_from_checkpoint(checkpoint_path, device=device, verbose=True)
        
        if model is None:
            console.print(f"[yellow]Skipping {model_name} due to loading error[/yellow]")
            all_results[model_name] = None
            continue
        
        # Debug model output on a sample batch
        console.print("\n[cyan]Debugging model output...[/cyan]")
        sample_batch = torch.randn(2, 1, config['image_size'], config['image_size']).to(device)
        debug_model_output(model, sample_batch, model_name)
        
        # Get test dataloader with appropriate image size
        image_size = config.get('image_size', 256)
        console.print(f"[dim]Using image size: {image_size}x{image_size}[/dim]")
        test_loader = get_test_dataloader(target_size=image_size)
        
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
    
    # Generate combined ROC curve if we have any results
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        console.print("\n[cyan]Generating combined ROC curve...[/cyan]")
        combined_roc_path = output_dir / 'roc_curves_comparison.png'
        plot_combined_roc_curves(valid_results, combined_roc_path)
    
    # Save numerical results
    results_summary = {}
    for model_name, results in all_results.items():
        if results is None:
            continue
            
        accuracy = np.sum(results['preds'] == results['labels']) / len(results['labels'])
        cm = confusion_matrix(results['labels'], results['preds'])
        
        # Calculate per-class metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results_summary[model_name] = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'total_samples': len(results['labels']),
            'confusion_matrix': cm.tolist()
        }
    
    # Save results summary
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    console.print("\n[bold green]✓ All visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


if __name__ == "__main__":
    main()