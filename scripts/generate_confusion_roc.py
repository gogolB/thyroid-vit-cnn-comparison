#!/usr/bin/env python3
"""
Generate confusion matrices and ROC curves for best models.
Saves visualizations to visualizations/ppt-report/
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
import pytorch_lightning as pl
from rich.console import Console

console = Console()

# Model configurations
MODELS = {
    'swin_tiny': {
        'checkpoint': 'checkpoints/best/swin_tiny-best.ckpt',
        'accuracy': 94.12,
        'type': 'vit'
    },
    'resnet50': {
        'checkpoint': 'checkpoints/best/resnet50-best.ckpt',
        'accuracy': 91.18,
        'type': 'cnn'
    },
    'efficientnet_b0': {
        'checkpoint': 'checkpoints/best/efficientnet_b0-best.ckpt',
        'accuracy': 89.71,
        'type': 'cnn'
    }
}

def load_model_from_checkpoint(checkpoint_path):
    """Load a model from checkpoint."""
    console.print(f"[cyan]Loading model from {checkpoint_path}...[/cyan]")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to extract the model class from the checkpoint
        if 'hyper_parameters' in checkpoint:
            model_name = checkpoint['hyper_parameters'].get('model', {}).get('name', '')
            
            # Import the appropriate model class
            if 'swin' in model_name:
                from src.models.vit.swin_transformer import SwinTransformer
                model_class = SwinTransformer
            elif 'resnet' in model_name:
                from src.models.cnn.resnet import ResNet50
                model_class = ResNet50
            elif 'efficientnet' in model_name:
                from src.models.cnn.efficientnet import EfficientNetB0
                model_class = EfficientNetB0
            else:
                raise ValueError(f"Unknown model type: {model_name}")
        
        # Load using PyTorch Lightning
        model = model_class.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        console.print(f"[green]✓ Successfully loaded {model_name}[/green]")
        return model
        
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        console.print("[yellow]Attempting alternative loading method...[/yellow]")
        
        # Alternative: try loading the LightningModule directly
        try:
            from src.training.lightning_module import ThyroidClassificationModule
            model = ThyroidClassificationModule.load_from_checkpoint(checkpoint_path)
            model.eval()
            console.print("[green]✓ Successfully loaded using LightningModule[/green]")
            return model
        except Exception as e2:
            console.print(f"[red]Alternative loading failed: {e2}[/red]")
            return None


def get_test_dataloader(batch_size=32):
    """Create test dataloader."""
    transform = get_validation_transforms(target_size=224, normalize=True)
    
    dataset = CARSThyroidDataset(
        root_dir='data/raw',
        split='test',
        transform=transform,
        target_size=224,
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
    """Generate predictions for the entire test set."""
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            images, labels = batch
            images = images.to(device)
            
            # Get model outputs
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output', outputs))
            else:
                logits = outputs
            
            # Get probabilities and predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Cancerous'],
                yticklabels=['Normal', 'Cancerous'],
                annot_kws={'size': 16})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add accuracy text
    accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
    plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2f}%', 
             ha='center', va='top', transform=plt.gca().transAxes,
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved confusion matrix to {save_path}[/green]")
    return cm


def plot_roc_curve(y_true, y_probs, model_name, save_path):
    """Plot and save ROC curve."""
    # Get probabilities for positive class (cancerous)
    y_score = y_probs[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved ROC curve to {save_path}[/green]")
    return roc_auc


def generate_classification_report(y_true, y_pred, model_name, save_path):
    """Generate and save classification report."""
    report = classification_report(y_true, y_pred, 
                                 target_names=['Normal', 'Cancerous'],
                                 output_dict=True)
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also create a text version
    report_text = classification_report(y_true, y_pred, 
                                      target_names=['Normal', 'Cancerous'])
    
    text_path = save_path.replace('.json', '.txt')
    with open(text_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*50 + "\n\n")
        f.write(report_text)
    
    console.print(f"[green]✓ Saved classification report to {save_path}[/green]")
    return report


def create_comparison_plot(all_results, save_path):
    """Create a comparison plot of all models."""
    models = list(all_results.keys())
    accuracies = [all_results[m]['accuracy'] for m in models]
    aucs = [all_results[m]['auc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, [a*100 for a in aucs], width, label='AUC×100', color='lightcoral')
    
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Score (%)', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved comparison plot to {save_path}[/green]")


def main():
    """Main function to generate all visualizations."""
    # Create output directory
    output_dir = Path('visualizations/ppt-report')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Generating Confusion Matrices and ROC Curves[/bold cyan]\n")
    
    # Load test dataloader
    test_dataloader = get_test_dataloader()
    
    # Store results for comparison
    all_results = {}
    
    # Process each model
    for model_name, model_info in MODELS.items():
        console.print(f"\n[bold yellow]Processing {model_name}...[/bold yellow]")
        
        checkpoint_path = Path(model_info['checkpoint'])
        if not checkpoint_path.exists():
            console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
            continue
        
        # Load model
        model = load_model_from_checkpoint(checkpoint_path)
        if model is None:
            continue
        
        # Generate predictions
        y_pred, y_true, y_probs = generate_predictions(model, test_dataloader)
        
        # Create visualizations
        cm = plot_confusion_matrix(
            y_true, y_pred, model_name,
            output_dir / f'confusion_matrix_{model_name}.png'
        )
        
        roc_auc = plot_roc_curve(
            y_true, y_probs, model_name,
            output_dir / f'roc_curve_{model_name}.png'
        )
        
        report = generate_classification_report(
            y_true, y_pred, model_name,
            output_dir / f'classification_report_{model_name}.json'
        )
        
        # Calculate accuracy from confusion matrix
        accuracy = np.sum(np.diag(cm)) / np.sum(cm) * 100
        
        # Store results
        all_results[model_name] = {
            'accuracy': accuracy,
            'auc': roc_auc,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        console.print(f"[green]✓ Completed {model_name}: Acc={accuracy:.2f}%, AUC={roc_auc:.3f}[/green]")
    
    # Create comparison plot
    if all_results:
        create_comparison_plot(all_results, output_dir / 'model_comparison.png')
        
        # Save summary results
        with open(output_dir / 'results_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    console.print("\n[bold green]✓ All visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


if __name__ == "__main__":
    main()
