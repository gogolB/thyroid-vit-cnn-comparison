#!/usr/bin/env python3
"""
Generate attention visualizations for Swin Transformer.
Creates attention maps showing what the model focuses on.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import CARSThyroidDataset
from src.data.transforms import get_validation_transforms
from torch.utils.data import DataLoader
from rich.console import Console

console = Console()


class AttentionExtractor:
    """Extract attention weights from Swin Transformer."""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention(name):
            def hook(module, input, output):
                # Swin Transformer attention output format varies
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Sometimes attention weights are the second output
                    self.attention_weights[name] = output[1].detach()
                elif hasattr(output, 'attention_weights'):
                    self.attention_weights[name] = output.attention_weights.detach()
            return hook
        
        # Register hooks for Swin Transformer layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and not 'drop' in name.lower():
                hook = module.register_forward_hook(get_attention(name))
                self.hooks.append(hook)
                
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_attention_maps(self, input_tensor):
        """Get attention maps for an input."""
        self.attention_weights = {}
        self.register_hooks()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        self.remove_hooks()
        return self.attention_weights


def load_swin_model(checkpoint_path):
    """Load Swin Transformer model."""
    console.print(f"[cyan]Loading Swin Transformer from {checkpoint_path}...[/cyan]")
    
    try:
        # Import Swin Transformer
        from src.models.vit.swin_transformer import SwinTransformer
        
        # Load checkpoint
        model = SwinTransformer.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        console.print("[green]✓ Successfully loaded Swin Transformer[/green]")
        return model
        
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        
        # Try alternative loading
        try:
            from src.training.lightning_module import ThyroidClassificationModule
            model = ThyroidClassificationModule.load_from_checkpoint(checkpoint_path)
            model.eval()
            console.print("[green]✓ Successfully loaded using LightningModule[/green]")
            return model
        except Exception as e2:
            console.print(f"[red]Alternative loading failed: {e2}[/red]")
            return None


def visualize_attention_rollout(attention_weights, image, save_path):
    """Visualize attention using rollout technique."""
    # Get the last layer's attention
    last_attn = None
    for key in sorted(attention_weights.keys()):
        if attention_weights[key].numel() > 0:
            last_attn = attention_weights[key]
    
    if last_attn is None:
        console.print("[yellow]No attention weights found[/yellow]")
        return
    
    # Average attention heads
    if last_attn.dim() == 4:  # [batch, heads, seq, seq]
        attn = last_attn.mean(dim=1)  # Average over heads
    else:
        attn = last_attn
    
    # Get attention map size
    if attn.dim() == 3:
        attn = attn[0]  # Remove batch dimension
    
    # Resize attention to match image size
    h, w = image.shape[-2:]
    attn_size = int(np.sqrt(attn.shape[0]))
    
    if attn.shape[0] == attn.shape[1]:
        # Reshape square attention matrix
        attn_map = attn.mean(dim=0).reshape(attn_size, attn_size)
    else:
        # Take mean over sequence dimension
        attn_map = attn.mean(dim=0).reshape(-1)
        attn_size = int(np.sqrt(len(attn_map)))
        attn_map = attn_map[:attn_size**2].reshape(attn_size, attn_size)
    
    # Resize to image size
    attn_map = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=(h, w),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Normalize attention map
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img_np = image.squeeze().cpu().numpy()
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attn_map, cmap='hot')
    axes[1].set_title('Attention Map', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(attn_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle('Swin Transformer Attention Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved attention visualization to {save_path}[/green]")


def create_multi_layer_attention_plot(attention_weights, image, save_path):
    """Create a plot showing attention from multiple layers."""
    # Select 4 layers evenly distributed
    layer_names = sorted([k for k in attention_weights.keys() if attention_weights[k].numel() > 0])
    
    if len(layer_names) < 4:
        selected_layers = layer_names
    else:
        indices = np.linspace(0, len(layer_names)-1, 4, dtype=int)
        selected_layers = [layer_names[i] for i in indices]
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Original image at top
    ax_orig = fig.add_subplot(gs[0, 1:3])
    img_np = image.squeeze().cpu().numpy()
    ax_orig.imshow(img_np, cmap='gray')
    ax_orig.set_title('Original CARS Image', fontsize=14, fontweight='bold')
    ax_orig.axis('off')
    
    # Attention maps from different layers
    for idx, layer_name in enumerate(selected_layers):
        row = 1 + idx // 2
        col = (idx % 2) * 2
        
        ax = fig.add_subplot(gs[row, col:col+2])
        
        # Process attention for this layer
        attn = attention_weights[layer_name]
        
        # Average over heads and batch
        if attn.dim() == 4:
            attn = attn[0].mean(dim=0)
        elif attn.dim() == 3:
            attn = attn[0]
        
        # Reshape and resize
        if attn.shape[0] == attn.shape[1]:
            attn_size = int(np.sqrt(attn.shape[0]))
            attn_map = attn.mean(dim=0).reshape(attn_size, attn_size)
        else:
            attn_map = attn.reshape(-1)
            attn_size = int(np.sqrt(len(attn_map)))
            attn_map = attn_map[:attn_size**2].reshape(attn_size, attn_size)
        
        # Resize to image size
        h, w = image.shape[-2:]
        attn_map = F.interpolate(
            attn_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        
        # Plot
        im = ax.imshow(attn_map, cmap='hot')
        
        # Extract layer depth from name
        layer_depth = layer_name.split('.')[1] if '.' in layer_name else '?'
        ax.set_title(f'Layer {layer_depth} Attention', fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Hierarchical Attention Maps - Swin Transformer', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved multi-layer attention plot to {save_path}[/green]")


def generate_class_specific_attention(model, dataloader, output_dir, n_samples=3):
    """Generate attention maps for samples from each class."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    extractor = AttentionExtractor(model)
    
    # Collect samples from each class
    normal_samples = []
    cancer_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            for i in range(len(labels)):
                if labels[i] == 0 and len(normal_samples) < n_samples:
                    normal_samples.append((images[i:i+1], labels[i]))
                elif labels[i] == 1 and len(cancer_samples) < n_samples:
                    cancer_samples.append((images[i:i+1], labels[i]))
                
                if len(normal_samples) >= n_samples and len(cancer_samples) >= n_samples:
                    break
            
            if len(normal_samples) >= n_samples and len(cancer_samples) >= n_samples:
                break
    
    # Create comparison figure
    fig, axes = plt.subplots(2, n_samples * 2, figsize=(n_samples * 8, 8))
    
    # Process normal samples
    for idx, (image, label) in enumerate(normal_samples):
        image = image.to(device)
        attention_weights = extractor.get_attention_maps(image)
        
        # Get last layer attention
        last_attn = None
        for key in sorted(attention_weights.keys()):
            if attention_weights[key].numel() > 0:
                last_attn = attention_weights[key]
        
        if last_attn is not None:
            # Process attention
            if last_attn.dim() == 4:
                attn = last_attn[0].mean(dim=0)
            else:
                attn = last_attn[0]
            
            # Create attention map
            attn_size = int(np.sqrt(attn.shape[0]))
            if attn.shape[0] == attn.shape[1]:
                attn_map = attn.mean(dim=0).reshape(attn_size, attn_size)
            else:
                attn_map = attn.reshape(-1)[:attn_size**2].reshape(attn_size, attn_size)
            
            # Resize
            h, w = image.shape[-2:]
            attn_map = F.interpolate(
                attn_map.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Normalize
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            
            # Plot original
            img_np = image.squeeze().cpu().numpy()
            axes[0, idx*2].imshow(img_np, cmap='gray')
            axes[0, idx*2].set_title(f'Normal #{idx+1}', fontsize=12)
            axes[0, idx*2].axis('off')
            
            # Plot attention overlay
            axes[0, idx*2+1].imshow(img_np, cmap='gray')
            axes[0, idx*2+1].imshow(attn_map, cmap='hot', alpha=0.6)
            axes[0, idx*2+1].set_title(f'Attention', fontsize=12)
            axes[0, idx*2+1].axis('off')
    
    # Process cancer samples
    for idx, (image, label) in enumerate(cancer_samples):
        image = image.to(device)
        attention_weights = extractor.get_attention_maps(image)
        
        # Similar processing as above
        last_attn = None
        for key in sorted(attention_weights.keys()):
            if attention_weights[key].numel() > 0:
                last_attn = attention_weights[key]
        
        if last_attn is not None:
            # Process attention
            if last_attn.dim() == 4:
                attn = last_attn[0].mean(dim=0)
            else:
                attn = last_attn[0]
            
            # Create attention map
            attn_size = int(np.sqrt(attn.shape[0]))
            if attn.shape[0] == attn.shape[1]:
                attn_map = attn.mean(dim=0).reshape(attn_size, attn_size)
            else:
                attn_map = attn.reshape(-1)[:attn_size**2].reshape(attn_size, attn_size)
            
            # Resize
            h, w = image.shape[-2:]
            attn_map = F.interpolate(
                attn_map.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()
            
            # Normalize
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            
            # Plot original
            img_np = image.squeeze().cpu().numpy()
            axes[1, idx*2].imshow(img_np, cmap='gray')
            axes[1, idx*2].set_title(f'Cancerous #{idx+1}', fontsize=12)
            axes[1, idx*2].axis('off')
            
            # Plot attention overlay
            axes[1, idx*2+1].imshow(img_np, cmap='gray')
            axes[1, idx*2+1].imshow(attn_map, cmap='hot', alpha=0.6)
            axes[1, idx*2+1].set_title(f'Attention', fontsize=12)
            axes[1, idx*2+1].axis('off')
    
    plt.suptitle('Swin Transformer Attention: Normal vs Cancerous Tissue', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'attention_comparison_by_class.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved class comparison to {save_path}[/green]")


def main():
    """Main function to generate attention visualizations."""
    # Create output directory
    output_dir = Path('visualizations/ppt-report')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold cyan]Generating Swin Transformer Attention Maps[/bold cyan]\n")
    
    # Load model
    checkpoint_path = Path('checkpoints/best/swin_tiny-best.ckpt')
    if not checkpoint_path.exists():
        console.print(f"[red]Swin checkpoint not found: {checkpoint_path}[/red]")
        return
    
    model = load_swin_model(checkpoint_path)
    if model is None:
        return
    
    # Create dataloader
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
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    # Generate visualizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create attention extractor
    extractor = AttentionExtractor(model)
    
    # Generate attention maps for a few samples
    console.print("[cyan]Generating individual attention maps...[/cyan]")
    
    for idx, (image, label) in enumerate(dataloader):
        if idx >= 5:  # Generate 5 examples
            break
        
        image = image.to(device)
        attention_weights = extractor.get_attention_maps(image)
        
        # Save individual attention visualization
        class_name = 'normal' if label.item() == 0 else 'cancerous'
        save_path = output_dir / f'attention_map_{class_name}_{idx}.png'
        visualize_attention_rollout(attention_weights, image, save_path)
        
        # Save multi-layer visualization for first sample of each class
        if idx < 2:
            save_path = output_dir / f'attention_layers_{class_name}.png'
            create_multi_layer_attention_plot(attention_weights, image, save_path)
    
    # Generate class comparison
    console.print("[cyan]Generating class-specific attention comparison...[/cyan]")
    generate_class_specific_attention(model, dataloader, output_dir, n_samples=3)
    
    console.print("\n[bold green]✓ All attention visualizations generated successfully![/bold green]")
    console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")


if __name__ == "__main__":
    main()
