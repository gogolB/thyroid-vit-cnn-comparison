"""
Definitive Grad-CAM Visualization Script for Vision Transformer Models.

This script is specifically tailored to generate high-quality, publication-ready
Grad-CAM heatmaps for ViT, DeiT, and Swin architectures, now enhanced with
contour lines for improved scientific clarity.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import cv2
from rich.console import Console
import warnings
import timm
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.vision_transformer import Block as ViTBlock

warnings.filterwarnings('ignore')
console = Console()


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast of CARS microscopy images with 2-98 percentile normalization."""
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = image.squeeze()

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    p2, p98 = np.percentile(image, (2, 98))
    enhanced = np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
    return enhanced


class ViTGradCAM:
    """Robust Grad-CAM for Vision Transformer architectures (ViT, DeiT, Swin)."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = []
        self._find_and_register_hook()

    def _find_and_register_hook(self):
        """Finds the final transformer block and hooks its final normalization layer."""
        target_layer = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, (SwinTransformerBlock, ViTBlock)):
                target_layer = module.norm2 if hasattr(module, 'norm2') else module.norm1
                break
        
        if target_layer:
            self.handles.append(target_layer.register_forward_hook(self._save_activation))
            self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))
            console.print(f"[green]✓ Registered Grad-CAM hook on the final transformer block's norm layer.[/green]")
        else:
            raise ValueError("Could not find a ViT/Swin block in the model.")

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generates the class activation heatmap, handling the [CLS] token."""
        self.model.eval()
        input_tensor.requires_grad = True
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to get gradients or activations for Grad-CAM.")

        pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
        activations = self.activations.squeeze(0)

        L, C = activations.shape
        H = W = int(np.sqrt(L))
        if H * W != L:
            H = W = int(np.sqrt(L - 1))
            if H * W == L - 1:
                console.print("[dim]CLS token detected. Removing for spatial mapping.[/dim]")
                activations = activations[1:]
            else:
                raise ValueError(f"Feature map length {L} cannot be reshaped into a square or (square+1).")

        for i in range(C):
            activations[:, i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy().reshape(H, W)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


def visualize_final_heatmap(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    actual_label: int,
    save_path: Optional[Path] = None,
    device: str = 'cpu'
):
    """
    Generates a simplified, high-impact 1x2 visualization with the final heatmap.
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(device))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    class_names = {0: 'Normal', 1: 'Cancerous'}
    actual_name = class_names[actual_label]
    predicted_name = class_names[predicted_class]
    is_correct = actual_label == predicted_class

    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    img_enhanced = enhance_contrast(original_image)
    axes[0].imshow(img_enhanced, cmap='gray')
    axes[0].set_title("Original Image", weight='bold')
    axes[0].axis('off')

    console.print("  Generating Grad-CAM Heatmap...")
    grad_cam = ViTGradCAM(model)
    heatmap = grad_cam.generate_heatmap(input_tensor.to(device), class_idx=predicted_class)
    grad_cam.remove_hooks()
    
    heatmap_resized = cv2.resize(heatmap, (img_enhanced.shape[1], img_enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Plot the base image and heatmap overlay
    axes[1].imshow(img_enhanced, cmap='gray')
    axes[1].imshow(heatmap_resized, cmap='inferno', alpha=0.5)
    
    # --- ENHANCEMENT: Add Contour Lines ---
    if heatmap.max() > 0.3:
        axes[1].contour(heatmap_resized, levels=[0.5, 0.7, 0.9], colors='white', linewidths=0.8, alpha=0.8)

    axes[1].set_title("Grad-CAM Focus", weight='bold')
    axes[1].axis('off')

    result_symbol = "✓" if is_correct else "✗"
    super_title = (
        f"ViT/Swin Analysis: {model_name} | Actual: {actual_name}\n"
        f"Predicted: {predicted_name} ({confidence:.1%}) | Result: {result_symbol}"
    )
    fig.suptitle(super_title, fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved final visualization to: {save_path}[/green]")
        plt.close(fig)
    else:
        plt.show()


def load_vit_model(model_name: str, checkpoint_path: str, device: str = 'cpu') -> Optional[nn.Module]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = timm.create_model(model_name, pretrained=False, num_classes=2, in_chans=1)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        console.print(f"[green]✓ Loaded {model_name} successfully from {Path(checkpoint_path).name}[/green]")
        model.eval()
        return model.to(device)
    except Exception as e:
        console.print(f"[red]Error loading model {model_name}: {e}[/red]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for ViT/Swin models.")
    parser.add_argument('--model-name', type=str, default='swin_tiny', help="The specific ViT/Swin model to visualize.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint file (.ckpt).")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path("outputs/vit_gradcam")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not checkpoint_path.exists():
        console.print(f"[red]Error: Checkpoint file not found at {checkpoint_path}[/red]")
        return

    model = load_vit_model(args.model_name, str(checkpoint_path), device=device)
    if model is None: return

    console.print("[cyan]Loading dataset for 'test' split...[/cyan]")
    try:
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        transform = create_quality_aware_transform(target_size=224, split='test')
        dataset = CARSThyroidDataset(root_dir=data_dir, split='test', transform=transform, target_size=224, normalize=False)
        raw_dataset = CARSThyroidDataset(root_dir=data_dir, split='test', transform=None, target_size=224, normalize=False)
        if len(dataset) == 0: raise ValueError("Test dataset is empty.")
        all_indices_in_split = list(range(len(dataset)))
        normal_indices = [i for i in all_indices_in_split if dataset.labels[dataset.indices[i]] == 0]
        cancer_indices = [i for i in all_indices_in_split if dataset.labels[dataset.indices[i]] == 1]
        sample_indices_to_process = []
        if normal_indices: sample_indices_to_process.append(normal_indices[0])
        if cancer_indices: sample_indices_to_process.append(cancer_indices[0])
        if not sample_indices_to_process: raise FileNotFoundError("Could not find representative samples in test split.")
        samples = [(raw_dataset[i][0], dataset[i][0].unsqueeze(0), dataset[i][1]) for i in sample_indices_to_process]
        console.print(f"[green]✓ Found {len(samples)} samples to process.[/green]")
    except Exception as e:
        console.print(f"[bold red]FATAL: Could not load dataset.[/bold red] Reason: {e}")
        return

    for i, (raw_img, tensor, label) in enumerate(samples):
        label_str = 'normal' if label == 0 else 'cancerous'
        console.print(f"\nProcessing sample {i+1} (Actual: {label_str})...")
        save_path = output_dir / f"{args.model_name}_sample_{i+1}_{label_str}.png"
        visualize_final_heatmap(model=model, model_name=args.model_name, original_image=raw_img, input_tensor=tensor, actual_label=label, save_path=save_path, device=device)

    console.print("\n[bold green]✓ ViT/Swin Grad-CAM Visualization Complete![/bold green]")
    console.print(f"   Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
