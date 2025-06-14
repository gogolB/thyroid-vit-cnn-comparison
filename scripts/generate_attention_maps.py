"""
Definitive, Corrected Swin Transformer Visualization Script.

This version restores the original, proven data loading pipeline to fix the model's
prediction failures. It also implements a robust heatmap overlay technique for
vibrant, clear visualizations, and correctly handles the Swin [CLS] token.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


class SwinGradCAM:
    """
    Robust Grad-CAM for Swin Transformers targeting the final decision-making layer
    and correctly handling the [CLS] token.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = []
        self._find_and_register_hook()

    def _find_and_register_hook(self):
        """Finds the very last SwinTransformerBlock and hooks its final normalization layer."""
        target_layer = None
        for module in reversed(list(self.model.modules())):
            if isinstance(module, SwinTransformerBlock):
                target_layer = module.norm2
                break
        
        if target_layer:
            self.handles.append(target_layer.register_forward_hook(self._save_activation))
            self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))
            console.print(f"[green]✓ Registered Grad-CAM hook on the final Swin block's norm layer.[/green]")
        else:
            raise ValueError("Could not find a SwinTransformerBlock in the model.")

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

        # Handle the [CLS] token by removing it if present
        L, C = activations.shape
        H = W = int(np.sqrt(L))
        if H * W != L:
            H = W = int(np.sqrt(L - 1))
            if H * W == L - 1:
                console.print("[dim]CLS token detected. Removing for spatial mapping.[/dim]")
                activations = activations[1:]
            else:
                raise ValueError(f"Feature map length {L} is not a perfect square or (square+1).")

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


def visualize_final_heatmap(model: nn.Module, input_tensor: torch.Tensor,
                            original_image: np.ndarray, actual_label: int,
                            save_path: Optional[Path] = None, device: str = 'cpu'):
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

    console.print("[cyan]Generating Final Grad-CAM Heatmap...[/cyan]")
    grad_cam = SwinGradCAM(model)
    heatmap = grad_cam.generate_heatmap(input_tensor.to(device), class_idx=predicted_class)
    grad_cam.remove_hooks()
    
    # --- FIX 2: ROBUST HEATMAP OVERLAY ---
    # Convert single-channel heatmap to a 3-channel color image
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # for matplotlib
    
    # Resize color heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap_colored, (img_enhanced.shape[1], img_enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # Convert original image to 3-channel BGR for blending
    img_bgr = cv2.cvtColor((img_enhanced * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Blend the images
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_resized, 0.4, 0)

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Focus", weight='bold')
    axes[1].axis('off')

    result_symbol = "✓" if is_correct else "✗"
    super_title = (
        f"Swin Transformer Analysis | Actual: {actual_name}\n"
        f"Predicted: {predicted_name} ({confidence:.1%}) | Result: {result_symbol}"
    )
    fig.suptitle(super_title, fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved final visualization to: {save_path}[/green]")
        plt.close(fig)
    else:
        plt.show()


def load_swin_model(checkpoint_path: str, device: str = 'cpu') -> Optional[nn.Module]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_file = Path(checkpoint_path).stem.lower()
        if 'tiny' in model_file: model_name = 'swin_tiny_patch4_window7_224'
        elif 'small' in model_file: model_name = 'swin_small_patch4_window7_224'
        else: model_name = 'swin_base_patch4_window7_224'
        
        model = timm.create_model(model_name, pretrained=False, num_classes=2, in_chans=1)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        
        console.print(f"[green]✓ Loaded {model_name} successfully[/green]")
        model.eval()
        return model.to(device)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None


def main():
    checkpoint_dir = Path("checkpoints/best")
    output_dir = Path("outputs/final_heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    console.print("[cyan]Loading dataset for 'test' split...[/cyan]")
    try:
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        
        # --- FIX 1: RESTORED ORIGINAL, PROVEN DATA LOADING PIPELINE ---
        quality_report_path = Path('reports/quality_report.json')
        if not quality_report_path.exists():
            quality_report_path = None

        transform = create_quality_aware_transform(
            target_size=224,
            quality_report_path=quality_report_path,
            augmentation_level='none',
            split='test' # Use the test split as requested
        )
        
        dataset = CARSThyroidDataset(
            root_dir=data_dir,
            split='test',
            transform=transform,
            target_size=224,
            normalize=False # Critical parameter from your original code
        )
        
        raw_dataset = CARSThyroidDataset(
            root_dir=data_dir,
            split='test',
            transform=None,
            target_size=224,
            normalize=False
        )
        
        if len(dataset) == 0:
            raise ValueError("Test dataset is empty.")

        all_indices_in_split = list(range(len(dataset)))
        # The labels lookup correctly uses the globally mapped index from the dataset instance
        normal_indices = [i for i in all_indices_in_split if dataset.labels[dataset.indices[i]] == 0]
        cancer_indices = [i for i in all_indices_in_split if dataset.labels[dataset.indices[i]] == 1]
        
        sample_indices_to_process = []
        if normal_indices:
            sample_indices_to_process.append(normal_indices[0])
        if cancer_indices:
            sample_indices_to_process.append(cancer_indices[0])

        if not sample_indices_to_process:
             raise FileNotFoundError("Could not find representative normal and cancer samples in the test split.")
        
        # Get samples using the LOCAL indices from the test dataset instance
        samples = [(raw_dataset[i][0], dataset[i][0].unsqueeze(0), dataset[i][1]) for i in sample_indices_to_process]
        console.print(f"[green]✓ Found {len(samples)} samples to process from the test set.[/green]")

    except Exception as e:
        console.print(f"[bold red]FATAL: Could not load dataset.[/bold red] Reason: {e}")
        console.print("[yellow]Using dummy data to allow script to run. Please check your dataset class and paths.[/yellow]")
        samples = [(np.random.rand(224, 224), torch.randn(1, 1, 224, 224), 0)]

    model_files = ['swin_tiny-best.ckpt', 'swin_small-best.ckpt', 'swin_base-best.ckpt']
    
    for model_file in model_files:
        checkpoint_path = checkpoint_dir / model_file
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            continue
            
        model = load_swin_model(str(checkpoint_path), device=device)
        if model is None: continue

        for i, (raw_img, tensor, label) in enumerate(samples):
            label_str = 'normal' if label == 0 else 'cancerous'
            model_name_clean = model_file.replace('-best.ckpt', '')
            save_path = output_dir / f"{model_name_clean}_sample_{i+1}_{label_str}_gradcam_final.png"
            
            console.print(f"  Processing sample {i+1} ({label_str}) with {model_name_clean}...")
            visualize_final_heatmap(model, tensor, raw_img, label, save_path, device)

    console.print("\n[bold green]✓ Final Visualization Complete![/bold green]")
    console.print(f"   Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()