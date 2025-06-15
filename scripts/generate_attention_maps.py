"""
Definitive Swin Transformer Visualization Script.

This final version corrects the feature extractor to robustly handle both 3D and 4D
tensor shapes produced by different Swin stages, ensuring all feature maps are
generated correctly for a complete and publication-ready analysis.
"""

import sys
import argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
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

        L, C = activations.shape
        H = W = int(np.sqrt(L))
        if H * W != L:
            H = W = int(np.sqrt(L - 1))
            if H * W == L - 1:
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


class SwinFeatureExtractor:
    """
    Extracts intermediate features from Swin stages and creates artifact-free
    spatial summary maps.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = {}
        self.hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach().cpu()
        return hook

    def extract_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hooks all stages and performs a forward pass to extract features."""
        hook_points = [('Patch Embed', self.model.patch_embed)]
        for i, layer in enumerate(self.model.layers):
            stage_name = f'Stage {i+1}'
            self.hooks.append(layer.register_forward_hook(self._make_hook(stage_name)))

        with torch.no_grad():
            _ = self.model(input_tensor)

        for handle in self.hooks:
            handle.remove()
        
        return self.features

    @staticmethod
    def get_spatial_summary(feature_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a high-dimensional feature tensor into a robust 2D summary map.
        This corrected version robustly handles both 3D and 4D tensor shapes.
        """
        # --- THE DEFINITIVE FIX: Handle both 3D and 4D Tensors ---
        if feature_tensor.dim() == 4: # Spatial format: [B, H, W, C]
            # Squeeze batch dimension, already in spatial format
            tensor_spatial = feature_tensor.squeeze(0) # -> [H, W, C]
        
        elif feature_tensor.dim() == 3: # Sequence format: [B, L, C]
            tensor_seq = feature_tensor.squeeze(0) # -> [L, C]
            L, C = tensor_seq.shape
            H = W = int(np.sqrt(L))
            if H * W != L:
                # Handle CLS token if present
                H = W = int(np.sqrt(L - 1))
                if H * W == L - 1:
                    tensor_seq = tensor_seq[1:]
                else: # Cannot form a square, return blank
                    return np.zeros((224, 224))
            tensor_spatial = tensor_seq.reshape(H, W, C) # -> [H, W, C]
        
        else: # Unexpected shape
            console.print(f"[yellow]Warning: Unexpected feature tensor shape {feature_tensor.shape}. Skipping.[/yellow]")
            return np.zeros((224, 224))
        
        # --- Common processing for the now-guaranteed [H, W, C] tensor ---
        summary_map = tensor_spatial.std(dim=-1).numpy()
        
        summary_map = cv2.GaussianBlur(summary_map, (3, 3), 0)
        if summary_map.max() > summary_map.min():
            summary_map = (summary_map - summary_map.min()) / (summary_map.max() - summary_map.min())
            
        return summary_map


def visualize_comprehensive_analysis(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    actual_label: int,
    save_path: Optional[Path] = None,
    device: str = 'cpu'
):
    """
    Generates a comprehensive 2x3 analysis figure including Grad-CAM and feature maps.
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)

    img_enhanced = enhance_contrast(original_image)
    axes[0, 0].imshow(img_enhanced, cmap='gray')
    axes[0, 0].set_title("Original Image", weight='bold')
    axes[0, 0].axis('off')

    console.print("  Generating Grad-CAM...")
    grad_cam = SwinGradCAM(model)
    heatmap = grad_cam.generate_heatmap(input_tensor.to(device), class_idx=predicted_class)
    grad_cam.remove_hooks()
    
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap_colored, (img_enhanced.shape[1], img_enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    img_bgr = cv2.cvtColor((img_enhanced * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_resized, 0.4, 0)

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title("Grad-CAM Focus", weight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].axis('off') 

    console.print("  Extracting intermediate features...")
    feature_extractor = SwinFeatureExtractor(model)
    features = feature_extractor.extract_features(input_tensor.to(device))
    
    feature_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
    stages_to_show = ['Stage 1', 'Stage 2', 'Stage 4']

    for i, stage_name in enumerate(stages_to_show):
        ax = feature_axes[i]
        if stage_name in features:
            summary_map = SwinFeatureExtractor.get_spatial_summary(features[stage_name])
            if summary_map.any():
                # Resize the feature map to a common display size for consistency
                display_map = cv2.resize(summary_map, (224, 224), interpolation=cv2.INTER_LINEAR)
                ax.imshow(display_map, cmap='viridis')
                ax.set_title(f"{stage_name} Features", weight='bold')
            else:
                ax.text(0.5, 0.5, "Invalid Features", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax.axis('off')

    result_symbol = "✓" if is_correct else "✗"
    super_title = (
        f"Comprehensive Analysis: {model_name} | Actual: {actual_name}\n"
        f"Predicted: {predicted_name} ({confidence:.1%}) | Result: {result_symbol}"
    )
    fig.suptitle(super_title, fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]✓ Saved comprehensive analysis to: {save_path}[/green]")
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
        
        console.print(f"[green]✓ Loaded {model_name}[/green]")
        model.eval()
        return model.to(device)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualizations for Swin Transformer models.")
    args = parser.parse_args()
    
    checkpoint_dir = Path("checkpoints/best")
    output_dir = Path("outputs/comprehensive_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    console.print("[cyan]Loading dataset for 'test' split...[/cyan]")
    try:
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        
        quality_report_path = Path('reports/quality_report.json')
        if not quality_report_path.exists(): quality_report_path = None

        transform = create_quality_aware_transform(
            target_size=224, quality_report_path=quality_report_path, augmentation_level='none', split='test'
        )
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
        console.print(f"[green]✓ Found {len(samples)} samples to process from the test set.[/green]")

    except Exception as e:
        console.print(f"[bold red]FATAL: Could not load dataset.[/bold red] Reason: {e}")
        return

    model_definitions = [
        {'name': 'Swin-Tiny', 'file': 'swin_tiny-best.ckpt'},
        {'name': 'Swin-Small', 'file': 'swin_small-best.ckpt'},
        {'name': 'Swin-Base', 'file': 'swin_base-best.ckpt'}
    ]

    for model_def in model_definitions:
        console.print(f"\n[bold cyan]Processing Model: {model_def['name']}[/bold cyan]")
        checkpoint_path = checkpoint_dir / model_def['file']
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}. Skipping.[/yellow]")
            continue
            
        model = load_swin_model(str(checkpoint_path), device=device)
        if model is None: continue

        for i, (raw_img, tensor, label) in enumerate(samples):
            label_str = 'normal' if label == 0 else 'cancerous'
            console.print(f"  Processing sample {i+1} (Actual: {label_str})...")

            save_path = output_dir / f"{model_def['name']}_sample_{i+1}_{label_str}_comprehensive.png"
            visualize_comprehensive_analysis(
                model=model,
                model_name=model_def['name'],
                original_image=raw_img,
                input_tensor=tensor,
                actual_label=label,
                save_path=save_path,
                device=device
            )

    console.print("\n[bold green]✓ Comprehensive Visualization Complete![/bold green]")
    console.print(f"   Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()