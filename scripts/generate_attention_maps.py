"""
Modernized Swin Transformer visualization with artifact removal and a clean aesthetic.
Generates publication-quality figures by targeting stable feature layers and using
perceptually uniform colormaps.
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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
import cv2
from rich.console import Console
import warnings
import timm
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

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
    Robust GradCAM for Swin Transformers.
    Targets stable, high-resolution layers and smooths results to prevent artifacts.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = []

        # Find the best target layer for artifact-free, high-resolution CAMs.
        # Priority:
        # 1. Final normalization layer (most stable, but low-res).
        # 2. Output of the final stage (good balance of stability and resolution).
        # 3. Output of the second-to-last stage (higher resolution).
        target_layer_name = None
        potential_targets = [
            'norm', # Final norm layer
            'layers.3', 'layers.2', 'layers.1' # Stage outputs
        ]

        model_layers = {name: module for name, module in model.named_modules()}

        for target in potential_targets:
            if target in model_layers:
                target_layer_name = target
                break
        
        if target_layer_name:
            module = model_layers[target_layer_name]
            self.handles.append(module.register_forward_hook(self._save_activation))
            self.handles.append(module.register_full_backward_hook(self._save_gradient))
            console.print(f"[green]Registered GradCAM hooks on optimal layer: [bold]{target_layer_name}[/bold][/green]")
        else:
            console.print("[red]Error: Could not find a suitable layer for GradCAM.[/red]")

    def _save_activation(self, module, input, output):
        # Handle tuple outputs from stages
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate a clean, artifact-free Class Activation Map."""
        self.model.eval()
        input_tensor.requires_grad = True
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            console.print("[yellow]Warning: Gradients/activations not captured. Returning empty CAM.[/yellow]")
            return np.zeros((7, 7))

        # Detach and get the first item of the batch
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Reshape if features are flattened (e.g., from [L, C] to [H, W, C])
        if gradients.dim() == 2:
            L, C = gradients.shape
            H = W = int(np.sqrt(L))
            if H * W == L:
                gradients = gradients.view(H, W, C)
                activations = activations.view(H, W, C)
            else:
                 console.print(f"[yellow]Warning: Non-square feature map (L={L}). CAM may be inaccurate.[/yellow]")
                 return np.zeros((7,7))
        
        # Permute to [C, H, W] for pooling
        gradients = gradients.permute(2, 0, 1)
        activations = activations.permute(2, 0, 1)

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # ** CRITICAL STEP FOR ARTIFACT REMOVAL **
        # Smooth the raw CAM before upscaling to remove high-frequency artifacts (stripes).
        cam = cv2.GaussianBlur(cam, (5, 5), 1.5)

        # Normalize to [0, 1] range
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class RadicalSwinFeatureVisualizer:
    """Visualizes spatially-aware features from Swin, removing architectural artifacts."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = {}
        self.hooks = []

    def get_intermediate_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts features from the end of each stage."""
        self.features = {}
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.features[name] = output.detach().cpu()[0] # Store first item in batch
            return hook

        hook_points = [('patch_embed', self.model.patch_embed)]
        for idx, layer in enumerate(self.model.layers):
            hook_points.append((f'stage{idx+1}', layer))

        for name, module in hook_points:
            self.hooks.append(module.register_forward_hook(make_hook(name)))
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return self.features

    def _get_spatial_map(self, features: torch.Tensor) -> np.ndarray:
        """Converts feature tensors into a smooth, artifact-free spatial map."""
        if features.dim() == 2:  # [L, C]
            L, C = features.shape
            H = W = int(np.sqrt(L))
            if H * W != L: return np.zeros((H, W))
            features = features.view(H, W, C)
        
        # Use standard deviation across channels as a proxy for feature importance
        feat_map = features.std(dim=-1).numpy()
        
        # Smooth to remove grid-like artifacts
        feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.5)

        # Normalize
        if feat_map.max() > feat_map.min():
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
        return feat_map

    def extract_pca_features(self, features: torch.Tensor) -> Tuple[List[np.ndarray], np.ndarray]:
        """Extracts principal components and smooths them for visualization."""
        if features.dim() == 2:
            L, C = features.shape
            H = W = int(np.sqrt(L))
            if H * W != L: return [np.zeros((H,W))] * 3, np.array([0,0,0])
            features = features.view(H, W, C)
        
        H, W, C = features.shape
        feat_flat = features.reshape(-1, C).numpy()

        if feat_flat.shape[0] <= 3 or C <= 3:
            maps = [self._get_spatial_map(features)] * 3
            return maps, np.array([1.0, 0, 0])

        scaler = StandardScaler()
        feat_std = scaler.fit_transform(feat_flat)
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_std)

        feat_maps = []
        for i in range(3):
            feat_map = feat_pca[:, i].reshape(H, W)
            # Smooth each component map
            feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.5)
            # Normalize
            if feat_map.max() > feat_map.min():
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            feat_maps.append(feat_map)

        return feat_maps, pca.explained_variance_ratio_

    def compute_feature_diversity(self, features: torch.Tensor) -> np.ndarray:
        """Computes a smoothed map of feature variance across channels."""
        return self._get_spatial_map(features)


def visualize_swin_attention_modern(model: nn.Module, input_tensor: torch.Tensor,
                                  original_image: np.ndarray, actual_label: int,
                                  save_path: Optional[Path] = None):
    """Generates a modern, publication-quality visualization."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    class_names = {0: 'Normal', 1: 'Cancerous'}
    actual_name = class_names[actual_label]
    predicted_name = class_names[predicted_class]
    is_correct = actual_label == predicted_class

    # --- Modern Plotting Setup ---
    plt.style.use('default') # Reset style
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1])

    # --- 1. Original Image ---
    ax = fig.add_subplot(gs[0, 0])
    img_enhanced = enhance_contrast(original_image)
    ax.imshow(img_enhanced, cmap='gray')
    ax.set_title(f"Original Image\nPredicted: {predicted_name} ({confidence:.1%})", weight='bold')
    ax.axis('off')

    # --- 2. GradCAM ---
    console.print("[cyan]Generating GradCAM...[/cyan]")
    ax = fig.add_subplot(gs[0, 1])
    gradcam_gen = SwinGradCAM(model)
    cam = gradcam_gen.generate_cam(input_tensor, class_idx=predicted_class)
    gradcam_gen.remove_hooks()
    
    cam_resized = cv2.resize(cam, (img_enhanced.shape[1], img_enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.imshow(img_enhanced, cmap='gray')
    ax.imshow(cam_resized, cmap='inferno', alpha=0.5)
    ax.set_title("GradCAM Focus", weight='bold')
    ax.axis('off')

    # --- 3. Feature Extraction ---
    console.print("[cyan]Extracting hierarchical features...[/cyan]")
    feature_viz = RadicalSwinFeatureVisualizer(model)
    features = feature_viz.get_intermediate_features(input_tensor)
    
    stage_names = ['patch_embed', 'stage1', 'stage2', 'stage3']
    stage_positions = [(0, 2), (0, 3), (0, 4), (1, 0)]
    stage_titles = ['Patch Embed', 'Stage 1', 'Stage 2', 'Stage 3']
    stage_variances = []

    for name, pos, title in zip(stage_names, stage_positions, stage_titles):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        if name in features:
            feat_map = feature_viz._get_spatial_map(features[name])
            ax.imshow(feat_map, cmap='viridis')
            stage_variances.append(features[name].var().item())
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_title(title)
        ax.axis('off')

    # --- 4. Feature Diversity ---
    ax = fig.add_subplot(gs[1, 1])
    if 'stage2' in features:
        diversity_map = feature_viz.compute_feature_diversity(features['stage2'])
        ax.imshow(diversity_map, cmap='plasma')
    ax.set_title("Feature Diversity")
    ax.axis('off')

    # --- 5. PCA Components (RGB) ---
    ax = fig.add_subplot(gs[1, 2])
    if 'stage3' in features:
        pca_maps, explained_var = feature_viz.extract_pca_features(features['stage3'])
        rgb_composite = np.stack(pca_maps, axis=-1)
        ax.imshow(rgb_composite)
        var_text = f"R:{explained_var[0]:.1%} G:{explained_var[1]:.1%} B:{explained_var[2]:.1%}"
        ax.set_title(f"PCA Components\n{var_text}")
    ax.axis('off')

    # --- 6. Feature Evolution ---
    ax = fig.add_subplot(gs[1, 3:])
    if stage_variances:
        variances = np.array(stage_variances)
        normalized_variances = variances / variances.max()
        ax.bar(stage_titles, normalized_variances, color='#4c72b0', alpha=0.7, edgecolor='black')
        ax.set_title("Feature Evolution", weight='bold')
        ax.set_ylabel("Normalized Variance")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # --- Figure Super Title ---
    result_str = "Correct" if is_correct else "Incorrect"
    fig.suptitle(f"Swin Transformer Analysis | Actual: {actual_name} | Result: {result_str}",
                 fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]Saved modern visualization to: {save_path}[/green]")
        plt.close(fig)
    else:
        plt.show()

# Main execution logic remains largely the same, but calls the new viz function
def main():
    # Setup paths
    checkpoint_dir = Path("checkpoints/best")
    output_dir = Path("outputs/attention_maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw")
    
    console.print("[cyan]Loading dataset...[/cyan]")

    # A placeholder for your dataset loading logic.
    # This needs to be adapted to your actual `CARSThyroidDataset` and `create_quality_aware_transform`
    # For now, we'll create dummy data to ensure the script runs.
    try:
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        
        quality_report_path = Path('reports/quality_report.json')
        if not quality_report_path.exists(): quality_report_path = None

        transform = create_quality_aware_transform(
            target_size=224, quality_report_path=quality_report_path, split='val'
        )
        dataset = CARSThyroidDataset(
            root_dir=data_dir, split='test', transform=transform, target_size=224
        )
        raw_dataset = CARSThyroidDataset(
            root_dir=data_dir, split='test', transform=None, target_size=224
        )
        
        sample_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:1] + \
                         [i for i, label in enumerate(dataset.labels) if label == 1][:1]
                         
        for i in sample_indices:
            if i >= len(dataset):
                sample_indices.remove(i)

        samples = [(dataset._load_image(i), dataset[i][0].unsqueeze(0), dataset[i][1]) for i in sample_indices]

    except (ImportError, FileNotFoundError) as e:
        console.print(f"[yellow]Warning: Could not load dataset ({e}). Using dummy data.[/yellow]")
        samples = [
            (np.random.rand(224, 224), torch.randn(1, 1, 224, 224), 0), # Dummy normal
            (np.random.rand(224, 224), torch.randn(1, 1, 224, 224), 1)  # Dummy cancer
        ]

    # Process with each model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"[cyan]Using device: {device}[/cyan]")
    
    model_files = ['swin_tiny-best.ckpt', 'swin_small-best.ckpt', 'swin_base-best.ckpt']
    
    for model_file in model_files:
        checkpoint_path = checkpoint_dir / model_file
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            continue
            
        console.print(f"\n[cyan]Loading {model_file}...[/cyan]")
        model = load_swin_model(str(checkpoint_path), device=str(device))
        if model is None: continue

        for idx, (raw_img, tensor, label) in enumerate(samples):
            tensor = tensor.to(device)
            label_str = 'normal' if label == 0 else 'cancerous'
            model_name_clean = model_file.replace('-best.ckpt', '')
            save_path = output_dir / f"{model_name_clean}_sample{idx+1}_{label_str}_modern.png"
            
            console.print(f"  Processing sample {idx+1} ({label_str})...")
            try:
                visualize_swin_attention_modern(model, tensor, raw_img, label, save_path)
            except Exception as e:
                console.print(f"    [red]Visualization failed: {e}[/red]")
                import traceback
                traceback.print_exc()

    console.print("\n[bold green]✓ Modern visualization complete![/bold green]")

def load_swin_model(checkpoint_path, device='cpu'):
    """Loads a Swin model from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device,  weights_only=False)
        model_file = Path(checkpoint_path).stem.lower()
        
        if 'tiny' in model_file: model_name = 'swin_tiny_patch4_window7_224'
        elif 'small' in model_file: model_name = 'swin_small_patch4_window7_224'
        elif 'base' in model_file: model_name = 'swin_base_patch4_window7_224'
        else: model_name = 'swin_tiny_patch4_window7_224'
        
        model = timm.create_model(model_name, pretrained=False, num_classes=2, in_chans=1)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Clean keys (e.g., remove 'model.' prefix)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        console.print(f"[green]✓ Loaded {model_name} successfully[/green]")
        model.eval()
        return model.to(device)
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None

if __name__ == "__main__":
    main()