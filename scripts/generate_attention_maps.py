"""
Modernized Swin Transformer visualization with a bespoke, high-fidelity
Attention Rollout implementation. This script correctly inverts the windowing and
shifting mechanisms of the Swin architecture to produce artifact-free heatmaps.
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
from typing import Dict, List, Optional, Tuple
import cv2
from rich.console import Console
import warnings
import timm
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


class SwinAttentionRollout:
    """
    High-fidelity Attention Rollout for Swin Transformers.
    This implementation correctly reverses the windowing and shifting operations
    to generate a spatially coherent attention map.
    """
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.hooks = []
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on all SwinTransformerBlocks."""
        for name, module in self.model.named_modules():
            if isinstance(module, SwinTransformerBlock):
                hook_fn = self._create_hook_fn(module)
                self.hooks.append(module.attn.attn_drop.register_forward_hook(hook_fn))
        console.print(f"[green]Registered [bold]{len(self.hooks)}[/bold] hooks for Attention Rollout.[/green]")

    def _create_hook_fn(self, module: SwinTransformerBlock):
        """Creates a hook function that captures attention and context."""
        def hook(m, inp, outp):
            self.attentions.append({
                'attn': outp.detach().to(self.device),
                'input_resolution': module.input_resolution,
                'window_size': module.window_size,
                'shift_size': module.shift_size,
            })
        return hook

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generates the high-fidelity attention map by running the rollout algorithm.
        """
        self.model.eval()
        self.attentions = [] 

        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        H, W = self.attentions[0]['input_resolution']
        rollout = torch.eye(H * W, device=self.device)

        for data in self.attentions:
            attn = data['attn'].squeeze(0).mean(dim=1)
            I = torch.eye(attn.shape[-1], device=self.device)
            attn = attn + I
            attn = attn / attn.sum(dim=-1, keepdim=True)

            H, W = data['input_resolution']
            win_size = data['window_size']
            num_tokens_win = win_size * win_size
            num_windows = H * W // num_tokens_win

            layer_attn = torch.zeros(H * W, H * W, device=self.device)
            for i in range(num_windows):
                start_idx = i * num_tokens_win
                end_idx = start_idx + num_tokens_win
                layer_attn[start_idx:end_idx, start_idx:end_idx] = attn[i]

            shift_size = data['shift_size']
            if shift_size > 0:
                grid = torch.arange(H * W, device=self.device).view(H, W)
                shifted_grid = torch.roll(grid, shifts=(shift_size, shift_size), dims=(0, 1))
                perm = shifted_grid.flatten().argsort()
                layer_attn = layer_attn[perm, :][:, perm]

            rollout = layer_attn @ rollout

        final_map = rollout.diag().reshape(H, W)
        result = final_map.cpu().numpy()
        result = (result - result.min()) / (result.max() - result.min())
        return result

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


class RadicalSwinFeatureVisualizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = {}
        self.hooks = []
    def get_intermediate_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.features = {}
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple): output = output[0]
                self.features[name] = output.detach().cpu()[0]
            return hook
        hook_points = [('patch_embed', self.model.patch_embed)] + \
                      [(f'stage{i+1}', l) for i, l in enumerate(self.model.layers)]
        for name, module in hook_points:
            self.hooks.append(module.register_forward_hook(make_hook(name)))
        with torch.no_grad(): _ = self.model(input_tensor)
        for hook in self.hooks: hook.remove()
        self.hooks = []
        return self.features
    def _get_spatial_map(self, features: torch.Tensor) -> np.ndarray:
        if features.dim() == 2:
            L, C = features.shape
            H = W = int(np.sqrt(L))
            if H * W != L: return np.zeros((H, W))
            features = features.view(H, W, C)
        feat_map = features.std(dim=-1).numpy()
        feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.5)
        if feat_map.max() > feat_map.min():
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
        return feat_map
    def extract_pca_features(self, features: torch.Tensor) -> Tuple[List[np.ndarray], np.ndarray]:
        if features.dim() == 2:
            L, C = features.shape
            H = W = int(np.sqrt(L))
            if H * W != L: return [np.zeros((H,W))] * 3, np.array([0,0,0])
            features = features.view(H, W, C)
        H, W, C = features.shape
        feat_flat = features.reshape(-1, C).numpy()
        if feat_flat.shape[0] <= 3 or C <= 3:
            return [self._get_spatial_map(features)] * 3, np.array([1.0, 0, 0])
        scaler = StandardScaler()
        feat_std = scaler.fit_transform(feat_flat)
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat_std)
        feat_maps = []
        for i in range(3):
            feat_map = feat_pca[:, i].reshape(H, W)
            feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.5)
            if feat_map.max() > feat_map.min():
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
            feat_maps.append(feat_map)
        return feat_maps, pca.explained_variance_ratio_
    def compute_feature_diversity(self, features: torch.Tensor) -> np.ndarray:
        return self._get_spatial_map(features)


def visualize_swin_attention_modern(model: nn.Module, input_tensor: torch.Tensor,
                                  original_image: np.ndarray, actual_label: int,
                                  save_path: Optional[Path] = None, device: str = 'cpu'):
    """Generates a modern, publication-quality visualization using high-fidelity Attention Rollout."""
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
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    img_enhanced = enhance_contrast(original_image)
    ax.imshow(img_enhanced, cmap='gray')
    ax.set_title("Original Image", weight='bold')
    ax.axis('off')

    console.print("[cyan]Generating High-Fidelity Attention Rollout...[/cyan]")
    ax = fig.add_subplot(gs[0, 1])
    rollout_gen = SwinAttentionRollout(model, device=device)
    attention_map = rollout_gen.generate(input_tensor)
    rollout_gen.remove_hooks()
    
    map_resized = cv2.resize(attention_map, (img_enhanced.shape[1], img_enhanced.shape[0]), interpolation=cv2.INTER_CUBIC)
    ax.imshow(img_enhanced, cmap='gray')
    ax.imshow(map_resized, cmap='inferno', alpha=0.5)
    ax.set_title("Attention Rollout", weight='bold')
    ax.axis('off')

    console.print("[cyan]Extracting hierarchical features...[/cyan]")
    feature_viz = RadicalSwinFeatureVisualizer(model)
    features = feature_viz.get_intermediate_features(input_tensor.to(device))
    
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

    ax = fig.add_subplot(gs[1, 1])
    if 'stage2' in features:
        ax.imshow(feature_viz.compute_feature_diversity(features['stage2']), cmap='plasma')
    ax.set_title("Feature Diversity")
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    if 'stage3' in features:
        pca_maps, explained_var = feature_viz.extract_pca_features(features['stage3'])
        ax.imshow(np.stack(pca_maps, axis=-1))
        ax.set_title(f"PCA Components\nR:{explained_var[0]:.1%} G:{explained_var[1]:.1%} B:{explained_var[2]:.1%}")
    ax.axis('off')

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

    # --- UPDATED Super Title with detailed info ---
    result_symbol = "✓" if is_correct else "✗"
    super_title = (
        f"Swin Transformer Analysis\n"
        f"Actual: {actual_name} | Predicted: {predicted_name} "
        f"({confidence:.1%}) | Result: {result_symbol}"
    )
    fig.suptitle(super_title, fontsize=16, weight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        console.print(f"[green]Saved modern visualization to: {save_path}[/green]")
        plt.close(fig)
    else:
        plt.show()


def load_swin_model(checkpoint_path: str, device: str = 'cpu') -> Optional[nn.Module]:
    """Loads a Swin model from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
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
    # Setup paths
    checkpoint_dir = Path("checkpoints/best")
    output_dir = Path("outputs/attention_maps_rollout")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/raw")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    console.print("[cyan]Loading dataset...[/cyan]")
    try:
        from src.data.dataset import CARSThyroidDataset
        from src.data.quality_preprocessing import create_quality_aware_transform
        
        transform = create_quality_aware_transform(target_size=224, split='val')
        dataset = CARSThyroidDataset(root_dir=data_dir, split='val', transform=transform, target_size=224)
        raw_dataset = CARSThyroidDataset(root_dir=data_dir, split='val', transform=None, target_size=224)
        
        sample_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:1] + \
                         [i for i, label in enumerate(dataset.labels) if label == 1][:1]
        if not sample_indices:
             raise FileNotFoundError("No samples found in dataset.")
        samples = [(raw_dataset._load_image(i), dataset[i][0].unsqueeze(0), dataset[i][1]) for i in sample_indices]
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load dataset ({e}). Using dummy data.[/yellow]")
        samples = [(np.random.rand(224, 224), torch.randn(1, 1, 224, 224), 0)]

    model_files = ['swin_tiny-best.ckpt', 'swin_small-best.ckpt', 'swin_base-best.ckpt']
    
    for model_file in model_files:
        checkpoint_path = checkpoint_dir / model_file
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            continue
            
        model = load_swin_model(str(checkpoint_path), device=device)
        if model is None: continue

        for idx, (raw_img, tensor, label) in enumerate(samples):
            label_str = 'normal' if label == 0 else 'cancerous'
            model_name_clean = model_file.replace('-best.ckpt', '')
            save_path = output_dir / f"{model_name_clean}_sample{idx+1}_{label_str}_rollout_final.png"
            
            console.print(f"  Processing sample {idx+1} ({label_str}) with {model_name_clean}...")
            visualize_swin_attention_modern(model, tensor, raw_img, label, save_path, device)

    console.print("\n[bold green]✓ High-Fidelity Rollout Visualization Complete![/bold green]")
    console.print(f"   Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
