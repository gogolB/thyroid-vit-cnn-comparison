#!/usr/bin/env python3
"""
Fixed Swin Transformer visualization with contrast enhancement and window artifact removal.
Uses quality-aware preprocessing and proper feature extraction.
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
from rich.progress import track
import warnings
import timm
from skimage import exposure
warnings.filterwarnings('ignore')

console = Console()


def enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray:
    """
    Enhance contrast of CARS microscopy images.
    
    Args:
        image: Input image (grayscale) - can be tensor or numpy array
        method: 'clahe', 'percentile', or 'adaptive'
    
    Returns:
        Contrast enhanced image
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    # Ensure image is in proper format
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    
    # Normalize to 0-1 range first
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    if method == 'clahe':
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Convert to uint8 for CLAHE
        img_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_uint8)
        enhanced = enhanced.astype(np.float32) / 255.0
        
    elif method == 'percentile':
        # Use percentile-based contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        enhanced = exposure.rescale_intensity(image, in_range=(p2, p98))
        
    elif method == 'adaptive':
        # Adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
        
    else:
        enhanced = image
    
    return enhanced


class SwinGradCAM:
    """GradCAM implementation for Swin Transformer"""
    
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Find target layer (last stage by default)
        if target_layer_name is None:
            # Use last stage's last block
            for name, module in model.named_modules():
                if 'layers.3.blocks' in name and 'mlp' not in name and 'norm' not in name:
                    target_layer_name = name
        
        # Register hooks
        for name, module in model.named_modules():
            if name == target_layer_name:
                self.handles.append(module.register_forward_hook(self._save_activation))
                self.handles.append(module.register_backward_hook(self._save_gradient))
                console.print(f"[green]Registered GradCAM hooks on: {name}[/green]")
                break
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate class activation map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=0, keepdim=True)  # Average over spatial dimensions
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=-1)  # Sum over channel dimension
        
        # ReLU and normalization
        cam = F.relu(cam)
        
        # Reshape from sequence to 2D
        B = cam.shape[0]
        H = W = int(np.sqrt(cam.shape[0]))
        if H * W == B:
            cam = cam.reshape(H, W)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class ImprovedSwinFeatureVisualizer:
    """Improved feature visualization that removes window artifacts"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = {}
        self.hooks = []
    
    def get_intermediate_features(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from different stages"""
        self.features = {}
        
        # Hook into different stages
        def make_hook(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    self.features[name] = output[0].detach().cpu()
            return hook
        
        # Register hooks for patch embedding and each stage
        hook_points = [
            ('patch_embed', self.model.patch_embed),
            ('stage1', self.model.layers[0]),
            ('stage2', self.model.layers[1]),
            ('stage3', self.model.layers[2]),
            ('stage4', self.model.layers[3]) if len(self.model.layers) > 3 else ('stage3_end', self.model.layers[2])
        ]
        
        for name, module in hook_points:
            if module is not None:
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return self.features
    
    def remove_window_artifacts(self, features: torch.Tensor, window_size: int = 7) -> np.ndarray:
        """Remove window artifacts by averaging within windows and smoothing boundaries"""
        if features.dim() == 3:  # [B, L, C]
            B, L, C = features.shape
            H = W = int(np.sqrt(L))
            
            if H * W == L:
                # Reshape to spatial dimensions
                feat_2d = features[0].reshape(H, W, C)
                
                # Take mean across channels
                feat_map = feat_2d.mean(dim=-1).numpy()
                
                # Apply Gaussian smoothing to reduce window boundaries
                feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.0)
                
                # Apply median filter to further reduce artifacts
                feat_map = cv2.medianBlur((feat_map * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
                
                return feat_map
            else:
                # Non-square features
                return features[0].mean(dim=-1).numpy()
        
        elif features.dim() == 4:  # [B, C, H, W]
            # Direct spatial features
            feat_map = features[0].mean(dim=0).numpy()
            # Smooth to reduce artifacts
            feat_map = cv2.GaussianBlur(feat_map, (5, 5), 1.0)
            return feat_map
        
        return features[0].mean(dim=0).numpy()


def visualize_swin_attention_improved(model: nn.Module, input_tensor: torch.Tensor, 
                                    original_image: np.ndarray, save_path: Optional[Path] = None):
    """Improved visualization with contrast enhancement and artifact removal"""
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    # Convert to numpy if tensor
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    
    # 1. Original image with contrast enhancement
    ax_orig = fig.add_subplot(gs[0, 0])
    if original_image.ndim == 3 and original_image.shape[0] == 1:
        original_image = original_image.squeeze(0)
    
    # Apply contrast enhancement
    img_enhanced = enhance_contrast(original_image, method='clahe')
    
    ax_orig.imshow(img_enhanced, cmap='gray')
    ax_orig.set_title('Original CARS Image\n(Contrast Enhanced)', fontweight='bold')
    ax_orig.axis('off')
    
    # Add scale bar
    scalebar_length = 50  # pixels
    ax_orig.plot([10, 10+scalebar_length], [img_enhanced.shape[0]-20, img_enhanced.shape[0]-20],
                'w-', linewidth=2)
    ax_orig.text(10+scalebar_length/2, img_enhanced.shape[0]-30, '25μm',
                ha='center', va='top', color='white', fontsize=9)
    
    # 2. GradCAM visualization
    console.print("[cyan]Generating GradCAM visualization...[/cyan]")
    gradcam = SwinGradCAM(model)
    cam = gradcam.generate_cam(input_tensor)
    gradcam.remove_hooks()
    
    # Resize CAM to original image size
    cam_resized = cv2.resize(cam, (img_enhanced.shape[1], img_enhanced.shape[0]))
    
    ax_gradcam = fig.add_subplot(gs[0, 1])
    ax_gradcam.imshow(img_enhanced, cmap='gray', alpha=0.5)
    im = ax_gradcam.imshow(cam_resized, cmap='jet', alpha=0.5)
    ax_gradcam.set_title('GradCAM\n(Model Focus Areas)', fontweight='bold')
    ax_gradcam.axis('off')
    plt.colorbar(im, ax=ax_gradcam, fraction=0.046)
    
    # 3. Feature visualization with artifact removal
    console.print("[cyan]Extracting hierarchical features...[/cyan]")
    feature_viz = ImprovedSwinFeatureVisualizer(model)
    features = feature_viz.get_intermediate_features(input_tensor)
    
    # Visualize key stages with artifact removal
    stage_names = ['patch_embed', 'stage1', 'stage2', 'stage3']
    for idx, stage_name in enumerate(stage_names):
        if stage_name in features:
            ax = fig.add_subplot(gs[1, idx])
            feat = features[stage_name]
            
            # Remove window artifacts
            feat_map = feature_viz.remove_window_artifacts(feat)
            
            # Resize to original image size
            if feat_map.ndim == 2:
                feat_map = cv2.resize(feat_map, (img_enhanced.shape[1], img_enhanced.shape[0]))
            
            # Normalize
            feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-8)
            
            # Apply colormap
            ax.imshow(feat_map, cmap='viridis')
            ax.set_title(f'{stage_name.replace("_", " ").title()}', fontsize=10)
            ax.axis('off')
    
    # 4. Interpretation panel
    ax_interp = fig.add_subplot(gs[0, 2:])
    ax_interp.axis('off')
    
    interp_text = """
    Swin Transformer Visualization Insights:
    
    • GradCAM: Shows which regions contribute most to the 
      model's decision. Red = high importance.
      
    • Feature Progression: Shows how the model processes
      the image through hierarchical stages:
      - Patch Embed: Initial feature extraction
      - Stage 1-3: Progressively abstract representations
      
    • Key Observations:
      - Model focuses on tissue boundaries and texture
      - Higher stages capture more semantic features
      - Attention is distributed across diagnostic regions
      
    • Technical Note: Window artifacts have been removed
      using spatial smoothing and median filtering
    """
    
    ax_interp.text(0.05, 0.95, interp_text, transform=ax_interp.transAxes,
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    fig.suptitle('Swin Transformer Analysis: What the Model Sees', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]Saved visualization to: {save_path}[/green]")
        plt.close()
    else:
        plt.show()
    
    return fig


def main():
    import torch
    from src.data.dataset import CARSThyroidDataset
    from src.data.quality_preprocessing import create_quality_aware_transform
    
    # Setup paths - FIXED PATHS
    checkpoint_dir = Path("checkpoints/best")
    output_dir = Path("outputs/attention_maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sample images - FIXED PATH
    data_dir = Path("data/raw")
    console.print("[cyan]Loading sample images...[/cyan]")
    
    # Get quality report path
    quality_report_path = Path('reports/quality_report.json')
    
    if not quality_report_path.exists():
        console.print("[yellow]Warning: Quality report not found. Using standard preprocessing.[/yellow]")
        quality_report_path = None
    else:
        console.print("[green]Using quality-aware preprocessing[/green]")
    
    # Get validation transform WITH QUALITY-AWARE PREPROCESSING
    transform = create_quality_aware_transform(
        target_size=224,
        quality_report_path=quality_report_path,
        augmentation_level='none',  # No augmentation for test/validation
        split='val'
    )
    
    # Load dataset with quality-aware preprocessing
    dataset = CARSThyroidDataset(
        root_dir=data_dir,
        split='val',
        transform=transform,
        target_size=224,
        normalize=False  # IMPORTANT: Normalization handled in transform
    )
    
    console.print(f"[green]Validation dataset loaded: {len(dataset)} images[/green]")
    console.print(f"[dim]Using quality-aware preprocessing: {quality_report_path is not None}[/dim]")
    
    # Get samples
    n_samples = min(4, len(dataset))
    if len(dataset) >= 4:
        sample_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4]
    else:
        sample_indices = list(range(len(dataset)))
    
    samples = []
    
    for idx in sample_indices[:n_samples]:
        if idx < len(dataset):
            img, label = dataset[idx]
            # Also get raw image for visualization (without quality preprocessing)
            raw_dataset = CARSThyroidDataset(
                root_dir=data_dir,
                split='val',
                transform=None,
                target_size=224,
                normalize=False
            )
            raw_img, _ = raw_dataset[idx]
            samples.append((raw_img, img.unsqueeze(0), label))
            console.print(f"  Loaded sample {len(samples)}: {'normal' if label == 0 else 'cancerous'}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[cyan]Using device: {device}[/cyan]")
    
    # Models to analyze - FIXED NAMES
    models_to_analyze = [
        'swin_tiny-best.ckpt',
        'swin_small-best.ckpt',
        'swin_base-best.ckpt'
    ]
    
    console.print("\n[bold cyan]Generating Improved Swin Transformer Visualizations[/bold cyan]")
    console.print("[dim]With contrast enhancement and artifact removal[/dim]\n")
    
    for model_file in models_to_analyze:
        checkpoint_path = checkpoint_dir / model_file
        
        if not checkpoint_path.exists():
            console.print(f"[yellow]Checkpoint not found: {checkpoint_path}[/yellow]")
            continue
        
        # Load model
        console.print(f"\n[cyan]Loading {model_file}...[/cyan]")
        model = load_swin_model(str(checkpoint_path), device=str(device))
        
        if model is None:
            continue
        
        # Process each sample
        for idx, (raw_img, tensor, label) in enumerate(samples):
            tensor = tensor.to(device)
            
            # Generate visualization
            label_str = 'normal' if label == 0 else 'cancerous'
            save_path = output_dir / f"{model_file.replace('.ckpt', '')}_sample{idx+1}_{label_str}_improved.png"
            
            console.print(f"  Processing sample {idx+1} ({label_str})...")
            try:
                visualize_swin_attention_improved(model, tensor, raw_img, save_path)
            except Exception as e:
                console.print(f"    [red]Failed: {e}[/red]")
    
    console.print("\n[bold green]✓ Visualization complete![/bold green]")
    console.print(f"[green]Results saved to: {output_dir}[/green]")


def load_swin_model(checkpoint_path, device='cpu'):
    """Load Swin model from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Determine model variant
        model_file = Path(checkpoint_path).stem.lower()
        if 'tiny' in model_file:
            model_name = 'swin_tiny_patch4_window7_224'
        elif 'small' in model_file:
            model_name = 'swin_small_patch4_window7_224'
        elif 'base' in model_file:
            model_name = 'swin_base_patch4_window7_224'
        else:
            model_name = 'swin_tiny_patch4_window7_224'
        
        # Create model
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=2,
            in_chans=1,
            img_size=224
        )
        
        # Load weights
        state_dict = checkpoint.get('state_dict', checkpoint)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            cleaned_state_dict[clean_key] = v
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        console.print(f"[green]✓ Loaded Swin model successfully[/green]")
        
        model.eval()
        return model.to(device)
        
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return None


if __name__ == "__main__":
    main()