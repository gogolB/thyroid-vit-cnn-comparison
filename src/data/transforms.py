"""
Image transformations and augmentations for CARS microscopy images.
Includes both basic and microscopy-specific augmentations.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Union, Tuple
import cv2
import random
from scipy.ndimage import gaussian_filter, map_coordinates
import albumentations as A
from albumentations.pytorch import ToTensorV2

from rich.console import Console
from rich.table import Table

console = Console()


class MicroscopyNormalize(nn.Module):
    """
    Normalize microscopy images from uint16 range to standard range.
    
    Args:
        input_range: Input range, default (0, 65535) for uint16
        output_range: Output range, default (0, 1)
        clip_percentile: Optional percentile clipping for outliers
    """
    
    def __init__(
        self, 
        input_range: Tuple[float, float] = (0, 65535),
        output_range: Tuple[float, float] = (0, 1),
        clip_percentile: Optional[Tuple[float, float]] = None
    ):
        super().__init__()
        self.input_range = input_range
        self.output_range = output_range
        self.clip_percentile = clip_percentile
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply percentile clipping if specified
        if self.clip_percentile is not None:
            low, high = self.clip_percentile
            if x.dim() == 3:  # Single image [C, H, W]
                for c in range(x.shape[0]):
                    channel = x[c]
                    p_low = torch.quantile(channel, low / 100)
                    p_high = torch.quantile(channel, high / 100)
                    x[c] = torch.clamp(channel, p_low, p_high)
            else:  # Batch [B, C, H, W]
                for b in range(x.shape[0]):
                    for c in range(x.shape[1]):
                        channel = x[b, c]
                        p_low = torch.quantile(channel, low / 100)
                        p_high = torch.quantile(channel, high / 100)
                        x[b, c] = torch.clamp(channel, p_low, p_high)
        
        # Normalize to output range
        x = (x - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
        x = x * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        
        return x


class ElasticTransform(nn.Module):
    """
    Elastic deformation for microscopy images.
    Simulates tissue deformation during imaging.
    """
    
    def __init__(self, alpha: float = 50, sigma: float = 5, p: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        # Convert to numpy for processing
        device = x.device
        x_np = x.cpu().numpy()
        
        # Apply elastic transform
        if x_np.ndim == 3:  # [C, H, W]
            x_np = self._elastic_transform_2d(x_np[0])  # Process single channel
            x_np = x_np[np.newaxis, ...]  # Add channel dimension back
        else:  # [B, C, H, W]
            batch_size = x_np.shape[0]
            for i in range(batch_size):
                x_np[i, 0] = self._elastic_transform_2d(x_np[i, 0])
        
        return torch.from_numpy(x_np).to(device)
    
    def _elastic_transform_2d(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape
        
        # Random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation
        return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


class MicroscopyAugmentation(nn.Module):
    """
    Microscopy-specific augmentation including intensity variations,
    noise injection, and optical artifacts.
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        blur_sigma_range: Tuple[float, float] = (0, 1.0),
        p: float = 0.5
    ):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_sigma_range = blur_sigma_range
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        # Random brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(*self.brightness_range)
            x = x * brightness_factor
        
        # Random contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(*self.contrast_range)
            mean = x.mean(dim=(-2, -1), keepdim=True)
            x = (x - mean) * contrast_factor + mean
        
        # Add Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Random blur (simulating focus issues)
        if random.random() < 0.3:
            sigma = random.uniform(*self.blur_sigma_range)
            if sigma > 0:
                x = TF.gaussian_blur(x, kernel_size=5, sigma=sigma)
        
        # Clamp values
        x = torch.clamp(x, 0, 1)
        
        return x


class RandomPatchDrop(nn.Module):
    """
    Randomly drop patches from the image to simulate artifacts or missing data.
    """
    
    def __init__(self, patch_size: int = 32, max_patches: int = 5, p: float = 0.3):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        
        c, h, w = x.shape[-3:]
        num_patches = random.randint(1, self.max_patches)
        
        for _ in range(num_patches):
            # Random position
            y = random.randint(0, h - self.patch_size)
            x_pos = random.randint(0, w - self.patch_size)
            
            # Drop patch (set to mean value)
            mean_val = x[..., y:y+self.patch_size, x_pos:x_pos+self.patch_size].mean()
            x[..., y:y+self.patch_size, x_pos:x_pos+self.patch_size] = mean_val
        
        return x


def get_training_transforms(
    target_size: int = 256,
    normalize: bool = True,
    augmentation_level: str = 'medium'
) -> nn.Sequential:
    """
    Get training transforms with augmentations.
    
    Args:
        target_size: Target image size
        normalize: Whether to normalize images
        augmentation_level: 'light', 'medium', or 'heavy'
    
    Returns:
        Transform pipeline
    """
    
    transforms = []
    
    # Normalization (if not done in dataset)
    if normalize:
        transforms.append(MicroscopyNormalize(
            input_range=(0, 65535),
            output_range=(0, 1),
            clip_percentile=(1, 99) if augmentation_level != 'light' else None
        ))
    
    # Basic geometric transforms
    if augmentation_level != 'none':
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180 if augmentation_level == 'heavy' else 90),
        ])
    
    # Microscopy-specific augmentations
    if augmentation_level in ['medium', 'heavy']:
        transforms.extend([
            ElasticTransform(
                alpha=50 if augmentation_level == 'medium' else 80,
                sigma=5,
                p=0.3 if augmentation_level == 'medium' else 0.5
            ),
            MicroscopyAugmentation(
                brightness_range=(0.8, 1.2) if augmentation_level == 'medium' else (0.7, 1.3),
                contrast_range=(0.8, 1.2) if augmentation_level == 'medium' else (0.7, 1.3),
                noise_std=0.02 if augmentation_level == 'medium' else 0.03,
                p=0.5 if augmentation_level == 'medium' else 0.7
            ),
        ])
    
    # Heavy augmentations
    if augmentation_level == 'heavy':
        transforms.extend([
            RandomPatchDrop(patch_size=32, max_patches=5, p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        ])
    
    # Ensure correct size
    transforms.append(T.Resize((target_size, target_size)))
    
    return nn.Sequential(*transforms)


def get_validation_transforms(
    target_size: int = 256,
    normalize: bool = True
) -> nn.Sequential:
    """
    Get validation/test transforms (minimal augmentation).
    
    Args:
        target_size: Target image size
        normalize: Whether to normalize images
    
    Returns:
        Transform pipeline
    """
    
    transforms = []
    
    # Normalization
    if normalize:
        transforms.append(MicroscopyNormalize(
            input_range=(0, 65535),
            output_range=(0, 1),
            clip_percentile=(1, 99)
        ))
    
    # Resize
    transforms.append(T.Resize((target_size, target_size)))
    
    return nn.Sequential(*transforms)


def get_tta_transforms(
    target_size: int = 256,
    normalize: bool = True,
    n_augmentations: int = 5
) -> List[nn.Sequential]:
    """
    Get test-time augmentation transforms.
    
    Args:
        target_size: Target image size
        normalize: Whether to normalize images
        n_augmentations: Number of augmentations to generate
    
    Returns:
        List of transform pipelines
    """
    
    base_transform = MicroscopyNormalize(
        input_range=(0, 65535),
        output_range=(0, 1),
        clip_percentile=(1, 99)
    ) if normalize else nn.Identity()
    
    tta_transforms = [
        # Original
        nn.Sequential(base_transform, T.Resize((target_size, target_size))),
        
        # Horizontal flip
        nn.Sequential(
            base_transform,
            T.Resize((target_size, target_size)),
            T.RandomHorizontalFlip(p=1.0)
        ),
        
        # Vertical flip
        nn.Sequential(
            base_transform,
            T.Resize((target_size, target_size)),
            T.RandomVerticalFlip(p=1.0)
        ),
        
        # 90 degree rotation
        nn.Sequential(
            base_transform,
            T.Resize((target_size, target_size)),
            T.RandomRotation(degrees=(90, 90))
        ),
        
        # 270 degree rotation
        nn.Sequential(
            base_transform,
            T.Resize((target_size, target_size)),
            T.RandomRotation(degrees=(270, 270))
        ),
    ]
    
    return tta_transforms[:n_augmentations]


def print_augmentation_summary(augmentation_level: str = 'medium'):
    """Print a summary of augmentations for a given level."""
    
    table = Table(title=f"Augmentation Summary: {augmentation_level.capitalize()}", 
                  show_header=True, header_style="bold magenta")
    table.add_column("Transform", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="white")
    table.add_column("Probability", style="yellow")
    
    if augmentation_level == 'light':
        table.add_row("Horizontal Flip", "-", "0.5")
        table.add_row("Vertical Flip", "-", "0.5")
        table.add_row("Rotation", "±90°", "1.0")
        table.add_row("Normalization", "No clipping", "1.0")
        
    elif augmentation_level == 'medium':
        table.add_row("Horizontal Flip", "-", "0.5")
        table.add_row("Vertical Flip", "-", "0.5")
        table.add_row("Rotation", "±90°", "1.0")
        table.add_row("Elastic Transform", "α=50, σ=5", "0.3")
        table.add_row("Brightness", "0.8-1.2×", "0.25")
        table.add_row("Contrast", "0.8-1.2×", "0.25")
        table.add_row("Gaussian Noise", "σ=0.02", "0.15")
        table.add_row("Blur", "σ=0-1.0", "0.15")
        table.add_row("Normalization", "1-99% clip", "1.0")
        
    elif augmentation_level == 'heavy':
        table.add_row("Horizontal Flip", "-", "0.5")
        table.add_row("Vertical Flip", "-", "0.5")
        table.add_row("Rotation", "±180°", "1.0")
        table.add_row("Elastic Transform", "α=80, σ=5", "0.5")
        table.add_row("Brightness", "0.7-1.3×", "0.35")
        table.add_row("Contrast", "0.7-1.3×", "0.35")
        table.add_row("Gaussian Noise", "σ=0.03", "0.21")
        table.add_row("Blur", "σ=0-2.0", "0.21")
        table.add_row("Patch Drop", "32×32, max 5", "0.3")
        table.add_row("Gaussian Blur", "σ=0.1-2.0", "0.3")
        table.add_row("Normalization", "1-99% clip", "1.0")
    
    console.print(table)


# Demo function
def demo_transforms():
    """Demonstrate the transforms on a sample image."""
    
    console.print("[bold cyan]Transform Pipeline Demo[/bold cyan]\n")
    
    # Create a dummy image (512x512 uint16)
    dummy_img = torch.randint(0, 65535, (1, 512, 512), dtype=torch.float32)
    console.print(f"[cyan]Input image:[/cyan] shape={dummy_img.shape}, "
                  f"dtype={dummy_img.dtype}, range=[{dummy_img.min():.0f}, {dummy_img.max():.0f}]")
    
    # Test each augmentation level
    for level in ['light', 'medium', 'heavy']:
        console.print(f"\n[yellow]Testing {level} augmentation:[/yellow]")
        print_augmentation_summary(level)
        
        transform = get_training_transforms(
            target_size=256,
            normalize=True,
            augmentation_level=level
        )
        
        # Apply transform
        transformed = transform(dummy_img)
        console.print(f"[green]Output:[/green] shape={transformed.shape}, "
                      f"range=[{transformed.min():.3f}, {transformed.max():.3f}]")
    
    # Test TTA
    console.print("\n[yellow]Test-Time Augmentation:[/yellow]")
    tta_transforms = get_tta_transforms(target_size=256, normalize=True)
    console.print(f"[green]Generated {len(tta_transforms)} TTA transforms[/green]")


if __name__ == "__main__":
    demo_transforms()
