"""
Quality-aware preprocessing module for CARS thyroid images.
Handles extreme dark, low contrast, and artifact issues.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import json

from rich.console import Console
from rich.progress import track

console = Console()


class QualityAwarePreprocessor(nn.Module):
    """
    Quality-aware preprocessing for CARS microscopy images.
    Applies different preprocessing strategies based on image quality issues.
    """
    
    def __init__(
        self,
        quality_report_path: Optional[Path] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        
        # Load quality report if provided
        self.quality_indices = self._load_quality_indices(quality_report_path)
        
        # Updated preprocessing parameters - more moderate
        self.params = {
            'extreme_dark': {
                'gamma': 0.8,  # Increased from 0.6 for less aggressive brightening
                'clahe_clip_limit': 2.0,  # Reduced from 3.0
                'clahe_grid_size': (16, 16)  # Increased for smoother effect
            },
            'low_contrast': {
                'clahe_clip_limit': 1.5,  # Reduced from 2.0
                'clahe_grid_size': (16, 16),  # Increased for smoother effect
                'contrast_factor': 1.3  # Reduced from 1.5
            },
            'artifacts': {
                'percentile_clip': 99.9,  # Increased from 99.5 - less aggressive
                'median_filter_size': 3,  # Reduced from 5
                'bilateral_d': 5,  # Reduced from 9
                'bilateral_sigma_color': 50,  # Reduced from 75
                'bilateral_sigma_space': 50  # Reduced from 75
            }
        }
    
    def _load_quality_indices(self, report_path: Optional[Path]) -> Dict[str, Dict[str, list]]:
        """Load quality issue indices from report."""
        if report_path is None or not report_path.exists():
            return {}
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        indices = {}
        for split in ['train', 'val', 'test']:
            if split in report['dataset_stats']:
                issues = report['dataset_stats'][split]['metrics']['quality_issues']
                indices[split] = {
                    'extreme_dark': set(issues['extreme_dark']),
                    'low_contrast': set(issues['low_contrast']),
                    'artifacts': set(issues['potential_artifacts'])
                }
        
        return indices
    
    def identify_quality_issues(self, img: np.ndarray) -> list:
        """
        Identify quality issues in an image based on statistics.
        Updated thresholds based on actual data analysis.
        """
        issues = []
        
        # Calculate statistics
        mean_val = np.mean(img)
        std_val = np.std(img)
        max_val = np.max(img)
        
        # Updated detection logic based on test results
        # Extreme dark: low mean (removed std requirement)
        if mean_val < 150:
            issues.append('extreme_dark')
        
        # Low contrast: low std (increased threshold)
        elif std_val < 80:  # Increased from 50
            issues.append('low_contrast')
        
        # Artifacts: high dynamic range ratio
        if max_val > 0 and mean_val > 0:
            dynamic_range_ratio = max_val / mean_val
            if dynamic_range_ratio > 30:  # Ratio-based detection
                issues.append('artifacts')
        
        return issues
    
    def apply_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to brighten dark images."""
        # Normalize to [0, 1]
        img_norm = img.astype(np.float32) / 65535.0
        
        # Apply gamma correction
        img_gamma = np.power(img_norm, gamma)
        
        # Convert back to uint16
        return (img_gamma * 65535).astype(np.uint16)
    
    def apply_clahe(self, img: np.ndarray, clip_limit: float, grid_size: Tuple[int, int]) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        # CLAHE works on uint8, so we need to convert
        img_8bit = (img / 256).astype(np.uint8)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        
        # Apply CLAHE
        img_clahe = clahe.apply(img_8bit)
        
        # Convert back to uint16 range
        return img_clahe.astype(np.uint16) * 256
    
    def suppress_artifacts(self, img: np.ndarray, percentile: float) -> np.ndarray:
        """Suppress bright artifacts using percentile clipping and filtering."""
        # Clip extreme values
        p_high = np.percentile(img, percentile)
        img_clipped = np.clip(img, 0, p_high)
        
        # Apply median filter to remove salt-and-pepper noise
        img_8bit = (img_clipped / 256).astype(np.uint8)
        img_median = cv2.medianBlur(img_8bit, self.params['artifacts']['median_filter_size'])
        
        # Optional: Apply bilateral filter for edge-preserving smoothing
        # Only apply if the image still has significant artifacts
        if np.max(img_median) > 250:  # Still has bright spots in 8-bit range
            img_bilateral = cv2.bilateralFilter(
                img_median,
                d=self.params['artifacts']['bilateral_d'],
                sigmaColor=self.params['artifacts']['bilateral_sigma_color'],
                sigmaSpace=self.params['artifacts']['bilateral_sigma_space']
            )
            return img_bilateral.astype(np.uint16) * 256
        else:
            return img_median.astype(np.uint16) * 256
    
    def validate_preprocessing(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """
        Validate preprocessing results and prevent excessive changes.
        Returns the validated processed image.
        """
        orig_mean = np.mean(original)
        proc_mean = np.mean(processed)
        
        # Prevent excessive brightening
        if proc_mean > orig_mean * 10:
            # Scale back the processing
            blend_factor = 0.5
            processed = (original * (1 - blend_factor) + processed * blend_factor).astype(np.uint16)
        
        # Prevent excessive darkening
        elif proc_mean < orig_mean * 0.1:
            # Scale back the processing
            blend_factor = 0.3
            processed = (original * (1 - blend_factor) + processed * blend_factor).astype(np.uint16)
        
        return processed
    
    def preprocess_image(self, img: np.ndarray, quality_issues: Optional[list] = None) -> np.ndarray:
        """Apply preprocessing based on quality issues."""
        if quality_issues is None:
            quality_issues = self.identify_quality_issues(img)
        
        # Apply preprocessing in order of severity
        processed = img.copy()
        
        # First handle artifacts (if present)
        if 'artifacts' in quality_issues:
            processed = self.suppress_artifacts(processed, self.params['artifacts']['percentile_clip'])
        
        # Then handle extreme dark
        if 'extreme_dark' in quality_issues:
            # Apply gamma correction first
            processed = self.apply_gamma_correction(processed, self.params['extreme_dark']['gamma'])
            # Then CLAHE
            processed = self.apply_clahe(
                processed,
                self.params['extreme_dark']['clahe_clip_limit'],
                self.params['extreme_dark']['clahe_grid_size']
            )
        
        # Finally handle low contrast (if not already handled by extreme_dark)
        elif 'low_contrast' in quality_issues:
            processed = self.apply_clahe(
                processed,
                self.params['low_contrast']['clahe_clip_limit'],
                self.params['low_contrast']['clahe_grid_size']
            )
        
        # Validate the preprocessing
        processed = self.validate_preprocessing(img, processed)
        
        return processed
    
    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply quality-aware preprocessing to a batch of images.
        
        Args:
            x: Input tensor of shape [B, C, H, W] or [C, H, W]
            indices: Optional indices of images in the dataset
            
        Returns:
            Preprocessed tensor
        """
        # Handle single image
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single_image = True
        else:
            single_image = False
        
        device = x.device
        x_np = x.cpu().numpy()
        
        # Process each image in batch
        processed = []
        for i in range(x_np.shape[0]):
            img = x_np[i, 0]  # Get single channel
            
            # Determine quality issues
            if indices is not None and hasattr(self, 'current_split'):
                # Use precomputed indices if available
                img_idx = indices[i].item()
                quality_issues = []
                for issue_type, issue_indices in self.quality_indices.get(self.current_split, {}).items():
                    if img_idx in issue_indices:
                        quality_issues.append(issue_type)
            else:
                # Identify issues dynamically
                quality_issues = self.identify_quality_issues(img)
            
            # Apply preprocessing
            img_processed = self.preprocess_image(img, quality_issues)
            processed.append(img_processed)
        
        # Convert back to tensor
        processed = np.stack(processed)[:, np.newaxis, :, :]  # Add channel dimension
        x_processed = torch.from_numpy(processed).float().to(device)
        
        if single_image:
            x_processed = x_processed.squeeze(0)
        
        return x_processed


class AdaptiveNormalization(nn.Module):
    """
    Adaptive normalization that handles quality issues.
    """
    
    def __init__(
        self,
        method: str = 'percentile',
        percentiles: Tuple[float, float] = (1, 99),
        quality_aware: bool = True
    ):
        super().__init__()
        self.method = method
        self.percentiles = percentiles
        self.quality_aware = quality_aware
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize images with quality awareness.
        
        Args:
            x: Input tensor [B, C, H, W] or [C, H, W]
            
        Returns:
            Normalized tensor in range [0, 1]
        """
        if self.method == 'percentile':
            # Calculate percentiles per image
            if x.dim() == 3:
                p_low = torch.quantile(x, self.percentiles[0] / 100)
                p_high = torch.quantile(x, self.percentiles[1] / 100)
                x = torch.clamp(x, p_low, p_high)
                x = (x - p_low) / (p_high - p_low + 1e-8)
            else:
                # Batch processing
                b, c, h, w = x.shape
                x_flat = x.view(b, c, -1)
                
                p_low = torch.quantile(x_flat, self.percentiles[0] / 100, dim=2, keepdim=True)
                p_high = torch.quantile(x_flat, self.percentiles[1] / 100, dim=2, keepdim=True)
                
                x_flat = torch.clamp(x_flat, p_low, p_high)
                x_flat = (x_flat - p_low) / (p_high - p_low + 1e-8)
                
                x = x_flat.view(b, c, h, w)
        
        elif self.method == 'minmax':
            if x.dim() == 3:
                x_min = x.min()
                x_max = x.max()
                x = (x - x_min) / (x_max - x_min + 1e-8)
            else:
                # Per-image normalization
                x_min = x.view(x.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
                x_max = x.view(x.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
                x = (x - x_min) / (x_max - x_min + 1e-8)
        
        return x


def create_quality_aware_transform(
    target_size: int = 256,
    quality_report_path: Optional[Path] = None,
    augmentation_level: str = 'medium',
    split: str = 'train'
):
    """
    Create a transform pipeline with quality-aware preprocessing.
    
    Args:
        target_size: Target image size
        quality_report_path: Path to quality report JSON
        augmentation_level: Level of augmentation to apply
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Transform pipeline
    """
    from src.data.transforms import get_training_transforms, get_validation_transforms
    
    # Create quality preprocessor
    quality_preprocessor = QualityAwarePreprocessor(quality_report_path)
    quality_preprocessor.current_split = split
    
    # Create adaptive normalization
    adaptive_norm = AdaptiveNormalization(
        method='percentile',
        percentiles=(1, 99),
        quality_aware=True
    )
    
    # Get standard augmentations based on split
    if split == 'train':
        standard_transforms = get_training_transforms(
            target_size=target_size,
            normalize=False,  # We'll normalize separately
            augmentation_level=augmentation_level
        )
    else:
        standard_transforms = get_validation_transforms(
            target_size=target_size,
            normalize=False
        )
    
    # Combine into pipeline
    transform = nn.Sequential(
        quality_preprocessor,
        adaptive_norm,
        standard_transforms
    )
    
    return transform


# Demo function
if __name__ == "__main__":
    # Test quality-aware preprocessing
    console.print("[bold cyan]Testing Quality-Aware Preprocessing[/bold cyan]")
    
    # Create dummy images with different quality issues
    
    # Extreme dark image
    dark_img = np.random.randint(120, 150, (512, 512), dtype=np.uint16)
    console.print(f"\nDark image - Mean: {np.mean(dark_img):.1f}, Std: {np.std(dark_img):.1f}")
    
    # Low contrast image
    low_contrast = np.full((512, 512), 140, dtype=np.uint16) + np.random.randint(-10, 10, (512, 512))
    console.print(f"Low contrast - Mean: {np.mean(low_contrast):.1f}, Std: {np.std(low_contrast):.1f}")
    
    # Image with artifacts
    artifact_img = np.random.randint(150, 300, (512, 512), dtype=np.uint16)
    artifact_img[100:110, 100:110] = 15000  # Add bright spot
    console.print(f"Artifact image - Mean: {np.mean(artifact_img):.1f}, Max: {np.max(artifact_img)}")
    
    # Test preprocessor
    preprocessor = QualityAwarePreprocessor()
    
    # Process each image
    for name, img in [("Dark", dark_img), ("Low Contrast", low_contrast), ("Artifacts", artifact_img)]:
        issues = preprocessor.identify_quality_issues(img)
        processed = preprocessor.preprocess_image(img)
        
        console.print(f"\n[yellow]{name}:[/yellow]")
        console.print(f"  Issues detected: {issues}")
        console.print(f"  Before - Mean: {np.mean(img):.1f}, Std: {np.std(img):.1f}, Max: {np.max(img)}")
        console.print(f"  After  - Mean: {np.mean(processed):.1f}, Std: {np.std(processed):.1f}, Max: {np.max(processed)}")