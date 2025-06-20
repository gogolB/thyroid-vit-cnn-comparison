"""
Vision Transformer specific data transformations
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, List, Optional
import random


class RandAugment(nn.Module):
    """
    RandAugment: Randomly select and apply n augmentations with magnitude m
    
    Args:
        n: Number of augmentations to apply
        m: Magnitude of augmentations (0-30, typically 9)
        grayscale: Whether input images are grayscale
    """
    
    def __init__(self, n: int = 2, m: int = 9, grayscale: bool = True):
        super().__init__()
        self.n = n
        self.m = m
        self.grayscale = grayscale
        
        # Define augmentation space
        self.augmentations = [
            ("AutoContrast", self._auto_contrast),
            ("Brightness", self._brightness),
            ("Contrast", self._contrast),
            ("Equalize", self._equalize),
            ("Posterize", self._posterize),
            ("Rotate", self._rotate),
            ("Sharpness", self._sharpness),
            ("ShearX", self._shear_x),
            ("ShearY", self._shear_y),
            ("Solarize", self._solarize),
            ("TranslateX", self._translate_x),
            ("TranslateY", self._translate_y),
        ]
        
        # Skip color augmentations for grayscale
        if not grayscale:
            self.augmentations.extend([
                ("Color", self._color),
            ])
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply RandAugment to image"""
        # Randomly select n augmentations
        ops = random.choices(self.augmentations, k=self.n)
        
        # Apply each augmentation
        for name, func in ops:
            img = func(img, self.m)
        
        return img
    
    def _get_magnitude(self, m: int, min_val: float, max_val: float) -> float:
        """Convert magnitude to actual parameter value"""
        # Linear interpolation based on magnitude
        return min_val + (max_val - min_val) * m / 30.0
    
    def _auto_contrast(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Apply autocontrast (magnitude independent)"""
        if img.dtype == torch.float32:
            img_uint8 = (img * 255).to(torch.uint8)
            img_uint8 = TF.autocontrast(img_uint8)
            return img_uint8.float() / 255.0
        return TF.autocontrast(img)
    
    def _brightness(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust brightness"""
        factor = self._get_magnitude(m, 0.05, 1.95)
        return TF.adjust_brightness(img, factor)
    
    def _contrast(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust contrast"""
        factor = self._get_magnitude(m, 0.05, 1.95)
        return TF.adjust_contrast(img, factor)
    
    def _color(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust color saturation (only for RGB)"""
        factor = self._get_magnitude(m, 0.05, 1.95)
        return TF.adjust_saturation(img, factor)
    
    def _equalize(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Apply histogram equalization (magnitude independent)"""
        if img.dtype == torch.float32:
            img_uint8 = (img * 255).to(torch.uint8)
            img_uint8 = TF.equalize(img_uint8)
            return img_uint8.float() / 255.0
        return TF.equalize(img)
    
    def _posterize(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Reduce number of bits per channel"""
        bits = int(self._get_magnitude(m, 8, 4))
        if img.dtype == torch.float32:
            img_uint8 = (img * 255).to(torch.uint8)
            img_uint8 = TF.posterize(img_uint8, bits)
            return img_uint8.float() / 255.0
        return TF.posterize(img, bits)
    
    def _rotate(self, img, m: int):
        """Rotate image"""
        angle = self._get_magnitude(m, -30, 30)
        
        # Handle fill value based on image type
        if hasattr(img, 'shape'):
            fill_value = 1.0 if img.dtype == torch.float32 else 255
        else:
            fill_value = 255  # PIL default
        
        return TF.rotate(img, angle, fill=fill_value)
    
    def _sharpness(self, img: torch.Tensor, m: int) -> torch.Tensor:
        """Adjust sharpness"""
        factor = self._get_magnitude(m, 0.05, 1.95)
        return TF.adjust_sharpness(img, factor)
    
    def _shear_x(self, img, m: int):
        """Shear along X axis"""
        shear = self._get_magnitude(m, -0.3, 0.3)
        
        # Handle fill value
        if hasattr(img, 'shape'):
            fill_value = 1.0 if img.dtype == torch.float32 else 255
        else:
            fill_value = 255
        
        return TF.affine(img, angle=0, translate=(0, 0), scale=1.0, 
                        shear=(np.degrees(shear), 0), fill=fill_value)
    
    def _shear_y(self, img, m: int):
        """Shear along Y axis"""
        shear = self._get_magnitude(m, -0.3, 0.3)
        
        # Handle fill value
        if hasattr(img, 'shape'):
            fill_value = 1.0 if img.dtype == torch.float32 else 255
        else:
            fill_value = 255
            
        return TF.affine(img, angle=0, translate=(0, 0), scale=1.0,
                        shear=(0, np.degrees(shear)), fill=fill_value)
    
    def _solarize(self, img, m: int):
        """Solarize image (invert pixels above threshold)"""
        threshold = self._get_magnitude(m, 256, 0) / 256.0
        
        # Handle based on image type
        if hasattr(img, 'shape'):
            # Tensor - threshold in [0, 1]
            return TF.solarize(img, threshold)
        else:
            # PIL - threshold in [0, 255]
            return TF.solarize(img, int(threshold * 255))
    
    def _translate_x(self, img, m: int):
        """Translate along X axis"""
        translation = self._get_magnitude(m, -0.45, 0.45)
        
        # Handle both PIL and tensor
        if hasattr(img, 'shape'):
            # Tensor
            pixels = int(translation * img.shape[-1])
            fill_value = 1.0 if img.dtype == torch.float32 else 255
        else:
            # PIL Image
            pixels = int(translation * img.size[0])  # PIL uses .size not .shape
            fill_value = 255  # PIL images use 0-255
        
        return TF.affine(img, angle=0, translate=(pixels, 0), scale=1.0,
                        shear=(0, 0), fill=fill_value)
    
    def _translate_y(self, img, m: int):
        """Translate along Y axis"""
        translation = self._get_magnitude(m, -0.45, 0.45)
        
        # Handle both PIL and tensor
        if hasattr(img, 'shape'):
            # Tensor
            pixels = int(translation * img.shape[-2])
            fill_value = 1.0 if img.dtype == torch.float32 else 255
        else:
            # PIL Image
            pixels = int(translation * img.size[1])  # PIL uses .size not .shape
            fill_value = 255
        
        return TF.affine(img, angle=0, translate=(0, pixels), scale=1.0,
                        shear=(0, 0), fill=fill_value)


class QualityAwarePatchAugment(nn.Module):
    """
    Apply augmentation based on patch quality scores.
    Simplified version that works directly with tensors.
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        quality_threshold: float = 0.7,
        strong_aug_prob: float = 0.8,
        patch_drop_prob: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.quality_threshold = quality_threshold
        self.strong_aug_prob = strong_aug_prob
        self.patch_drop_prob = patch_drop_prob
    
    def compute_patch_quality(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute quality score for each patch.
        Returns quality map of shape (H//patch_size, W//patch_size)
        """
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # For grayscale medical images, use local statistics
        B, C, H, W = img.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        
        # Reshape into patches
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.reshape(B, C, pH, pW, -1)
        
        # Compute quality metrics per patch
        # 1. Contrast (std dev)
        contrast = patches.std(dim=-1)
        
        # 2. Mean intensity (avoid too dark/bright)
        mean_intensity = patches.mean(dim=-1)
        intensity_quality = 1 - 2 * torch.abs(mean_intensity - 0.5)
        
        # 3. Edge density (gradient magnitude)
        # Simplified: use local variance as proxy
        local_var = patches.var(dim=-1)
        
        # Combine metrics
        quality = (contrast + intensity_quality + local_var) / 3
        quality = quality.squeeze(1)  # Remove channel dim for grayscale
        
        return quality
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply quality-aware augmentation"""
        # Compute patch quality
        quality_map = self.compute_patch_quality(img)
        
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        # Create augmentation mask
        B, C, H, W = img.shape
        pH = H // self.patch_size
        pW = W // self.patch_size
        
        # Process each patch based on quality
        augmented = img.clone()
        
        for i in range(pH):
            for j in range(pW):
                # Get patch coordinates
                y1 = i * self.patch_size
                y2 = (i + 1) * self.patch_size
                x1 = j * self.patch_size
                x2 = (j + 1) * self.patch_size
                
                patch_quality = quality_map[:, i, j].mean().item()
                
                # Apply augmentation based on quality
                if patch_quality < self.quality_threshold:
                    # Low quality patch - apply strong augmentation
                    if random.random() < self.strong_aug_prob:
                        patch = augmented[:, :, y1:y2, x1:x2]
                        
                        # Apply simple tensor-based augmentations
                        aug_type = random.choice(['noise', 'blur', 'brightness', 'contrast'])
                        
                        if aug_type == 'noise':
                            # Add Gaussian noise
                            noise = torch.randn_like(patch) * 0.1
                            patch = torch.clamp(patch + noise, 0, 1)
                        elif aug_type == 'blur':
                            # Simple blur using average pooling
                            if patch.shape[-1] > 2:
                                patch = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(patch)
                        elif aug_type == 'brightness':
                            # Adjust brightness
                            factor = random.uniform(0.7, 1.3)
                            patch = torch.clamp(patch * factor, 0, 1)
                        elif aug_type == 'contrast':
                            # Adjust contrast
                            factor = random.uniform(0.7, 1.3)
                            mean = patch.mean()
                            patch = torch.clamp((patch - mean) * factor + mean, 0, 1)
                        
                        augmented[:, :, y1:y2, x1:x2] = patch
                    
                    # Optionally drop very low quality patches
                    if patch_quality < 0.3 and random.random() < self.patch_drop_prob:
                        augmented[:, :, y1:y2, x1:x2] = 0  # Black patch
                else:
                    # High quality patch - light augmentation
                    if random.random() < 0.3:  # 30% chance
                        patch = augmented[:, :, y1:y2, x1:x2]
                        
                        # Apply light augmentation
                        aug_type = random.choice(['slight_noise', 'slight_brightness'])
                        
                        if aug_type == 'slight_noise':
                            noise = torch.randn_like(patch) * 0.05
                            patch = torch.clamp(patch + noise, 0, 1)
                        elif aug_type == 'slight_brightness':
                            factor = random.uniform(0.9, 1.1)
                            patch = torch.clamp(patch * factor, 0, 1)
                        
                        augmented[:, :, y1:y2, x1:x2] = patch
        
        return augmented


def create_vit_transform(
    img_size: int = 256,
    is_training: bool = True,
    pretrained: bool = False,
    pretrained_type: str = 'imagenet',
    input_size_override: Optional[int] = None,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    use_quality_aware: bool = True,
    patch_size: int = 16
) -> T.Compose:
    """
    Enhanced transformation pipeline for Vision Transformers.
    
    Order is critical:
    1. PIL Image transforms (Resize, RandomFlip, RandAugment)
    2. ToTensor() 
    3. Tensor transforms (QualityAware, Normalize)
    """
    # Handle input size
    if input_size_override:
        img_size = input_size_override
    elif pretrained and pretrained_type == 'imagenet':
        img_size = 224  # Standard ImageNet size
    
    transforms = []
    
    # ========== PIL Image Transforms ==========
    # Resize first (works on PIL)
    transforms.append(T.Resize((img_size, img_size), antialias=True))
    
    if is_training:
        # Basic augmentations (work on PIL)
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        
        # RandAugment (works on PIL)
        if randaugment_n > 0:
            transforms.append(RandAugment(n=randaugment_n, m=randaugment_m, grayscale=True))
    
    # ========== Convert to Tensor ==========
    transforms.append(T.ToTensor())
    
    # ========== Tensor Transforms ==========
    # Quality-aware MUST come after ToTensor
    if is_training and use_quality_aware:
        transforms.append(QualityAwarePatchAugment(patch_size=patch_size))
    
    # Handle pretrained model requirements
    if pretrained and pretrained_type in ['imagenet', 'imagenet21k']:
        # Convert grayscale to RGB by repeating channels
        transforms.append(T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x))
        
        # Use ImageNet normalization
        transforms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    else:
        # Standard grayscale normalization
        transforms.append(T.Normalize(mean=[0.5], std=[0.5]))
    
    return T.Compose(transforms)

class MixUp:
    """
    MixUp augmentation for Vision Transformers
    """
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
        
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        labels_a, labels_b = labels, labels[index]
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """
    CutMix augmentation for Vision Transformers
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.shape[0]
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size)
        
        # Get random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape, lam)
        
        # Apply CutMix
        # Corrected indexing: bby1:bby2 for height (dim 2), bbx1:bbx2 for width (dim 3)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-1] * images.shape[-2]))
        
        labels_a, labels_b = labels, labels[index]
        return images, labels_a, labels_b, lam
    
    def _rand_bbox(self, shape, lam):
        H, W = shape[2], shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
