"""
Vision Transformer specific data transformations
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from typing import Tuple, Optional, List
import random


class RandAugment:
    """
    RandAugment for Vision Transformers
    Adapted for grayscale medical images
    """
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        self.augmentations = [
            ("rotation", self._rotation),
            ("translation", self._translation),
            ("shear", self._shear),
            ("brightness", self._brightness),
            ("contrast", self._contrast),
            ("sharpness", self._sharpness),
            ("gaussian_blur", self._gaussian_blur),
            ("elastic", self._elastic_transform),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augmentations, k=self.n)
        for op_name, op_func in ops:
            img = op_func(img, self.m)
        return img
    
    def _rotation(self, img, magnitude):
        angle = magnitude * 30 / 9  # Max 30 degrees
        return T.functional.rotate(img, angle)
    
    def _translation(self, img, magnitude):
        translate = magnitude * 0.3 / 9  # Max 30% translation
        return T.RandomAffine(degrees=0, translate=(translate, translate))(img)
    
    # ... (implement other augmentations)


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
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
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


def create_vit_transform(
    img_size: int = 256,
    is_training: bool = True,
    randaugment_n: int = 2,
    randaugment_m: int = 9,
    use_quality_aware: bool = True
) -> T.Compose:
    """
    Create transformation pipeline for Vision Transformers
    """
    if is_training:
        transforms = [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
        
        if randaugment_n > 0:
            transforms.append(RandAugment(n=randaugment_n, m=randaugment_m))
            
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # For grayscale
        ])
    else:
        transforms = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ]
    
    return T.Compose(transforms)

