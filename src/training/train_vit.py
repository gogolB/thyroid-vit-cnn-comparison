"""
PyTorch Lightning module for Vision Transformer training on thyroid classification.
Supports ViT, DeiT, and Swin Transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from typing import Optional, Dict, Any
from omegaconf import DictConfig, OmegaConf
import wandb

from src.models.vit import get_vit_model

from src.training.lightning_modules import ThyroidViTModule

# Content removed as per refactoring task.
# ThyroidViTModule is now imported from src.training.lightning_modules