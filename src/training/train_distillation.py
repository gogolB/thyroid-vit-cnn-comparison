"""
PyTorch Lightning module for knowledge distillation training of Vision Transformers.
Supports distilling knowledge from CNN or ViT teachers into DeiT students.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score
from typing import Optional, Dict, Any, Tuple, Union, List
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path

from src.models.vit import get_vit_model
from src.models.vit.deit_models import DistillationLoss
from src.utils.models import TeacherModelLoader, EnsembleTeacher
from src.utils.training import get_device
from src.training.lightning_modules import ThyroidDistillationModule

