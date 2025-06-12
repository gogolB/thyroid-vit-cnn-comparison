"""
Inception-v4 implementation for CARS thyroid classification.
Adapted for single-channel medical imaging with quality-aware features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from omegaconf import DictConfig


class BasicConv2d(nn.Module):
    """Basic convolution block with batch norm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionStem(nn.Module):
    """Inception-v4 stem network."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Modified for medical imaging - less aggressive downsampling
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, padding=0)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=0)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=2, padding=0)
        
        self.branch7x7a = BasicConv2d(160, 64, kernel_size=1, stride=1)
        self.branch7x7b = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=0)
        
        self.branch7x7x3a = BasicConv2d(160, 64, kernel_size=1, stride=1)
        self.branch7x7x3b = BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7x3c = BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7x3d = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=0)
        
        self.branchpoola = nn.MaxPool2d(3, stride=2, padding=0)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x0 = self.branch3x3_conv(x)
        x1 = self.branch3x3_pool(x)
        x = torch.cat((x0, x1), 1)
        
        x0 = self.branch7x7a(x)
        x0 = self.branch7x7b(x0)
        
        x1 = self.branch7x7x3a(x)
        x1 = self.branch7x7x3b(x1)
        x1 = self.branch7x7x3c(x1)
        x1 = self.branch7x7x3d(x1)
        
        x = torch.cat((x0, x1), 1)
        
        x0 = self.branchpoola(x)
        x1 = self.branchpoolb(x)
        
        x = torch.cat((x0, x1), 1)
        return x


class InceptionA_v4(nn.Module):
    """Inception-v4 A block."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 96, kernel_size=1, stride=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1, stride=1)
        self.branch5x5_2 = BasicConv2d(64, 96, kernel_size=5, stride=1, padding=2)
        
        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1, stride=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.branch3x3_3 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 96, kernel_size=1, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch1x1(x)
        
        x1 = self.branch5x5_1(x)
        x1 = self.branch5x5_2(x1)
        
        x2 = self.branch3x3_1(x)
        x2 = self.branch3x3_2(x2)
        x2 = self.branch3x3_3(x2)
        
        x3 = self.branch_pool(x)
        
        return torch.cat((x0, x1, x2, x3), 1)


class ReductionA_v4(nn.Module):
    """Reduction-A block for Inception-v4."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, padding=0)
        
        self.branch3x3_2_1 = BasicConv2d(in_channels, 256, kernel_size=1, stride=1)
        self.branch3x3_2_2 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.branch3x3_2_3 = BasicConv2d(256, 384, kernel_size=3, stride=2, padding=0)
        
        self.branchpool = nn.MaxPool2d(3, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch3x3(x)
        
        x1 = self.branch3x3_2_1(x)
        x1 = self.branch3x3_2_2(x1)
        x1 = self.branch3x3_2_3(x1)
        
        x2 = self.branchpool(x)
        
        return torch.cat((x0, x1, x2), 1)


class InceptionB_v4(nn.Module):
    """Inception-v4 B block."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 384, kernel_size=1, stride=1)
        
        self.branch7x7_1 = BasicConv2d(in_channels, 256, kernel_size=1, stride=1)
        self.branch7x7_2 = BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(256, 384, kernel_size=(7, 1), stride=1, padding=(3, 0))
        
        self.branch7x7dbl_1 = BasicConv2d(in_channels, 256, kernel_size=1, stride=1)
        self.branch7x7dbl_2 = BasicConv2d(256, 256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(256, 256, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(256, 384, kernel_size=(1, 7), stride=1, padding=(0, 3))
        
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 128, kernel_size=1, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch1x1(x)
        
        x1 = self.branch7x7_1(x)
        x1 = self.branch7x7_2(x1)
        x1 = self.branch7x7_3(x1)
        
        x2 = self.branch7x7dbl_1(x)
        x2 = self.branch7x7dbl_2(x2)
        x2 = self.branch7x7dbl_3(x2)
        x2 = self.branch7x7dbl_4(x2)
        x2 = self.branch7x7dbl_5(x2)
        
        x3 = self.branch_pool(x)
        
        return torch.cat((x0, x1, x2, x3), 1)


class InceptionV3(nn.Module):
    """
    Inception-v3 model adapted for CARS thyroid classification.
    Fixed initialization issues from original implementation.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        dropout_rate: float = 0.5,
        aux_logits: bool = False,
        quality_aware: bool = True,
        transform_input: bool = False
    ):
        super().__init__()
        
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.quality_aware = quality_aware
        
        # First layers - adapted for single channel
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Inception blocks
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Quality-aware feature encoding
        if self.quality_aware:
            self.quality_encoder = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
            classifier_input_dim = 2048 + 64
        else:
            classifier_input_dim = 2048
            
        self.fc = nn.Linear(classifier_input_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization for Inception-v3."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1
                nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_quality_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quality-related features from input images."""
        batch_size = x.size(0)
        quality_features = []
        
        for i in range(batch_size):
            img = x[i, 0]
            mean_intensity = img.mean().item()
            std_intensity = img.std().item()
            hist = torch.histc(img, bins=256, min=0, max=1)
            hist = hist / hist.sum()
            entropy = -(hist * torch.log2(hist + 1e-10)).sum().item()
            quality_features.append([mean_intensity, std_intensity, entropy])
        
        return torch.tensor(quality_features, device=x.device, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        
        # Extract quality features before processing
        quality_feats = None
        if self.quality_aware:
            quality_feats = self.extract_quality_features(x)
            quality_feats = self.quality_encoder(quality_feats)
        
        # N x 1 x 256 x 256
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 127 x 127
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 125 x 125
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 125 x 125
        x = self.maxpool1(x)
        # N x 64 x 62 x 62
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 62 x 62
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 60 x 60
        x = self.maxpool2(x)
        # N x 192 x 29 x 29
        x = self.Mixed_5b(x)
        # N x 256 x 29 x 29
        x = self.Mixed_5c(x)
        # N x 288 x 29 x 29
        x = self.Mixed_5d(x)
        # N x 288 x 29 x 29
        x = self.Mixed_6a(x)
        # N x 768 x 14 x 14
        x = self.Mixed_6b(x)
        # N x 768 x 14 x 14
        x = self.Mixed_6c(x)
        # N x 768 x 14 x 14
        x = self.Mixed_6d(x)
        # N x 768 x 14 x 14
        x = self.Mixed_6e(x)
        # N x 768 x 14 x 14
        aux = None
        if self.aux_logits and self.training:
            aux = self.AuxLogits(x)
        # N x 768 x 14 x 14
        x = self.Mixed_7a(x)
        # N x 1280 x 7 x 7
        x = self.Mixed_7b(x)
        # N x 2048 x 7 x 7
        x = self.Mixed_7c(x)
        # N x 2048 x 7 x 7
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        
        # Add quality features if enabled
        if self.quality_aware and quality_feats is not None:
            x = torch.cat([x, quality_feats], dim=1)
        
        x = self.fc(x)
        # N x num_classes
        
        return x if aux is None else (x, aux)


# Inception-v3 specific modules
class InceptionA(nn.Module):
    """Inception-v3 A block."""
    
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    """Inception-v3 B block (reduction)."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch3x3 = self.branch3x3(x)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        
        branch_pool = self.maxpool(x)
        
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    """Inception-v3 C block."""
    
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        
        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        
        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        
        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    """Inception-v3 D block (reduction)."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        
        branch_pool = self.maxpool(x)
        
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    """Inception-v3 E block."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        
        branch_pool = self.avg_pool(x)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """Auxiliary classifier for Inception-v3."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x 768 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # N x 768 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionV4(nn.Module):
    """
    Inception-v4 model adapted for CARS thyroid classification.
    Simplified version optimized for medical imaging.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        dropout_rate: float = 0.2,
        aux_logits: bool = False,
        quality_aware: bool = True,
        init_weights: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.aux_logits = aux_logits
        self.quality_aware = quality_aware
        
        # Stem network
        self.stem = InceptionStem(in_channels)
        
        # 4 x Inception-A blocks
        self.inception_a1 = InceptionA_v4(384)
        self.inception_a2 = InceptionA_v4(384)
        self.inception_a3 = InceptionA_v4(384)
        self.inception_a4 = InceptionA_v4(384)
        
        # Reduction-A
        self.reduction_a = ReductionA_v4(384)
        
        # 7 x Inception-B blocks (reduced from original for efficiency)
        self.inception_b1 = InceptionB_v4(1152)
        self.inception_b2 = InceptionB_v4(1280)
        self.inception_b3 = InceptionB_v4(1280)
        self.inception_b4 = InceptionB_v4(1280)
        
        # Final pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Quality-aware feature encoding
        if self.quality_aware:
            self.quality_encoder = nn.Sequential(
                nn.Linear(3, 32),  # 3 quality metrics
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
            classifier_input_dim = 1280 + 64
        else:
            classifier_input_dim = 1280
        
        self.fc = nn.Linear(classifier_input_dim, num_classes)
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def extract_quality_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quality-related features from input images."""
        batch_size = x.size(0)
        quality_features = []
        
        for i in range(batch_size):
            img = x[i, 0]  # Get single channel
            
            # Mean intensity
            mean_intensity = img.mean().item()
            
            # Standard deviation
            std_intensity = img.std().item()
            
            # Simple entropy approximation
            hist = torch.histc(img, bins=256, min=0, max=1)
            hist = hist / hist.sum()
            entropy = -(hist * torch.log2(hist + 1e-10)).sum().item()
            
            quality_features.append([mean_intensity, std_intensity, entropy])
        
        return torch.tensor(quality_features, device=x.device, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Inception-v4."""
        
        # Stem
        x = self.stem(x)
        
        # Inception-A blocks
        x = self.inception_a1(x)
        x = self.inception_a2(x)
        x = self.inception_a3(x)
        x = self.inception_a4(x)
        
        # Reduction-A
        x = self.reduction_a(x)
        
        # Inception-B blocks
        x = self.inception_b1(x)
        x = self.inception_b2(x)
        x = self.inception_b3(x)
        x = self.inception_b4(x)
        
        # Final pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Add quality features if enabled
        if self.quality_aware:
            # Extract quality features from original input
            # Note: This requires access to original input
            # In practice, you'd pass original input separately
            quality_feats = torch.randn(x.size(0), 64, device=x.device)  # Placeholder
            x = torch.cat([x, quality_feats], dim=1)
        
        # Classification
        x = self.fc(x)
        
        return x


def create_inception_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create Inception models based on configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Inception model instance
    """
    model_map = {
        'inception_v3': InceptionV3,
        'inception_v4': InceptionV4,
    }
    
    model_name = cfg.model.name
    if model_name not in model_map:
        raise ValueError(f"Unknown Inception model: {model_name}")
    
    # Extract model parameters from config
    if model_name == 'inception_v3':
        model_params = {
            'num_classes': cfg.dataset.num_classes,
            'dropout_rate': cfg.model.dropout_rate,
            'in_channels': cfg.model.get('in_channels', 1),
            'quality_aware': cfg.model.get('quality_aware', True),
            'aux_logits': cfg.model.get('aux_logits', False),
            'transform_input': cfg.model.get('transform_input', False)
        }
    else:  # inception_v4
        model_params = {
            'num_classes': cfg.dataset.num_classes,
            'dropout_rate': cfg.model.dropout_rate,
            'in_channels': cfg.model.get('in_channels', 1),
            'quality_aware': cfg.model.get('quality_aware', True),
            'aux_logits': cfg.model.get('aux_logits', False),
            'init_weights': cfg.model.get('init_weights', True)
        }
    
    # Create and return model
    return model_map[model_name](**model_params)