"""TSN — Temporal Segment Network.

Family: 2D-CNN + temporal pooling. Each frame is encoded independently by a 2D
ResNet backbone; per-frame features are averaged over time (consensus) before the
classifier. No temporal modelling inside the backbone — the simplest baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class TSN(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 8,
                 pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x)                 # [B*T, feat]
        feat = feat.reshape(B, T, -1).mean(dim=1)  # temporal average pooling
        return self.fc(self.dropout(feat))
