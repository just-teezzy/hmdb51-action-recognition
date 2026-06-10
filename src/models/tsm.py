"""TSM — Temporal Shift Module.

Family: 2D-CNN with zero-FLOP temporal modelling. Same ResNet backbone as TSN, but
a fraction of channels is shifted forward/backward along the temporal axis at the
input of every residual block, letting 2D convolutions mix information across
neighbouring frames at no extra parameter cost (Lin et al., ICCV 2019).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class TemporalShift(nn.Module):
    """Shift ``1/fold_div`` of channels back/forward in time, then run ``net``."""

    def __init__(self, net: nn.Module, n_segment: int, fold_div: int = 8):
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shift(x, self.n_segment, self.fold_div)
        return self.net(x)

    @staticmethod
    def _shift(x: torch.Tensor, n_segment: int, fold_div: int) -> torch.Tensor:
        nt, c, h, w = x.size()
        b = nt // n_segment
        x = x.view(b, n_segment, c, h, w)
        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # shift left  (future)
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right (past)
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # unchanged
        return out.view(nt, c, h, w)


class TSM(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 8,
                 pretrained: bool = True, fold_div: int = 8, dropout: float = 0.5):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self._inject_shift(net, num_frames, fold_div)
        self.backbone = net
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)
        self.num_frames = num_frames

    @staticmethod
    def _inject_shift(net, n_segment: int, fold_div: int) -> None:
        for layer in (net.layer1, net.layer2, net.layer3, net.layer4):
            for block in layer:
                block.conv1 = TemporalShift(block.conv1, n_segment, fold_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).reshape(B, T, -1).mean(dim=1)
        return self.fc(self.dropout(feat))
