"""R(2+1)D — factorised 3D convolutions.

Family: 3D-CNN. Each 3D convolution is decomposed into a 2D spatial convolution
followed by a 1D temporal convolution, which adds non-linearity and eases
optimisation. We use torchvision's ``r2plus1d_18`` with Kinetics-400 weights and
replace the classification head (Tran et al., CVPR 2018).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class R2Plus1D(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, **_):
        super().__init__()
        weights = (torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
                   if pretrained else None)
        net = torchvision.models.video.r2plus1d_18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W] -> [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
