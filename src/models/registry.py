"""Model factory: one uniform constructor for all 5 families.

Every model returned here obeys the same contract:
    forward(x) with x of shape [B, T, C, H, W]  ->  logits [B, num_classes]
so train / eval / demo never need to know which architecture they hold.
"""
from __future__ import annotations

import torch.nn as nn

from src.config import MODEL_NAMES
from src.models.i3d import I3D
from src.models.r2plus1d import R2Plus1D
from src.models.tsm import TSM
from src.models.tsn import TSN
from src.models.videomae import VideoMAE


def build_model(name: str, num_classes: int, num_frames: int = 16,
                img_size: int = 112, pretrained: bool = True,
                tiny: bool = False) -> nn.Module:
    name = name.lower()
    if name == "tsn":
        return TSN(num_classes, num_frames=num_frames, pretrained=pretrained)
    if name == "tsm":
        return TSM(num_classes, num_frames=num_frames, pretrained=pretrained)
    if name == "i3d":
        return I3D(num_classes, pretrained=pretrained)
    if name == "r2plus1d":
        return R2Plus1D(num_classes, pretrained=pretrained)
    if name == "videomae":
        return VideoMAE(num_classes, num_frames=num_frames, img_size=img_size,
                        pretrained=pretrained, tiny=tiny, backend="videomae")
    if name == "timesformer":
        return VideoMAE(num_classes, num_frames=num_frames, img_size=img_size,
                        pretrained=pretrained, tiny=tiny, backend="timesformer")
    raise ValueError(f"Unknown model '{name}'. Choices: {', '.join(MODEL_NAMES)}")
