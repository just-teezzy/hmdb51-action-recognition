"""VideoMAE — masked-autoencoder Video Transformer (transformer family).

Family: pure Transformer over space-time tubelets. We fine-tune HuggingFace's
``MCG-NJU/videomae-base-finetuned-kinetics``. A ``timesformer`` backend and a
``tiny`` (random-init, small) configuration are provided as light fallbacks for
limited memory / the smoke test (no network, no large weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VideoMAE(nn.Module):
    def __init__(self, num_classes: int, num_frames: int = 16, img_size: int = 224,
                 pretrained: bool = True, tiny: bool = False,
                 backend: str = "videomae", **_):
        super().__init__()
        self.backend = backend
        self.num_frames = num_frames
        self.img_size = img_size
        if backend == "timesformer":
            self.model = self._build_timesformer(num_classes, num_frames, img_size,
                                                 pretrained, tiny)
        else:
            self.model = self._build_videomae(num_classes, num_frames, img_size,
                                              pretrained, tiny)

    @staticmethod
    def _build_videomae(num_classes, num_frames, img_size, pretrained, tiny):
        from transformers import (VideoMAEConfig, VideoMAEForVideoClassification)
        if tiny or not pretrained:
            cfg = VideoMAEConfig(
                image_size=img_size, num_frames=num_frames, num_labels=num_classes,
                hidden_size=192, num_hidden_layers=4, num_attention_heads=3,
                intermediate_size=768, tubelet_size=2)
            return VideoMAEForVideoClassification(cfg)
        return VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=num_classes, ignore_mismatched_sizes=True)

    @staticmethod
    def _build_timesformer(num_classes, num_frames, img_size, pretrained, tiny):
        from transformers import (TimesformerConfig, TimesformerForVideoClassification)
        if tiny or not pretrained:
            cfg = TimesformerConfig(
                image_size=img_size, num_frames=num_frames, num_labels=num_classes,
                hidden_size=192, num_hidden_layers=4, num_attention_heads=3,
                intermediate_size=768)
            return TimesformerForVideoClassification(cfg)
        return TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W]
        # HuggingFace video models expect pixel_values [B, T, C, H, W].
        return self.model(pixel_values=x).logits
