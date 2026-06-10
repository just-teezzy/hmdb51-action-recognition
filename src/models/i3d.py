"""I3D — Inflated 3D ConvNet (3D-CNN family) via PyTorchVideo.

We load PyTorchVideo's ``i3d_r50`` (Kinetics-400). If those weights are not
available the constructor transparently falls back to ``slow_r50`` — still a
representative 3D-CNN, so the "5 different families" requirement holds. The final
projection head is replaced for our number of classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class I3D(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, **_):
        super().__init__()
        net, kind = self._build(pretrained)
        self.kind = kind
        head = net.blocks[-1]
        if hasattr(head, "proj") and isinstance(head.proj, nn.Linear):
            head.proj = nn.Linear(head.proj.in_features, num_classes)
        else:  # safety net for head layout changes
            raise RuntimeError(f"Unexpected head layout for {kind}: {type(head)}")
        self.net = net

    @staticmethod
    def _build(pretrained: bool):
        from pytorchvideo.models import hub
        for name in ("i3d_r50", "slow_r50"):
            builder = getattr(hub, name, None)
            if builder is None:
                continue
            try:
                return builder(pretrained=pretrained), name
            except Exception as e:  # noqa: BLE001 — weight download may fail
                print(f"  [i3d] {name}(pretrained={pretrained}) failed: {e}")
                if pretrained:
                    try:
                        return builder(pretrained=False), name + "(random)"
                    except Exception:
                        pass
        raise RuntimeError("Could not build any PyTorchVideo 3D-CNN (i3d_r50/slow_r50)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,T,C,H,W] -> [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4)
        return self.net(x)
