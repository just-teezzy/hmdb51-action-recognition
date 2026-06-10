"""Reusable single-video inference (shared by the demo and the smoke test)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src import config
from src.data.dataset import decode_video, sample_indices, transform_clip
from src.models.registry import build_model
from src.utils import get_device, load_checkpoint

_CACHE: dict = {}


def load_model(model_name: str, ckpt: Optional[str] = None, tiny: bool = False,
               device=None):
    device = device or get_device()
    key = (model_name, str(ckpt), tiny, str(device))
    if key in _CACHE:
        return _CACHE[key]
    mcfg = config.get_config(model_name)
    model = build_model(model_name, config.NUM_CLASSES, num_frames=mcfg["num_frames"],
                        img_size=mcfg["img_size"], pretrained=False, tiny=tiny)
    path = Path(ckpt) if ckpt else (config.CHECKPOINT_DIR / f"{model_name}_best.pt")
    if not path.exists():
        path = config.CHECKPOINT_DIR / f"{model_name}.pt"
    loaded = path.exists()
    if loaded:
        load_checkpoint(path, model, map_location=str(device))
    model.to(device).eval()
    _CACHE[key] = (model, mcfg, device, loaded)
    return _CACHE[key]


@torch.no_grad()
def predict_video(video_path: str, model_name: str = "r2plus1d",
                  ckpt: Optional[str] = None, tiny: bool = False, topk: int = 3):
    """Return prediction dict for one video: top-k labels, latency, preview frames."""
    model, mcfg, device, loaded = load_model(model_name, ckpt, tiny)
    frames = decode_video(video_path)
    idx = sample_indices(len(frames), mcfg["num_frames"], train=False)
    clip = transform_clip(frames[idx], mcfg["img_size"], train=False)
    clip = clip.unsqueeze(0).to(device)

    t0 = time.perf_counter()
    logits = model(clip)
    latency = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=1)[0]
    k = min(topk, probs.numel())
    top_p, top_i = probs.topk(k)
    top = [(config.IDX_TO_CLASS[int(i)], float(p)) for p, i in zip(top_p, top_i)]
    return {
        "model": model_name,
        "checkpoint_loaded": loaded,
        "top": top,
        "pred": top[0][0],
        "confidence": top[0][1],
        "latency_s": latency,
        "num_frames_decoded": int(len(frames)),
        "preview_frames": frames[idx],  # [T,H,W,3] uint8 for display
    }
