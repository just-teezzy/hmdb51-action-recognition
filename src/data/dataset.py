"""Video clip dataset.

Two dataset classes share the same output contract so the rest of the pipeline
is agnostic to where clips come from:

    __getitem__ -> (clip, label)
        clip : float32 tensor of shape [T, C, H, W], normalised
        label: int

* ``VideoClipDataset``    decodes real .avi files with PyAV.
* ``SyntheticClipDataset`` generates deterministic class-correlated clips and is
  used by the smoke test (no data download / no GPU needed).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import MEAN, STD


# ---------------------------------------------------------------------------
# Temporal sampling
# ---------------------------------------------------------------------------
def sample_indices(total: int, num_frames: int, train: bool) -> np.ndarray:
    """Uniformly split [0, total) into ``num_frames`` segments and pick one frame
    from each (jittered when training, centred otherwise)."""
    if total <= 0:
        return np.zeros(num_frames, dtype=int)
    if total < num_frames:
        # loop short clips
        return np.sort(np.mod(np.arange(num_frames), total))
    seg = total / num_frames
    if train:
        offsets = np.random.uniform(0, seg, size=num_frames)
    else:
        offsets = np.full(num_frames, seg / 2.0)
    idx = (np.arange(num_frames) * seg + offsets).astype(int)
    return np.clip(idx, 0, total - 1)


# ---------------------------------------------------------------------------
# Clip-level transforms (same spatial params across all T frames)
# ---------------------------------------------------------------------------
def _to_clip_tensor(frames: np.ndarray) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [T,3,H,W] float in [0,1]."""
    t = torch.from_numpy(frames).float().div_(255.0)
    return t.permute(0, 3, 1, 2).contiguous()


def transform_clip(frames: np.ndarray, img_size: int, train: bool) -> torch.Tensor:
    import torch.nn.functional as F

    clip = _to_clip_tensor(frames)               # [T,3,H,W]
    T, C, H, W = clip.shape

    # resize shorter side to ~1.15 * img_size
    target = int(round(img_size * 1.15))
    scale = target / min(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    clip = F.interpolate(clip, size=(nh, nw), mode="bilinear", align_corners=False)

    # spatial crop (shared box)
    if train:
        top = np.random.randint(0, nh - img_size + 1)
        left = np.random.randint(0, nw - img_size + 1)
    else:
        top = (nh - img_size) // 2
        left = (nw - img_size) // 2
    clip = clip[:, :, top:top + img_size, left:left + img_size]

    # horizontal flip
    if train and np.random.rand() < 0.5:
        clip = torch.flip(clip, dims=[3])

    # normalise
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    clip = (clip - mean) / std
    return clip


# ---------------------------------------------------------------------------
# PyAV decoding
# ---------------------------------------------------------------------------
def decode_video(path: str) -> np.ndarray:
    """Decode an entire short video to [N,H,W,3] uint8 (RGB)."""
    import av
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Real dataset
# ---------------------------------------------------------------------------
class VideoClipDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], num_frames: int,
                 img_size: int, train: bool):
        self.samples = samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        try:
            frames = decode_video(path)
            idx = sample_indices(len(frames), self.num_frames, self.train)
            clip = transform_clip(frames[idx], self.img_size, self.train)
        except Exception:
            # robust to an occasional unreadable file
            clip = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
        return clip, label


# ---------------------------------------------------------------------------
# Synthetic dataset (smoke test)
# ---------------------------------------------------------------------------
class SyntheticClipDataset(Dataset):
    """Deterministic, learnable clips: each class is a coloured square moving in a
    class-specific direction. Lets the full training loop reduce loss without any
    real data / network access."""

    def __init__(self, num_classes: int, per_class: int, num_frames: int,
                 img_size: int, train: bool, seed: int = 0):
        self.num_frames = num_frames
        self.img_size = img_size
        self.train = train
        rng = np.random.RandomState(seed)
        self.items = []
        for c in range(num_classes):
            for _ in range(per_class):
                self.items.append((c, rng.randint(0, 1_000_000)))
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.items)

    def _render(self, c: int, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        H = W = 64
        T = self.num_frames + 4
        clip = (rng.rand(T, H, W, 3) * 40).astype(np.uint8)  # faint noise
        color = np.zeros(3, dtype=np.uint8)
        color[c % 3] = 200
        color[(c // 3) % 3] = min(255, 100 + 30 * (c % 5))
        dx = 1 + (c % 3)
        dy = 1 + (c % 2)
        box = 16
        for t in range(T):
            y = (4 + dy * t) % (H - box)
            x = (4 + dx * t) % (W - box)
            clip[t, y:y + box, x:x + box, :] = color
        return clip

    def __getitem__(self, i: int):
        c, seed = self.items[i]
        frames = self._render(c, seed)
        idx = sample_indices(len(frames), self.num_frames, self.train)
        clip = transform_clip(frames[idx], self.img_size, self.train)
        return clip, c
