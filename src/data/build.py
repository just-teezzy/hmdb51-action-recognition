"""Build train/val/test torch Datasets (real HMDB51 or synthetic smoke data)."""
from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import Dataset

from src import config
from src.data.dataset import SyntheticClipDataset, VideoClipDataset
from src.data.splits import build_splits


def get_datasets(num_frames: int, img_size: int, synthetic: bool = False,
                 per_class: int = 12, limit: Optional[int] = None,
                 seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    if synthetic:
        train = SyntheticClipDataset(config.NUM_CLASSES, per_class, num_frames,
                                     img_size, train=True, seed=seed)
        val = SyntheticClipDataset(config.NUM_CLASSES, max(2, per_class // 3),
                                   num_frames, img_size, train=False, seed=seed + 1)
        test = SyntheticClipDataset(config.NUM_CLASSES, max(2, per_class // 3),
                                    num_frames, img_size, train=False, seed=seed + 2)
        return train, val, test

    splits = build_splits(seed=seed)

    def cap(samples):
        return samples[:limit] if limit else samples

    train = VideoClipDataset(cap(splits["train"]), num_frames, img_size, train=True)
    val = VideoClipDataset(cap(splits["val"]), num_frames, img_size, train=False)
    test = VideoClipDataset(cap(splits["test"]), num_frames, img_size, train=False)
    return train, val, test
