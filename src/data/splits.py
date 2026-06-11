"""Build train/val/test sample lists.

Two sources of the train/test partition:

1. **Official HMDB51 splits** (``<class>_test_split1.txt``, tag 1=train, 2=test,
   0=unused) when those files are present.
2. **Stratified seeded split** built directly from the downloaded clips when the
   official split files are absent (e.g. the HuggingFace mirror ships only the
   videos). This is deterministic (fixed seed) and per-class stratified, which is
   correct for our 10-class subset — the choice is documented in the report.

In both cases a validation set is carved out of the training portion
(stratified, fixed seed), since HMDB51 has no official validation split.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import CLASS_TO_IDX, CLASSES, MAX_PER_CLASS, RAW_DIR, SPLIT_DIR

Sample = Tuple[str, int]


def _read_split_file(path: Path) -> Dict[str, int]:
    tags = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            parts = line.split()
            tags[parts[0]] = int(parts[1])
    return tags


def _have_official(split_idx: int, split_dir: Path) -> bool:
    return all((split_dir / f"{c}_test_split{split_idx}.txt").exists() for c in CLASSES)


def _from_official(split_idx: int, raw_dir: Path, split_dir: Path
                   ) -> Tuple[List[Sample], List[Sample]]:
    train, test = [], []
    for cls in CLASSES:
        label = CLASS_TO_IDX[cls]
        tags = _read_split_file(split_dir / f"{cls}_test_split{split_idx}.txt")
        for fname, tag in tags.items():
            vpath = raw_dir / cls / fname
            if not vpath.exists():
                continue
            if tag == 1:
                train.append((str(vpath), label))
            elif tag == 2:
                test.append((str(vpath), label))
    return train, test


def _stratified(raw_dir: Path, test_ratio: float, seed: int,
                max_per_class: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)
    train, test = [], []
    for cls in CLASSES:
        label = CLASS_TO_IDX[cls]
        clips = sorted(str(p) for p in (raw_dir / cls).glob("*.avi"))
        rng.shuffle(clips)
        if max_per_class:
            clips = clips[:max_per_class]      # tame class imbalance (seeded)
        n_test = max(1, int(round(len(clips) * test_ratio))) if clips else 0
        test += [(p, label) for p in clips[:n_test]]
        train += [(p, label) for p in clips[n_test:]]
    return train, test


def build_splits(split_idx: int = 1, val_ratio: float = 0.15, test_ratio: float = 0.3,
                 seed: int = 42, raw_dir: Path = RAW_DIR, split_dir: Path = SPLIT_DIR,
                 prefer_official: bool = True, max_per_class: int = MAX_PER_CLASS
                 ) -> Dict[str, List[Sample]]:
    """Return {"train": [...], "val": [...], "test": [...]} of (abs_path, label)."""
    if prefer_official and _have_official(split_idx, split_dir):
        train, test = _from_official(split_idx, raw_dir, split_dir)
        source = "official"
    else:
        train, test = _stratified(raw_dir, test_ratio, seed, max_per_class)
        source = f"stratified(seeded, max_per_class={max_per_class})"

    # carve a stratified validation set out of train (fixed seed)
    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for s in train:
        by_class.setdefault(s[1], []).append(s)
    final_train, val = [], []
    for items in by_class.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if items else 0
        val.extend(items[:n_val])
        final_train.extend(items[n_val:])

    rng.shuffle(final_train)
    return {"train": final_train, "val": val, "test": test, "_source": source}


if __name__ == "__main__":
    splits = build_splits()
    print(f"split source: {splits['_source']}")
    for k in ("train", "val", "test"):
        print(f"{k:5s}: {len(splits[k])} clips")
