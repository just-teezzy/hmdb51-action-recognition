"""Download the HMDB51 10-class subset.

Primary source: **HuggingFace mirror** ``jili5044/hmdb51`` — the clips are already
extracted as ``.avi`` inside per-class folders, so there is no RAR/unrar dance.
This is the recommended path and works out-of-the-box in Colab.

Fallback source: the original serre-lab RAR archives (``--source serre-lab``),
kept for completeness; it needs an ``unrar`` backend.

Usage
-----
    python -m src.data.download_hmdb                 # HF mirror (recommended)
    python -m src.data.download_hmdb --source serre-lab
    python -m src.data.download_hmdb --check
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import CLASSES, DATA_DIR, RAW_DIR, SPLIT_DIR

# --- HuggingFace mirror ----------------------------------------------------
HF_REPO = "jili5044/hmdb51"

# --- serre-lab fallback ----------------------------------------------------
VIDEO_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
SPLIT_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"
MIN_SIZE = {"hmdb51_org.rar": 1_900_000_000, "test_train_splits.rar": 100_000}


# ===========================================================================
# Primary: HuggingFace mirror
# ===========================================================================
def download_hf() -> None:
    """Download only our classes from the HF mirror and lay them out as
    ``data/raw/<class>/*.avi``."""
    from huggingface_hub import snapshot_download

    print(f"[hf] snapshot_download({HF_REPO}) for {len(CLASSES)} classes")
    # Limit the download to paths mentioning our class names (fnmatch over the
    # full relative path — the class folder name is part of it).
    patterns = [f"*{c}*" for c in CLASSES]
    local = snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                              allow_patterns=patterns)
    local = Path(local)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    copied = {c: 0 for c in CLASSES}
    for avi in local.rglob("*.avi"):
        cls = avi.parent.name              # layout: .../<class>/<video>.avi
        if cls not in CLASSES:
            continue
        dst_dir = RAW_DIR / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / avi.name
        if not dst.exists():
            try:
                os.link(avi, dst)          # hard-link to avoid duplicating GBs
            except OSError:
                shutil.copy2(avi, dst)
        copied[cls] += 1

    missing = [c for c, n in copied.items() if n == 0]
    for c in CLASSES:
        print(f"  {c:12s} {copied[c]:4d} clips")
    if missing:
        print(f"  WARNING no clips found for: {missing} "
              f"(inspect the mirror layout under {local})")


# ===========================================================================
# Fallback: serre-lab RAR archives
# ===========================================================================
def _extract_rar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import patoolib
        patoolib.extract_archive(str(archive), outdir=str(dest), verbosity=-1)
        return
    except Exception as e:  # noqa: BLE001
        print(f"  patool failed ({e}); trying rarfile…")
    import rarfile
    with rarfile.RarFile(str(archive)) as rf:
        rf.extractall(str(dest))


def _verify_size(dest: Path) -> bool:
    min_size = MIN_SIZE.get(dest.name, 1)
    actual = dest.stat().st_size if dest.exists() else 0
    if dest.exists() and actual < min_size:
        print(f"  size check: {dest.name} {actual:,}B < {min_size:,}B (incomplete)")
        return False
    return actual >= min_size


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _verify_size(dest):
        print(f"  already downloaded & verified: {dest.name}")
        return dest
    print(f"  downloading (resumable) {url} -> {dest.name}")
    rc = subprocess.run(["wget", "-c", "--tries=10", "--read-timeout=60",
                         "--waitretry=5", "-O", str(dest), url]).returncode
    if rc != 0:
        print("  wget failed, retrying with curl -C -")
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), url], check=True)
    if not _verify_size(dest):
        raise RuntimeError(f"{dest.name} incomplete after download; re-run to resume.")
    return dest


def download_serre_lab() -> None:
    print("[serre-lab] splits")
    arc = _download(SPLIT_URL, DATA_DIR / "test_train_splits.rar")
    tmp = DATA_DIR / "_splits_tmp"
    _extract_rar(arc, tmp)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for f in tmp.rglob("*_test_split*.txt"):
        if f.name.split("_test_split")[0] in CLASSES:
            shutil.copy(f, SPLIT_DIR / f.name)
    shutil.rmtree(tmp, ignore_errors=True)

    print("[serre-lab] videos")
    arc = _download(VIDEO_URL, DATA_DIR / "hmdb51_org.rar")
    tmp = DATA_DIR / "_video_tmp"
    _extract_rar(arc, tmp)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        inner = tmp / f"{cls}.rar"
        if inner.exists():
            _extract_rar(inner, RAW_DIR)
        else:
            print(f"  WARNING missing inner archive: {inner.name}")
    shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
def check() -> None:
    print("[check]")
    total = 0
    for cls in CLASSES:
        d = RAW_DIR / cls
        n = len(list(d.glob("*.avi"))) if d.exists() else 0
        sp = (SPLIT_DIR / f"{cls}_test_split1.txt").exists()
        total += n
        print(f"  {cls:12s} videos={n:4d} official_split={'yes' if sp else 'no'}")
    print(f"  total videos: {total}")
    if not any((SPLIT_DIR / f"{c}_test_split1.txt").exists() for c in CLASSES):
        print("  note: no official split files -> a stratified split (seeded) is "
              "built automatically by src.data.splits")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "serre-lab"], default="hf")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.source == "hf":
        download_hf()
    else:
        download_serre_lab()
    check()


if __name__ == "__main__":
    sys.exit(main())
