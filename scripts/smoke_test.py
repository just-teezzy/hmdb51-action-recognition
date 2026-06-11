"""End-to-end smoke test on tiny synthetic data (CPU, no downloads).

Exercises every pipeline stage so bugs surface before the real Colab run:
    models forward/backward -> train -> evaluate -> compare -> infer/demo -> report

Run:
    python scripts/smoke_test.py
Exit code 0 = all stages passed.
"""
from __future__ import annotations

import gc
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Redirect all outputs to a throwaway dir so the smoke run (synthetic data) never
# overwrites the real Colab results committed under results/ / checkpoints/.
_SMOKE_OUT = Path(tempfile.mkdtemp(prefix="venya_smoke_"))
os.environ.setdefault("VENYA_RESULTS_DIR", str(_SMOKE_OUT / "results"))
os.environ.setdefault("VENYA_CHECKPOINT_DIR", str(_SMOKE_OUT / "checkpoints"))
os.environ.setdefault("VENYA_REPORT_DIR", str(_SMOKE_OUT / "report"))
os.environ.setdefault("VENYA_DEMO_DB", str(_SMOKE_OUT / "history.db"))

from src import config  # noqa: E402
from src.models.registry import build_model  # noqa: E402

PY = sys.executable
RESULTS = []


def stage(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def run(cmd: list) -> bool:
    p = subprocess.run([PY] + cmd, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
    return p.returncode == 0


# --- Stage 1: every model forward + backward -------------------------------
def stage_models() -> None:
    shapes = {
        "tsn": (1, 8, 3, 64, 64),
        "tsm": (1, 8, 3, 64, 64),
        "r2plus1d": (1, 8, 3, 112, 112),
        "i3d": (1, 16, 3, 112, 112),      # adaptive head pool -> works at 112 too
        "videomae": (1, 16, 3, 112, 112),
    }
    crit = torch.nn.CrossEntropyLoss()
    for name, shp in shapes.items():
        try:
            kw = dict(pretrained=False, num_frames=shp[1], img_size=shp[-1])
            if name == "videomae":
                kw["tiny"] = True
            model = build_model(name, config.NUM_CLASSES, **kw)
            x = torch.randn(*shp)
            logits = model(x)
            assert logits.shape == (1, config.NUM_CLASSES), logits.shape
            loss = crit(logits, torch.tensor([0]))
            loss.backward()
            stage(f"model:{name}", True, f"logits {tuple(logits.shape)}")
            del model, x, logits, loss
            gc.collect()
        except Exception as e:  # noqa: BLE001
            stage(f"model:{name}", False, repr(e))


# --- Stage 2-4: train / evaluate / compare via CLI -------------------------
def stage_pipeline() -> None:
    data = ["--synthetic", "--frames", "8", "--img-size", "64",
            "--per-class", "6", "--batch-size", "4"]
    stage("train:tsn", run(["-m", "src.train", "--model", "tsn", "--epochs", "1",
                            "--no-pretrained", "--no-freeze"] + data))
    stage("evaluate:tsn", run(["-m", "src.evaluate", "--model", "tsn"] + data))
    stage("compare", run(["-m", "src.compare"]))


# --- Stage 5: inference + demo --------------------------------------------
def _make_dummy_video(path: str, n: int = 20, h: int = 96, w: int = 96) -> None:
    import av
    container = av.open(path, "w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for t in range(n):
        img = (np.random.rand(h, w, 3) * 60).astype(np.uint8)
        x = (t * 3) % (w - 30)
        img[20:50, x:x + 30] = [200, 50, 50]
        for pkt in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def stage_demo() -> None:
    try:
        from src.inference import predict_video
        from demo import history
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            vid = f.name
        _make_dummy_video(vid)
        res = predict_video(vid, model_name="tsn", topk=3)
        assert res["pred"] in config.CLASSES
        history.log_run("dummy.mp4", "tsn", res["pred"], res["confidence"],
                        res["latency_s"], res["top"])
        assert history.summary()["total"] >= 1
        stage("inference+history", True,
              f"pred={res['pred']} conf={res['confidence']:.2f}")
    except Exception as e:  # noqa: BLE001
        stage("inference+history", False, repr(e))

    try:
        py_compile.compile(str(ROOT / "demo" / "app.py"), doraise=True)
        stage("demo:app.py compiles", True)
    except Exception as e:  # noqa: BLE001
        stage("demo:app.py compiles", False, repr(e))


# --- Stage 6: report -------------------------------------------------------
def stage_report() -> None:
    stage("report", run(["-m", "reports.generate_report"]))


def main() -> int:
    print(f"python: {PY}")
    print(f"torch:  {torch.__version__}\n")
    stage_models()
    stage_pipeline()
    stage_demo()
    stage_report()
    print("\n=== SUMMARY ===")
    n_ok = sum(1 for _, ok in RESULTS if ok)
    for name, ok in RESULTS:
        print(f"  {'OK ' if ok else 'ERR'}  {name}")
    print(f"{n_ok}/{len(RESULTS)} stages passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
