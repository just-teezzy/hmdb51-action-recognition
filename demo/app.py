"""Streamlit demo: upload a video -> predicted action, confidence (top-3), preview.

Run with:
    streamlit run demo/app.py
Inference runs on CPU with a small number of frames.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src import config
from demo import history


def main() -> None:
    st.set_page_config(page_title="Action Recognition Demo", layout="wide")
    st.title("🎬 Распознавание действий на видео")

    # ---- sidebar: model choice (best-accuracy first) ----
    PREF = ["videomae", "i3d", "r2plus1d", "timesformer", "tsm", "tsn"]
    def _has_ckpt(n):
        return ((config.CHECKPOINT_DIR / f"{n}_best.pt").exists()
                or (config.CHECKPOINT_DIR / f"{n}.pt").exists())
    order = [n for n in PREF if n in config.MODEL_NAMES]
    available = [n for n in order if _has_ckpt(n)]
    default_models = available or order
    model_name = st.sidebar.selectbox("Модель", default_models)
    tiny = st.sidebar.checkbox("tiny transformer (если нет весов)", value=False)
    if not available:
        st.sidebar.warning("Чекпоинтов нет — предсказания случайные (демо-режим).")

    st.sidebar.markdown("**Классы:**")
    st.sidebar.write(", ".join(config.CLASSES))

    # ---- upload ----
    uploaded = st.file_uploader("Загрузите видеоролик",
                                type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        from src.inference import predict_video
        with st.spinner("Обработка…"):
            res = predict_video(tmp_path, model_name=model_name, tiny=tiny, topk=3)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(tmp_path)
            st.caption("Сэмплированные кадры (вход модели):")
            frames = res["preview_frames"]
            ncol = min(8, len(frames))
            cols = st.columns(ncol)
            for i in range(ncol):
                cols[i].image(frames[i], use_column_width=True)
        with col2:
            st.subheader(f"Действие: **{res['pred']}**")
            st.metric("Уверенность", f"{res['confidence']*100:.1f}%")
            st.caption(f"Задержка: {res['latency_s']*1000:.0f} ms · "
                       f"кадров декодировано: {res['num_frames_decoded']}")
            st.markdown("**Top-3:**")
            for label, prob in res["top"]:
                st.write(f"{label}")
                st.progress(min(1.0, float(prob)))

        history.log_run(uploaded.name, model_name, res["pred"], res["confidence"],
                        res["latency_s"], res["top"])

    # ---- history & stats ----
    st.divider()
    st.subheader("История обработки")
    stats = history.summary()
    if stats.get("total", 0):
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего роликов", stats["total"])
        c2.metric("Ср. уверенность", f"{stats['avg_confidence']*100:.1f}%")
        c3.metric("Ср. задержка", f"{stats['avg_latency_s']*1000:.0f} ms")
        st.write("По классам:", stats["per_class"])
        st.dataframe(history.fetch_all())
    else:
        st.info("Пока нет обработанных роликов.")


if __name__ == "__main__":
    main()
