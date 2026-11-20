"""Frame sampling, static background and contrast enhancement."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IndexError(f"cannot read frame {index} of {path}")
    return frame


def median_background(path: str | Path, n_samples: int = 60, max_frames: int | None = None) -> np.ndarray:
    """Per-pixel temporal median over frames sampled uniformly from the video.

    Moving vehicles occupy a pixel in less than half of the samples, so the
    median returns the empty road. Vehicles parked for the whole clip stay in
    the background (expected)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # raw h264 streams report 0
        total = max_frames or 3000
    if max_frames:
        total = min(total, max_frames)
    wanted = set(np.linspace(0, total - 1, min(n_samples, total)).astype(int).tolist())
    frames, idx = [], 0
    while idx < total:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def clahe_lightness(img_bgr: np.ndarray, clip: float = 2.5, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L channel of HLS so hue/saturation of paint (white/yellow) are kept."""
    h, l, s = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([h, l, s]), cv2.COLOR_HLS2BGR)
