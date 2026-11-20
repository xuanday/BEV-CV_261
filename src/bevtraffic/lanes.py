"""Road-marking extraction with HLS colour masks, Canny and the probabilistic Hough transform.

Used for two things:
  * picking calibration points (segment end points on lane lines / stop lines)
  * checking a calibration: after warping to BEV, lane lines must become parallel
    and point along the road axis, and the vanishing point of the image lines
    must agree with the homography.
"""
from __future__ import annotations

import cv2
import numpy as np


def marking_mask(img_bgr: np.ndarray, white_l: int = 175, white_s_max: int = 110) -> np.ndarray:
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, (0, white_l, 0), (180, 255, white_s_max))
    yellow = cv2.inRange(hls, (12, 85, 75), (38, 255, 255))
    return cv2.bitwise_or(white, yellow)


def hough_segments(img_bgr: np.ndarray, roi: np.ndarray | None = None, canny=(50, 150),
                   threshold: int = 25, min_len: int = 25, max_gap: int = 20, white_l: int = 175) -> np.ndarray:
    """Returns (N, 4) segments x1, y1, x2, y2. Raise white_l on bright (concrete) roads."""
    mask = marking_mask(img_bgr, white_l=white_l)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.bitwise_and(gray, gray, mask=mask), (5, 5), 0), *canny)
    if roi is not None:
        m = np.zeros_like(edges)
        cv2.fillPoly(m, [np.asarray(roi, dtype=np.int32)], 255)
        edges = cv2.bitwise_and(edges, m)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_len, maxLineGap=max_gap)
    return np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)


def segment_angles(seg: np.ndarray) -> np.ndarray:
    """Angle to the image vertical axis in degrees, in (-90, 90]."""
    dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
    a = np.degrees(np.arctan2(dx, dy))
    a[a > 90] -= 180
    a[a <= -90] += 180
    return a


def vanishing_point(seg: np.ndarray, max_angle: float = 70.0) -> tuple[np.ndarray | None, int]:
    """Least-squares intersection of the (roughly road-aligned) segments.
    Each segment gives a line n.x = c; minimise sum w (n.x - c)^2 with w = length."""
    if len(seg) < 2:
        return None, 0
    keep = np.abs(segment_angles(seg)) < max_angle
    s = seg[keep]
    if len(s) < 2:
        return None, 0
    d = s[:, 2:] - s[:, :2]
    L = np.linalg.norm(d, axis=1)
    n = np.stack([d[:, 1], -d[:, 0]], 1) / L[:, None]
    c = np.sum(n * s[:, :2], 1)
    A = n * np.sqrt(L)[:, None]
    b = c * np.sqrt(L)
    vp, *_ = np.linalg.lstsq(A, b, rcond=None)
    return vp, int(keep.sum())


def bev_parallelism(bev_bgr: np.ndarray, road_axis_deg: float = 0.0, tol: float = 20.0) -> dict:
    """Detect markings on a BEV image and measure how far the near-axis segments
    deviate from the road axis. A good calibration gives a small, centred spread."""
    seg = hough_segments(bev_bgr, threshold=20, min_len=15, max_gap=10)
    if len(seg) == 0:
        return {"n": 0}
    ang = segment_angles(seg) - road_axis_deg
    L = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    m = np.abs(ang) < tol
    if not m.any():
        return {"n": 0}
    w = L[m] / L[m].sum()
    mean = float(np.sum(w * ang[m]))
    return {"n": int(m.sum()), "mean_dev_deg": mean, "std_dev_deg": float(np.sqrt(np.sum(w * (ang[m] - mean) ** 2)))}


def draw_segments(img: np.ndarray, seg: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for x1, y1, x2, y2 in seg.astype(int):
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out
