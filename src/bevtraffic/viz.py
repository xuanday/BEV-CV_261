"""Drawing helpers: calibration sheet, BEV grid, dashboard video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration

COLORS = {"motorbike": (255, 170, 0), "car": (60, 220, 60), "bus": (0, 200, 255), "truck": (60, 60, 255)}


def draw_bev_grid(bev: np.ndarray, cal: Calibration, step_m: float = 5.0) -> np.ndarray:
    out = bev.copy()
    h, w = out.shape[:2]
    for y in np.arange(np.ceil(cal.y_range[0] / step_m) * step_m, cal.y_range[1] + 1e-6, step_m):
        py = int(round((y - cal.y_range[0]) * cal.px_per_m))
        cv2.line(out, (0, py), (w, py), (0, 255, 255), 1)
        cv2.putText(out, f"{y:.0f}m", (2, max(10, py - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for x in np.arange(np.ceil(cal.x_range[0] / step_m) * step_m, cal.x_range[1] + 1e-6, step_m):
        px = int(round((x - cal.x_range[0]) * cal.px_per_m))
        cv2.line(out, (px, 0), (px, h), (0, 255, 255), 1)
    for ln in cal.lanes:
        p = cal.world_to_bev_px(ln.polygon_m).round().astype(np.int32)
        cv2.polylines(out, [p], True, (255, 0, 255), 1)
    for tw in cal.tripwires:
        a, b = cal.world_to_bev_px([tw.p1, tw.p2]).round().astype(int)
        cv2.line(out, tuple(a), tuple(b), (0, 0, 255), 2)
    return out


def calibration_sheet(img: np.ndarray, cal: Calibration, out_path: str | Path | None = None, height: int = 720) -> np.ndarray:
    """Left: frame with fit points (green), check points (red, with predicted position)
    and the measurement zone. Right: BEV warp with a metre grid."""
    vis = img.copy()
    if cal.zone_uv is not None:
        cv2.polylines(vis, [cal.zone_uv.astype(np.int32)], True, (0, 255, 255), 2)
    for name, color in (("fit", (0, 255, 0)), ("check", (0, 0, 255))):
        pts = cal.fit_points if name == "fit" else cal.check_points
        for (u, v), (x, y) in zip(pts.get("img", []), pts.get("world", [])):
            cv2.circle(vis, (int(u), int(v)), 6, color, 2)
            pu, pv = cal.to_image([[x, y]])[0]
            cv2.drawMarker(vis, (int(pu), int(pv)), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(vis, f"({x:g},{y:g})", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    bev = draw_bev_grid(cal.warp_bev(img), cal)
    a = cv2.resize(vis, (int(vis.shape[1] * height / vis.shape[0]), height))
    b = cv2.resize(bev, (int(bev.shape[1] * height / bev.shape[0]), height))
    sheet = np.hstack([a, b])
    if out_path:
        cv2.imwrite(str(out_path), sheet)
    return sheet


def render_dashboard(video: str | Path, res: dict, cal: Calibration, out_path: str | Path,
                     max_frames: int | None = None, height: int = 540, start: int = 0, step: int = 1) -> int:
    """Camera view (boxes, id, speed) next to the BEV view (dots, trails, congestion map).
    step > 1 keeps every step-th frame (use the tracking stride)."""
    pts = res["points"]
    by_f = {f: g for f, g in pts.groupby("frame_id")}
    viol = set(res["violations"]["track_id"]) if len(res.get("violations", [])) else set()
    counted = res["crossings"].sort_values("frame_id") if len(res.get("crossings", [])) else None
    thw = res.get("headway")
    cong = res.get("congestion")
    cap = cv2.VideoCapture(str(video))
    fps = float(pts["fps"].iloc[0])
    W, H = cal.image_size
    bw, bh = cal.bev_size
    cam_w = int(W * height / H); bev_w = int(bw * height / bh)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (cam_w + bev_w, height))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    heat = None
    if cong is not None and cong.max() > 0:
        heat = cv2.applyColorMap(np.uint8(255 * cong / cong.max()), cv2.COLORMAP_JET)
    trails: dict = {}
    f, written = start, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and f >= start + max_frames):
            break
        if (f - start) % step:
            f += 1
            continue
        g = by_f.get(f)
        cam = frame.copy()
        if cal.zone_uv is not None:
            cv2.polylines(cam, [cal.zone_uv.astype(np.int32)], True, (0, 255, 255), 2)
        bev = cal.warp_bev(frame)
        if heat is not None:
            m = cong > 0.02 * cong.max()
            bev[m] = cv2.addWeighted(bev, 0.5, heat, 0.5, 0)[m]
        bev = draw_bev_grid(bev, cal)
        if g is not None:
            for r in g.itertuples():
                x1, y1, x2, y2 = map(int, r.bbox_xyxy)
                col = (0, 0, 255) if r.track_id in viol else COLORS.get(r.cls, (200, 200, 200))
                cv2.rectangle(cam, (x1, y1), (x2, y2), col, 2)
                sp = "" if np.isnan(r.ema_kmh) else f" {r.ema_kmh:.0f}km/h"
                cv2.putText(cam, f"#{r.track_id} {r.cls}{sp}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                if r.in_zone and not r.edge:
                    bx, by = cal.world_to_bev_px([[r.x, r.y]])[0].astype(int)
                    trails.setdefault(r.track_id, []).append((bx, by))
                    tr = trails[r.track_id][-30:]
                    for a, b in zip(tr[:-1], tr[1:]):
                        cv2.line(bev, a, b, col, 1)
                    cv2.circle(bev, (bx, by), 4, col, -1)
        n_count = int((counted["frame_id"] <= f).sum()) if counted is not None else 0
        cv2.rectangle(cam, (0, 0), (520, 44), (0, 0, 0), -1)
        cv2.putText(cam, f"{cal.camera_id}  t={f / fps:5.1f}s  counted={n_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        wr.write(np.hstack([cv2.resize(cam, (cam_w, height)), cv2.resize(bev, (bev_w, height))]))
        f += 1; written += 1
    cap.release(); wr.release()
    return written
