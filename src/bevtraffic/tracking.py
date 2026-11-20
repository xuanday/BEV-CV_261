"""Detection + multi-object tracking -> list of track records (Data/contracts/schema.json)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

CLASSES = {0: "motorbike", 1: "car", 2: "bus", 3: "truck"}
# COCO ids kept when running the pretrained model, mapped to the 4 project classes
COCO_TO_PROJECT = {3: 0, 2: 1, 5: 2, 7: 3}   # motorcycle, car, bus, truck

TRACKER_CFGS = {
    "bytetrack_vn": {"tracker_type": "bytetrack", "track_high_thresh": 0.45, "track_low_thresh": 0.10,
                     "new_track_thresh": 0.55, "track_buffer": 30, "match_thresh": 0.80, "fuse_score": True},
    "botsort_vn": {"tracker_type": "botsort", "track_high_thresh": 0.45, "track_low_thresh": 0.10,
                   "new_track_thresh": 0.55, "track_buffer": 45, "match_thresh": 0.80, "fuse_score": True,
                   "gmc_method": "sparseOptFlow", "proximity_thresh": 0.50, "appearance_thresh": 0.25,
                   "with_reid": False, "model": "auto"},
}


def write_tracker_cfg(name: str, out_dir: str | Path) -> Path:
    import yaml
    p = Path(out_dir) / f"{name}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(TRACKER_CFGS[name], sort_keys=False))
    return p


def track_video(weights: str | Path, video: str | Path, camera_id: str, tracker: str = "bytetrack_vn",
                coco: bool = False, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                max_frames: int | None = None, fps: float | None = None, device: str | None = None,
                vid_stride: int = 1, tracker_dir: str | Path = "outputs/trackers") -> tuple[list, dict]:
    """Runs model.track over the whole stream. frame_id is the real frame index of the
    source video (frames without detections and skipped frames are still counted).
    Pass fps explicitly for raw .h264 streams, which carry no timing."""
    from ultralytics import YOLO
    import cv2

    if fps is None:
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
    tracker_path = tracker if str(tracker).endswith(".yaml") else str(write_tracker_cfg(tracker, tracker_dir))
    model = YOLO(str(weights))
    classes = sorted(COCO_TO_PROJECT) if coco else None
    kw = dict(source=str(video), tracker=tracker_path, persist=True, stream=True, conf=conf, iou=iou,
              imgsz=imgsz, classes=classes, vid_stride=vid_stride, verbose=False)
    if device is not None:
        kw["device"] = device

    records, t0, n = [], time.perf_counter(), 0
    for k, res in enumerate(model.track(**kw)):
        fid = k * vid_stride
        if max_frames is not None and fid >= max_frames:
            break
        n = k + 1
        b = res.boxes
        if b is None or b.id is None:
            continue
        for xyxy, tid, c, s in zip(b.xyxy.cpu().numpy(), b.id.int().cpu().tolist(),
                                   b.cls.int().cpu().tolist(), b.conf.cpu().tolist()):
            cid = COCO_TO_PROJECT.get(c) if coco else int(c)
            if cid is None:
                continue
            x1, y1, x2, y2 = (round(float(v), 2) for v in xyxy)
            records.append({
                "camera_id": camera_id, "frame_id": fid, "timestamp_ms": round(fid * 1000.0 / fps, 3),
                "fps": fps, "track_id": int(tid), "class_id": cid, "class_name": CLASSES[cid],
                "conf": round(float(s), 4), "bbox_xyxy": [x1, y1, x2, y2],
                "footpoint_uv": [round((x1 + x2) / 2, 2), y2],
            })
    dt = time.perf_counter() - t0
    return records, {"frames": n, "seconds": round(dt, 2), "fps_proc": round(n / max(dt, 1e-6), 2)}


def majority_class(records: list) -> dict:
    """Track id -> most frequent class (weighted by confidence). Fixes class flicker inside a track."""
    from collections import defaultdict
    acc = defaultdict(lambda: np.zeros(4))
    for r in records:
        acc[r["track_id"]][r["class_id"]] += r.get("conf", 1.0)
    return {tid: CLASSES[int(np.argmax(v))] for tid, v in acc.items()}
