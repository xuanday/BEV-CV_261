"""Ground-truth loaders and evaluation.

* MOTChallenge-style GT (VNTraffic, AICC22-Custom) -> MOTA / IDF1 / HOTA
* Luvizon et al. vehicles.xml (inductive-loop speeds) -> per-vehicle speed error
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- GT loaders
def load_mot_txt(path: str | Path, frame_offset: int = 0) -> dict:
    """frame -> [(id, [x1, y1, x2, y2])]. frame_offset is added to the file's frame
    number to get the 0-based video frame (VNTraffic: 0, AICC22-Custom: -1)."""
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        f, tid = int(float(p[0])) + frame_offset, int(float(p[1]))
        x, y, w, h = map(float, p[2:6])
        out[f].append((tid, [x, y, x + w, y + h]))
    return dict(out)


def gt_to_records(gt: dict, fps: float, camera_id: str = "gt") -> list:
    """GT boxes in the tracks_stream format (class unknown -> 'car')."""
    recs = []
    for f, items in sorted(gt.items()):
        if f < 0:
            continue
        for tid, (x1, y1, x2, y2) in items:
            recs.append({"camera_id": camera_id, "frame_id": f, "timestamp_ms": f * 1000.0 / fps, "fps": fps,
                         "track_id": tid, "class_id": 1, "class_name": "car", "conf": 1.0,
                         "bbox_xyxy": [x1, y1, x2, y2], "footpoint_uv": [(x1 + x2) / 2, y2]})
    return recs


def records_by_frame(records: list) -> dict:
    out = defaultdict(list)
    for r in records:
        out[r["frame_id"]].append((r["track_id"], r["bbox_xyxy"]))
    return dict(out)


def iou_matrix(a, b) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def estimate_frame_offset(gt_raw: dict, pred: dict, offsets=range(-3, 4)) -> pd.DataFrame:
    """Mean best-IoU between GT boxes (file frame f) and predictions at video frame f + k.
    The best k is the offset to use when loading the GT."""
    rows = []
    for k in offsets:
        vals = []
        for f, g in gt_raw.items():
            p = pred.get(f + k, [])
            if not p:
                vals += [0.0] * len(g)
                continue
            vals += iou_matrix([b for _, b in g], [b for _, b in p]).max(1).tolist()
        rows.append({"offset": k, "mean_best_iou": float(np.mean(vals))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- MOT metrics
def mot_metrics(gt: dict, pred: dict, name: str = "tracker", iou_thr: float = 0.5) -> dict:
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for f in sorted(set(gt) | set(pred)):
        g, p = gt.get(f, []), pred.get(f, [])
        if g and p:
            iou = iou_matrix([b for _, b in g], [b for _, b in p])
            dist = np.where(iou >= iou_thr, 1 - iou, np.nan)
        else:
            dist = np.empty((len(g), len(p)))
        acc.update([i for i, _ in g], [i for i, _ in p], dist)
    s = mm.metrics.create().compute(acc, metrics=["mota", "motp", "idf1", "num_switches", "num_false_positives",
                                                  "num_misses", "mostly_tracked", "num_unique_objects"], name=name)
    r = s.iloc[0]
    return {"tracker": name, "MOTA": float(r["mota"]), "IDF1": float(r["idf1"]), "HOTA~": hota_approx(gt, pred),
            "IDSW": int(r["num_switches"]), "FP": int(r["num_false_positives"]), "FN": int(r["num_misses"]),
            "MT": int(r["mostly_tracked"]), "GT_ids": int(r["num_unique_objects"])}


def hota_approx(gt: dict, pred: dict) -> float:
    """HOTA averaged over IoU thresholds 0.05..0.95 with per-frame Hungarian matching.
    Simplified (no global score-based matching as in TrackEval) -> reported as HOTA~."""
    from scipy.optimize import linear_sum_assignment
    frames = sorted(set(gt) | set(pred))
    gt_n, pr_n = defaultdict(int), defaultdict(int)
    for f in frames:
        for i, _ in gt.get(f, []):
            gt_n[i] += 1
        for j, _ in pred.get(f, []):
            pr_n[j] += 1
    scores = []
    for a in np.linspace(0.05, 0.95, 19):
        tp = fp = fn = 0
        pair = defaultdict(int); matches = []
        for f in frames:
            g, p = gt.get(f, []), pred.get(f, [])
            if not g or not p:
                fp += len(p); fn += len(g)
                continue
            iou = iou_matrix([b for _, b in g], [b for _, b in p])
            r, c = linear_sum_assignment(-iou)
            m = iou[r, c] >= a
            tp += int(m.sum()); fn += len(g) - int(m.sum()); fp += len(p) - int(m.sum())
            for rr, cc in zip(r[m], c[m]):
                k = (g[rr][0], p[cc][0]); pair[k] += 1; matches.append(k)
        if tp == 0:
            scores.append(0.0)
            continue
        det_a = tp / (tp + fp + fn)
        ass_a = np.mean([pair[k] / (gt_n[k[0]] + pr_n[k[1]] - pair[k]) for k in matches])
        scores.append(float(np.sqrt(det_a * ass_a)))
    return float(np.mean(scores))


# --------------------------------------------------------------------------- Luvizon
def load_luvizon_xml(path: str | Path) -> pd.DataFrame:
    """One row per annotated vehicle. speed is the inductive-loop speed (km/h),
    plate_* is the licence plate box at frame `iframe`."""
    root = ET.parse(path).getroot()
    rows = []
    for v in root.iter("vehicle"):
        reg = v.find("region"); rad = v.find("radar")
        rows.append({
            "iframe": int(v.get("iframe")), "lane": int(v.get("lane")),
            "moto": v.get("moto") == "True", "has_speed": v.get("radar") == "True",
            "speed": float(rad.get("speed")) if rad is not None else np.nan,
            "plate_x": int(reg.get("x")), "plate_y": int(reg.get("y")),
            "plate_w": int(reg.get("w")), "plate_h": int(reg.get("h")),
        })
    return pd.DataFrame(rows)


def _track_speed_lsq(g: pd.DataFrame) -> tuple[float, int, float]:
    """Speed (km/h) from a straight-line LSQ fit of position vs time on one track."""
    if len(g) < 3:
        return np.nan, len(g), 0.0
    t = g["t"].to_numpy() - g["t"].to_numpy().mean()
    vx = np.polyfit(t, g["x"].to_numpy(), 1)[0]
    vy = np.polyfit(t, g["y"].to_numpy(), 1)[0]
    return float(np.hypot(vx, vy) * 3.6), len(g), float(t.max() - t.min())


def match_luvizon(gt: pd.DataFrame, points: pd.DataFrame, max_frame_gap: int = 2,
                  window_s: tuple = (0.0, 2.0), max_box_frac: float = 0.4,
                  image_area: float = 1920 * 1080) -> pd.DataFrame:
    """Match every GT vehicle (with a loop speed) to the predicted track whose box
    contains the plate centre at frame iframe (+-max_frame_gap), then measure the
    track speed on the valid ground points in [iframe + window_s[0], iframe + window_s[1]]."""
    pts = points
    by_frame = {f: g for f, g in pts.groupby("frame_id")}
    valid = pts[pts["in_zone"] & ~pts["edge"]]
    fps = float(pts["fps"].iloc[0])
    rows = []
    for gi, v in gt[gt["has_speed"]].iterrows():
        cx, cy = v.plate_x + v.plate_w / 2, v.plate_y + v.plate_h / 2
        best = None
        for df_ in range(0, max_frame_gap + 1):
            for f in {v.iframe - df_, v.iframe + df_}:
                g = by_frame.get(f)
                if g is None:
                    continue
                b = np.array(g["bbox_xyxy"].tolist())
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                # boxes covering a large part of the frame are false positives (merged vehicles)
                inside = (b[:, 0] <= cx) & (cx <= b[:, 2]) & (b[:, 1] <= cy) & (cy <= b[:, 3]) & (area < max_box_frac * image_area)
                if inside.any():
                    k = np.where(inside)[0][np.argmin(area[inside])]
                    best = int(g["track_id"].iloc[k])
                    break
            if best is not None:
                break
        row = {"gt_index": gi, "iframe": v.iframe, "lane": v.lane, "moto": v.moto, "gt_kmh": v.speed,
               "track_id": best, "pred_kmh": np.nan, "n_points": 0, "span_s": 0.0}
        if best is not None:
            t0 = v.iframe / fps
            g = valid[(valid["track_id"] == best) & (valid["t"] >= t0 + window_s[0]) & (valid["t"] <= t0 + window_s[1])]
            row["pred_kmh"], row["n_points"], row["span_s"] = _track_speed_lsq(g)
        rows.append(row)
    return pd.DataFrame(rows)


def speed_error_table(m: pd.DataFrame, min_points: int = 5) -> dict:
    ok = m[m["pred_kmh"].notna() & (m["n_points"] >= min_points)]
    e = ok["pred_kmh"] - ok["gt_kmh"]
    return {
        "gt_vehicles": int(len(m)), "measured": int(len(ok)), "coverage_%": 100.0 * len(ok) / max(1, len(m)),
        "MAE_kmh": float(e.abs().mean()) if len(ok) else np.nan,
        "RMSE_kmh": float(np.sqrt((e ** 2).mean())) if len(ok) else np.nan,
        "bias_kmh": float(e.mean()) if len(ok) else np.nan,
        "MAPE_%": float((e.abs() / ok["gt_kmh"]).mean() * 100) if len(ok) else np.nan,
        "within_-3_+2_%": float(((e >= -3) & (e <= 2)).mean() * 100) if len(ok) else np.nan,
    }


# --------------------------------------------------------------------------- track <-> GT matching
def match_tracks_iou(gt: dict, pred: dict, iou_thr: float = 0.3) -> dict:
    """Predicted track id -> GT track id by majority vote of per-frame best-IoU matches."""
    from collections import Counter
    votes = defaultdict(Counter)
    for f, p in pred.items():
        g = gt.get(f, [])
        if not g:
            continue
        iou = iou_matrix([b for _, b in p], [b for _, b in g])
        for i, (pid, _) in enumerate(p):
            j = int(iou[i].argmax())
            if iou[i, j] >= iou_thr:
                votes[pid][g[j][0]] += 1
    return {pid: c.most_common(1)[0][0] for pid, c in votes.items()}


def per_vehicle_speed(points: pd.DataFrame, min_points: int = 5, moving_kmh: float = 3.0) -> pd.Series:
    """Median window speed of each track over its valid in-zone points."""
    z = points[points["ema_kmh"].notna()]
    g = z.groupby("track_id")["win_kmh"]
    s = g.median()[g.size() >= min_points]
    return s[s >= moving_kmh]


def compare_vehicle_speeds(ref: pd.Series, est: pd.Series, mapping: dict | None = None) -> pd.DataFrame:
    """ref: GT track -> km/h. est: track -> km/h (mapped to GT ids through `mapping`)."""
    if mapping is not None:
        e = pd.Series({mapping[k]: v for k, v in est.items() if k in mapping})
        e = e.groupby(level=0).median()
    else:
        e = est
    df = pd.DataFrame({"ref_kmh": ref}).join(e.rename("est_kmh"), how="left")
    df["err_kmh"] = df["est_kmh"] - df["ref_kmh"]
    return df


def error_summary(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["est_kmh"])
    e = ok["err_kmh"]
    return {"vehicles": int(len(df)), "matched": int(len(ok)),
            "MAE_kmh": float(e.abs().mean()) if len(ok) else np.nan,
            "RMSE_kmh": float(np.sqrt((e ** 2).mean())) if len(ok) else np.nan,
            "bias_kmh": float(e.mean()) if len(ok) else np.nan,
            "MAPE_%": float((e.abs() / ok["ref_kmh"]).mean() * 100) if len(ok) else np.nan}
