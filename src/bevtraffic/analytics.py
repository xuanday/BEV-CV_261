"""Spatial analytics on the BEV ground plane.

Input: track records (one row per box per frame) + a Calibration.
All distances are metres in world coordinates, all times are seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from .calibration import Calibration
from .tracking import majority_class

DEFAULT_LENGTH_M = {"motorbike": 2.0, "car": 4.5, "bus": 10.5, "truck": 7.5}


@dataclass
class AnalyticsConfig:
    speed_window_s: float = 0.8      # regression window for speed
    ema_alpha: float = 0.3           # extra smoothing on top of the window estimate
    min_points: int = 3              # points needed in the window
    edge_margin_px: float = 3.0      # boxes touching the frame border have a wrong footpoint
    max_step_kmh: float = 250.0      # single steps faster than this are id switches -> dropped
    max_box_frac: float = 0.4        # boxes covering this much of the frame are merged false positives
    moving_kmh: float = 5.0
    congestion_kmh: float = 10.0
    thw_min_speed_kmh: float = 20.0
    thw_safe_s: float = 1.8
    thw_max_lateral_m: float = 1.5
    lane_margin_m: float = 0.3
    debounce_s: float = 0.8
    vehicle_length_m: dict = field(default_factory=lambda: dict(DEFAULT_LENGTH_M))


# --------------------------------------------------------------------------- projection
def project(records: list, cal: Calibration, cfg: AnalyticsConfig) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["t"] = df["timestamp_ms"] / 1000.0
    cls = majority_class(records)
    df["cls"] = df["track_id"].map(cls)
    b = np.array(df["bbox_xyxy"].tolist())
    W, H = cal.image_size
    m = cfg.edge_margin_px
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    # "edge" = footpoint not usable: box cut by the frame border, or an oversized merged box
    df["edge"] = (b[:, 0] <= m) | (b[:, 2] >= W - m) | (b[:, 3] >= H - m) | (area > cfg.max_box_frac * W * H)
    uv = np.array(df["footpoint_uv"].tolist(), dtype=np.float64)
    df["u"], df["v"] = uv[:, 0], uv[:, 1]
    df["in_zone"] = [cal.in_zone(u, v) for u, v in uv]
    xy = cal.to_world(uv)
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    return df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- speed
def _window_velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, win: float, kmin: int):
    """Least-squares velocity over the trailing window [t_i - win, t_i]."""
    n = len(t)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    j = 0
    for i in range(n):
        while t[i] - t[j] > win:
            j += 1
        if i - j + 1 >= kmin:
            tt = t[j:i + 1] - t[i]
            A = np.vstack([tt, np.ones_like(tt)]).T
            (vx[i], _), *_ = np.linalg.lstsq(A, x[j:i + 1], rcond=None)
            (vy[i], _), *_ = np.linalg.lstsq(A, y[j:i + 1], rcond=None)
    return vx, vy


def _kalman_speed(t, x, y, q=3.0, r=0.15):
    """Constant-velocity Kalman filter with the real dt between observations."""
    n = len(t)
    out = np.full(n, np.nan)
    s = np.array([x[0], y[0], 0.0, 0.0]); P = np.diag([r, r, 25.0, 25.0])
    Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float); R = np.eye(2) * r
    for i in range(1, n):
        dt = max(t[i] - t[i - 1], 1e-3)
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        s = F @ s; P = F @ P @ F.T + G @ G.T * q
        z = np.array([x[i], y[i]])
        K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + R)
        s = s + K @ (z - Hm @ s); P = (np.eye(4) - K @ Hm) @ P
        out[i] = np.hypot(s[2], s[3]) * 3.6
    return out


def add_speed(df: pd.DataFrame, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Adds raw_kmh (frame-to-frame), win_kmh (window regression), ema_kmh, kf_kmh, heading."""
    df = df.copy()
    for col in ("raw_kmh", "win_kmh", "ema_kmh", "kf_kmh", "hx", "hy"):
        df[col] = np.nan
    use = df["in_zone"] & ~df["edge"]
    parts = []
    for tid, g in df[use].groupby("track_id", sort=False):
        g = g.sort_values("t")
        t, x, y = g["t"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy()
        # drop teleport steps (usually an id switch or a bad footpoint)
        if len(t) > 1:
            step = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3) * 3.6
            ok = np.concatenate([[True], step < cfg.max_step_kmh])
            if not ok.all():
                parts.append(g[~ok].assign(edge=True))   # keep the row, but not as a ground point
            g, t, x, y = g[ok], t[ok], x[ok], y[ok]
        raw = np.full(len(t), np.nan)
        if len(t) > 1:
            raw[1:] = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3) * 3.6
        vx, vy = _window_velocity(t, x, y, cfg.speed_window_s, cfg.min_points)
        win = np.hypot(vx, vy) * 3.6
        ema = np.full(len(t), np.nan)
        prev = np.nan
        for i, w in enumerate(win):
            if np.isnan(w):
                continue
            prev = w if np.isnan(prev) else cfg.ema_alpha * w + (1 - cfg.ema_alpha) * prev
            ema[i] = prev
        sp = np.hypot(vx, vy)
        g = g.assign(raw_kmh=raw, win_kmh=win, ema_kmh=ema,
                     kf_kmh=_kalman_speed(t, x, y) if len(t) > 1 else np.nan,
                     hx=np.where(sp > 1e-6, vx / np.where(sp > 1e-6, sp, 1), np.nan),
                     hy=np.where(sp > 1e-6, vy / np.where(sp > 1e-6, sp, 1), np.nan))
        parts.append(g)
    rest = df[~use]
    out = pd.concat(parts + [rest]) if parts else df
    return out.sort_values(["track_id", "frame_id"]).reset_index(drop=True)


def track_summary(df: pd.DataFrame, cfg: AnalyticsConfig) -> pd.DataFrame:
    z = df[df["in_zone"] & ~df["edge"]]
    rows = []
    for tid, g in z.groupby("track_id"):
        path = float(np.sum(np.hypot(np.diff(g["x"]), np.diff(g["y"])))) if len(g) > 1 else 0.0
        disp = float(np.hypot(g["x"].iloc[-1] - g["x"].iloc[0], g["y"].iloc[-1] - g["y"].iloc[0]))
        dur = float(g["t"].iloc[-1] - g["t"].iloc[0])
        sp = g["ema_kmh"].dropna()
        rows.append({"track_id": tid, "cls": g["cls"].iloc[0], "n": len(g), "t0": float(g["t"].iloc[0]),
                     "duration_s": dur, "displacement_m": disp, "path_m": path,
                     "speed_median_kmh": float(sp.median()) if len(sp) else np.nan,
                     "speed_mean_disp_kmh": disp / dur * 3.6 if dur > 0 else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- counting
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1, p2, q1, q2) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def count_crossings(df: pd.DataFrame, cal: Calibration) -> pd.DataFrame:
    """One event per (track, tripwire): the first step of the trajectory that crosses
    the line in the allowed direction. Steps across the zone border are ignored."""
    z = df[df["in_zone"] & ~df["edge"]]
    ev = []
    for tid, g in z.groupby("track_id"):
        p = g[["x", "y"]].to_numpy(); f = g["frame_id"].to_numpy(); t = g["t"].to_numpy()
        for tw in cal.tripwires:
            for k in range(1, len(p)):
                if _segments_intersect(p[k - 1], p[k], tw.p1, tw.p2):
                    d = p[k] - p[k - 1]
                    if tw.direction is not None and float(np.dot(d, tw.direction)) <= 0:
                        continue
                    ev.append({"track_id": tid, "tripwire": tw.id, "cls": g["cls"].iloc[0],
                               "frame_id": int(f[k]), "t": float(t[k])})
                    break
    return pd.DataFrame(ev, columns=["track_id", "tripwire", "cls", "frame_id", "t"])


# --------------------------------------------------------------------------- lanes
def _in_poly(x, y, poly) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, dtype=np.float32), (float(x), float(y)), False) >= 0


def _shrink(poly, m):
    p = np.asarray(poly, dtype=np.float64)
    c = p.mean(0)
    # move every vertex m metres towards the centroid along x and y separately (rectangular lanes)
    return np.stack([p[:, 0] - np.sign(p[:, 0] - c[0]) * m, p[:, 1] - np.sign(p[:, 1] - c[1]) * m], 1)


def assign_lanes(df: pd.DataFrame, cal: Calibration, cfg: AnalyticsConfig) -> pd.DataFrame:
    df = df.copy()
    df["lane"] = None
    df["lane_core"] = False
    if not cal.lanes:
        return df
    lanes, cores = [], []
    for ln in cal.lanes:
        lanes.append(ln); cores.append(_shrink(ln.polygon_m, cfg.lane_margin_m))
    lane_col, core_col = [], []
    for x, y, ok in zip(df["x"], df["y"], df["in_zone"] & ~df["edge"]):
        lid, core = None, False
        if ok:
            for ln, cp in zip(lanes, cores):
                if _in_poly(x, y, ln.polygon_m):
                    lid, core = ln.id, _in_poly(x, y, cp)
                    break
        lane_col.append(lid); core_col.append(core)
    df["lane"], df["lane_core"] = lane_col, core_col
    return df


def lane_violations(df: pd.DataFrame, cal: Calibration, cfg: AnalyticsConfig) -> pd.DataFrame:
    """A track violates a lane rule when it stays inside the lane core for at least
    debounce_s while its (majority) class is not allowed there, or drives against
    the lane direction for that long."""
    rules = {ln.id: ln for ln in cal.lanes}
    ev = []
    for tid, g in df[df["lane"].notna()].groupby("track_id"):
        cls = g["cls"].iloc[0]
        for lid, gl in g.groupby("lane"):
            ln = rules[lid]
            bad_cls = ln.allowed is not None and cls not in ln.allowed
            wrong_way = (gl["hx"] * ln.direction[0] + gl["hy"] * ln.direction[1] < -0.5) & (gl["ema_kmh"] > 5)
            for kind, mask in (("class", gl["lane_core"] & bad_cls), ("wrong_way", gl["lane_core"] & wrong_way)):
                if not mask.any():
                    continue
                t = gl["t"].to_numpy(); m = mask.to_numpy()
                best, start = 0.0, None
                for i in range(len(t)):
                    if m[i]:
                        start = t[i] if start is None else start
                        best = max(best, t[i] - start)
                    else:
                        start = None
                if best >= cfg.debounce_s:
                    ev.append({"track_id": tid, "lane": lid, "cls": cls, "kind": kind, "duration_s": round(best, 2)})
    return pd.DataFrame(ev, columns=["track_id", "lane", "cls", "kind", "duration_s"])


# --------------------------------------------------------------------------- headway
def headway(df: pd.DataFrame, cal: Calibration, cfg: AnalyticsConfig) -> pd.DataFrame:
    """Time headway between consecutive vehicles in the same lane and direction.
    Returns one row per (follower, leader) pair that went below the safe THW,
    with the minimum THW and how long it lasted."""
    if not cal.lanes:
        return pd.DataFrame(columns=["follower", "leader", "lane", "min_thw_s", "frames", "follower_kmh"])
    dirs = {ln.id: np.asarray(ln.direction, float) for ln in cal.lanes}
    L = cfg.vehicle_length_m
    pairs = {}
    d = df[df["lane"].notna() & df["ema_kmh"].notna()]
    for (fid, lid), g in d.groupby(["frame_id", "lane"]):
        if len(g) < 2:
            continue
        u = dirs[lid] / np.linalg.norm(dirs[lid])
        nrm = np.array([-u[1], u[0]])
        p = g[["x", "y"]].to_numpy()
        s = p @ u; lat = p @ nrm
        order = np.argsort(s)
        rows = g.iloc[order]; s = s[order]; lat = lat[order]
        for i in range(len(rows) - 1):
            f, l = rows.iloc[i], rows.iloc[i + 1]
            if abs(lat[i + 1] - lat[i]) > cfg.thw_max_lateral_m or f["ema_kmh"] < cfg.thw_min_speed_kmh:
                continue
            gap = (s[i + 1] - s[i]) - 0.5 * (L.get(f["cls"], 4.5) + L.get(l["cls"], 4.5))
            thw = max(gap, 0.0) / (f["ema_kmh"] / 3.6)
            if thw < cfg.thw_safe_s:
                k = (int(f["track_id"]), int(l["track_id"]))
                e = pairs.setdefault(k, {"follower": k[0], "leader": k[1], "lane": lid, "min_thw_s": thw,
                                         "frames": 0, "follower_kmh": float(f["ema_kmh"])})
                e["frames"] += 1
                if thw < e["min_thw_s"]:
                    e["min_thw_s"], e["follower_kmh"] = thw, float(f["ema_kmh"])
    return pd.DataFrame(list(pairs.values()), columns=["follower", "leader", "lane", "min_thw_s", "frames", "follower_kmh"])


# --------------------------------------------------------------------------- congestion
def congestion_map(df: pd.DataFrame, cal: Calibration, cfg: AnalyticsConfig, sigma_m: float = 1.0) -> np.ndarray:
    """Dwell time (seconds) of slow vehicles per BEV pixel, blurred."""
    w, h = cal.bev_size
    grid = np.zeros((h, w), np.float32)
    d = df[df["ema_kmh"].notna() & (df["ema_kmh"] < cfg.congestion_kmh)]
    if len(d):
        fps = float(d["fps"].iloc[0]) if "fps" in d else 30.0
        px = cal.world_to_bev_px(d[["x", "y"]].to_numpy()).round().astype(int)
        ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        np.add.at(grid, (px[ok, 1], px[ok, 0]), 1.0 / fps)
    k = max(3, int(round(sigma_m * cal.px_per_m * 3)) | 1)
    return cv2.GaussianBlur(grid, (k, k), sigma_m * cal.px_per_m)


# --------------------------------------------------------------------------- all
def analyse(records: list, cal: Calibration, cfg: AnalyticsConfig | None = None) -> dict:
    cfg = cfg or AnalyticsConfig()
    df = project(records, cal, cfg)
    if df.empty:
        return {"points": df}
    df = add_speed(df, cfg)
    df = assign_lanes(df, cal, cfg)
    tracks = track_summary(df, cfg)
    cross = count_crossings(df, cal)
    viol = lane_violations(df, cal, cfg)
    thw = headway(df, cal, cfg)
    moving = df["ema_kmh"] >= cfg.moving_kmh
    summary = {
        "camera_id": cal.camera_id,
        "frames": int(df["frame_id"].max()) + 1,
        "records": int(len(df)),
        "tracks": int(df["track_id"].nunique()),
        "tracks_in_zone": int(len(tracks)),
        "tracks_moved_5m": int((tracks["displacement_m"] > 5).sum()) if len(tracks) else 0,
        "edge_boxes": int(df["edge"].sum()),
        "counted": int(cross["track_id"].nunique()) if len(cross) else 0,
        "counts_by_tripwire": cross.groupby("tripwire").size().to_dict() if len(cross) else {},
        "counts_by_class": cross.drop_duplicates("track_id")["cls"].value_counts().to_dict() if len(cross) else {},
        "speed_moving_mean_kmh": float(df.loc[moving, "ema_kmh"].mean()) if moving.any() else np.nan,
        "speed_moving_p85_kmh": float(df.loc[moving, "ema_kmh"].quantile(0.85)) if moving.any() else np.nan,
        "lane_violations": int(len(viol)),
        "thw_pairs": int(len(thw)),
    }
    return {"points": df, "tracks": tracks, "crossings": cross, "violations": viol, "headway": thw,
            "congestion": congestion_map(df, cal, cfg), "summary": summary}
