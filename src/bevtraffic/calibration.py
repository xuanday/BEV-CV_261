"""Ground-plane calibration.

A camera config (YAML) lists image points (u, v) clicked on road markings whose
positions on the road are known in metres (x, y). The homography H maps image
pixels to road metres. Some points can be kept out of the fit ("check_points")
so the calibration error is measured on points the fit has not seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


def _as_pts(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1, 2)


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    pts = _as_pts(pts)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


@dataclass
class Lane:
    id: str
    polygon_m: list
    direction: tuple = (0.0, 1.0)          # unit vector of normal travel in world coords
    allowed: list | None = None             # None = every class allowed
    name: str = ""


@dataclass
class Tripwire:
    id: str
    p1: tuple
    p2: tuple
    direction: tuple | None = None          # count only crossings moving along this vector


@dataclass
class Calibration:
    camera_id: str
    image_size: tuple                       # (width, height)
    H: np.ndarray                           # image px -> world m
    px_per_m: float = 20.0
    x_range: tuple = (0.0, 20.0)            # BEV extent in metres
    y_range: tuple = (0.0, 40.0)
    zone_uv: np.ndarray | None = None       # measurement zone polygon in the image
    lanes: list = field(default_factory=list)
    tripwires: list = field(default_factory=list)
    fit_points: dict = field(default_factory=dict)
    check_points: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Calibration":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cal = cfg["calibration"]
        fit = {"img": [], "world": []}
        chk = {"img": [], "world": []}
        for p in cal.get("points", []):
            fit["img"].append(p[:2]); fit["world"].append(p[2:4])
        for p in cal.get("check_points", []):
            chk["img"].append(p[:2]); chk["world"].append(p[2:4])

        if "H" in cal:
            H = np.asarray(cal["H"], dtype=np.float64)
        else:
            if len(fit["img"]) < 4:
                raise ValueError(f"{path}: need >= 4 calibration points")
            H, _ = cv2.findHomography(_as_pts(fit["img"]), _as_pts(fit["world"]), 0)
        scale = float(cal.get("image_scale", 1.0))   # points clicked on a resized frame
        if scale != 1.0:
            S = np.diag([1.0 / scale, 1.0 / scale, 1.0])
            H = H @ S
            fit["img"] = (_as_pts(fit["img"]) * scale).tolist() if fit["img"] else []
            chk["img"] = (_as_pts(chk["img"]) * scale).tolist() if chk["img"] else []

        bev = cfg.get("bev", {})
        lanes = [Lane(id=str(l["id"]), polygon_m=l["polygon_m"], direction=tuple(l.get("direction", (0, 1))),
                      allowed=l.get("allowed"), name=l.get("name", "")) for l in cfg.get("lanes", [])]
        trips = [Tripwire(id=str(t["id"]), p1=tuple(t["p1"]), p2=tuple(t["p2"]),
                          direction=tuple(t["direction"]) if t.get("direction") else None)
                 for t in cfg.get("tripwires", [])]
        zone = cfg.get("measure_zone_uv")
        return cls(
            camera_id=cfg["camera_id"],
            image_size=tuple(cfg["image_size"]),
            H=H / H[2, 2],
            px_per_m=float(bev.get("px_per_m", 20.0)),
            x_range=tuple(bev.get("x_range", (0, 20))),
            y_range=tuple(bev.get("y_range", (0, 40))),
            zone_uv=np.asarray(zone, dtype=np.float32) if zone else None,
            lanes=lanes,
            tripwires=trips,
            fit_points=fit,
            check_points=chk,
            meta={k: cfg.get(k) for k in ("source", "notes", "fps", "reference")},
        )

    # ------------------------------------------------------------ mapping
    def to_world(self, uv) -> np.ndarray:
        return apply_h(self.H, uv)

    def to_image(self, xy) -> np.ndarray:
        return apply_h(np.linalg.inv(self.H), xy)

    @property
    def bev_size(self) -> tuple:
        w = int(round((self.x_range[1] - self.x_range[0]) * self.px_per_m))
        h = int(round((self.y_range[1] - self.y_range[0]) * self.px_per_m))
        return w, h

    @property
    def world_to_bev(self) -> np.ndarray:
        s = self.px_per_m
        return np.array([[s, 0, -s * self.x_range[0]], [0, s, -s * self.y_range[0]], [0, 0, 1]], dtype=np.float64)

    @property
    def H_bev(self) -> np.ndarray:
        """image px -> BEV image px"""
        return self.world_to_bev @ self.H

    def world_to_bev_px(self, xy) -> np.ndarray:
        return apply_h(self.world_to_bev, xy)

    def warp_bev(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.H_bev, self.bev_size)

    def in_zone(self, u: float, v: float) -> bool:
        if self.zone_uv is None:
            return True
        return cv2.pointPolygonTest(self.zone_uv, (float(u), float(v)), False) >= 0

    # ------------------------------------------------------------ quality
    def report(self) -> dict:
        """Residuals in metres on fit points and on held-out check points."""
        out = {"camera_id": self.camera_id, "n_fit": len(self.fit_points.get("img", [])),
               "n_check": len(self.check_points.get("img", []))}
        for name, pts in (("fit", self.fit_points), ("check", self.check_points)):
            if pts.get("img"):
                err = np.linalg.norm(self.to_world(pts["img"]) - _as_pts(pts["world"]), axis=1)
                out[f"{name}_rmse_m"] = float(np.sqrt(np.mean(err ** 2)))
                out[f"{name}_max_m"] = float(err.max())
                # same check in the image plane
                err_px = np.linalg.norm(self.to_image(pts["world"]) - _as_pts(pts["img"]), axis=1)
                out[f"{name}_rmse_px"] = float(np.sqrt(np.mean(err_px ** 2)))
        return out

    def metres_per_pixel(self, u: float, v: float) -> float:
        """Ground distance covered by one image pixel (vertical) at (u, v).
        Shows how fast precision drops towards the horizon."""
        a, b = self.to_world([[u, v], [u, v + 1]])
        return float(np.linalg.norm(a - b))
