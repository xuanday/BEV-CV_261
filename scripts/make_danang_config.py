"""Calibration of the Da Nang CCTV camera (nhthau/CCTV-giao-thong-da-nang, LF_ZmCAo38w_C00x).

No surveyed points are available, so the ground homography is built from two
vanishing points and one known width:

  1. VP1 (road direction): yellow centre line and right kerb, both parallel to the road.
  2. VP2 (across the road): the two rows of white transverse dashes.
  3. Square pixels + principal point at the image centre -> focal length from
     orthogonality: (v1 - c).(v2 - c) + f^2 = 0.
  4. R = [r_x r_y r_z] from K^-1 v1, K^-1 v2; ground normal = r_x x r_y.
  5. Camera height from a parked Toyota Fortuner (body width 1.855 m) seen at
     the bottom of the frame.

All image measurements were read on frame 0 of danang_cam_01 (1920x1080) and are
listed below so they can be checked or re-measured. The narrow transverse VP makes
f sensitive (see the printed sensitivity); speeds on this camera are approximate.

python scripts/make_danang_config.py
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

W, H = 1920, 1080
# lines as two points each (u, v)
CENTRE_LINE = [(930, 455), (1200, 735)]
RIGHT_KERB = [(1405, 440), (1880, 700)]
DASH_ROW_FAR = [(590, 438), (1350, 450)]
DASH_ROW_NEAR = [(830, 712), (1680, 720)]
FORTUNER_BUMPER = [(408, 900), (792, 900)]     # body edges just above the ground
FORTUNER_WIDTH_M = 1.855


def line(p, q):
    return np.cross([*p, 1.0], [*q, 1.0])


def vp(l1, l2):
    v = np.cross(line(*l1), line(*l2))
    return v[:2] / v[2]


def build(f_scale=1.0):
    c = np.array([W / 2, H / 2])
    v1, v2 = vp(CENTRE_LINE, RIGHT_KERB), vp(DASH_ROW_FAR, DASH_ROW_NEAR)
    f = np.sqrt(-np.dot(v1 - c, v2 - c)) * f_scale
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    ry = Ki @ [*v1, 1]; ry /= np.linalg.norm(ry)
    rx = Ki @ [*v2, 1]; rx -= ry * (rx @ ry); rx /= np.linalg.norm(rx)
    n = np.cross(rx, ry)
    # ground origin = point seen at the bottom centre of the image, camera height h = 1 for now
    ray = Ki @ [W / 2, H - 1, 1]
    if n @ ray < 0:
        n = -n
    O = ray / (n @ ray)
    G2I = K @ np.column_stack([rx, ry, O])            # ground (x, y, 1) -> image
    I2G = np.linalg.inv(G2I)
    p = lambda uv: (I2G @ [*uv, 1])[:2] / (I2G @ [*uv, 1])[2]
    width_units = np.linalg.norm(p(FORTUNER_BUMPER[0]) - p(FORTUNER_BUMPER[1]))
    h = FORTUNER_WIDTH_M / width_units
    Hm = np.diag([h, h, 1.0]) @ I2G
    # orient: +y towards the camera (traffic driving away has negative vy)
    a, b = (Hm @ [*CENTRE_LINE[0], 1]), (Hm @ [*CENTRE_LINE[1], 1])
    if (b[1] / b[2]) < (a[1] / a[2]):
        Hm = np.diag([1, -1, 1.0]) @ Hm
    return Hm / Hm[2, 2], {"f_px": float(f), "hfov_deg": float(np.degrees(2 * np.arctan(W / 2 / f))),
                          "camera_height_m": float(h), "vp_road": v1.round(1).tolist(), "vp_cross": v2.round(1).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/cameras/danang_lf_zmcao38w.yaml")
    a = ap.parse_args()
    Hm, info = build()
    pr = lambda H_, uv: (H_ @ [*uv, 1])[:2] / (H_ @ [*uv, 1])[2]
    zone = np.array([[600, 300], [1450, 300], [1919, 760], [1919, 1040], [380, 1040]], float)
    zw = np.array([pr(Hm, z) for z in zone])
    off = np.array([-zw[:, 0].min() + 1, -zw[:, 1].min() + 1])
    T = np.eye(3); T[:2, 2] = off
    Hm = T @ Hm
    zw = np.array([pr(Hm, z) for z in zone])

    # sensitivity of a 10 m road-direction distance to the focal length
    sens = {}
    for k in (0.8, 1.0, 1.25):
        Hk, _ = build(k)
        A, B = pr(Hk, CENTRE_LINE[0]), pr(Hk, CENTRE_LINE[1])
        sens[f"f x{k}"] = round(float(np.linalg.norm(A - B)), 2)
    centre = [pr(Hm, CENTRE_LINE[0]), pr(Hm, CENTRE_LINE[1])]
    xc = float(np.mean([c[0] for c in centre]))
    y_mid = float(np.median(zw[:, 1]))
    cfg = {
        "camera_id": "danang_lf_zmcao38w",
        "source": "HuggingFace nhthau/CCTV-giao-thong-da-nang, LF_ZmCAo38w_C001..C003 (same fixed camera)",
        "reference": "Two vanishing points + Fortuner width 1.855 m, see scripts/make_danang_config.py",
        "notes": f"approximate calibration: f={info['f_px']:.0f}px (HFOV {info['hfov_deg']:.1f} deg), "
                 f"camera height {info['camera_height_m']:.1f} m; centre-line segment length vs f: {sens}",
        "image_size": [W, H],
        "fps": 30.0,
        "calibration": {"H": Hm.round(10).tolist(), "vanishing_points": info},
        "bev": {"px_per_m": 10, "x_range": [0.0, float(np.ceil(zw[:, 0].max() + 1))],
                "y_range": [0.0, float(np.ceil(zw[:, 1].max() + 1))]},
        "measure_zone_uv": zone.tolist(),
        "lanes": [
            {"id": "away", "name": "right half (traffic moving away from the camera)",
             "polygon_m": [[xc, 0.0], [float(zw[:, 0].max()), 0.0], [float(zw[:, 0].max()), float(zw[:, 1].max())], [xc, float(zw[:, 1].max())]],
             "direction": [0.0, -1.0]},
            {"id": "towards", "name": "left half (towards the camera, parking strip on the far left)",
             "polygon_m": [[0.0, 0.0], [xc, 0.0], [xc, float(zw[:, 1].max())], [0.0, float(zw[:, 1].max())]],
             "direction": [0.0, 1.0]},
        ],
        "tripwires": [
            {"id": "away", "p1": [xc, y_mid], "p2": [float(zw[:, 0].max()), y_mid], "direction": [0.0, -1.0]},
            {"id": "towards", "p1": [0.0, y_mid], "p2": [xc, y_mid], "direction": [0.0, 1.0]},
        ],
    }
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(info, sens)
    print("zone in metres:", zw.round(1).tolist())


if __name__ == "__main__":
    main()
