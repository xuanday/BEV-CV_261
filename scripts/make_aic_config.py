"""Build configs/cameras/aicc22_s01_c002.yaml from the AI City Challenge calibration.

calibration.txt of CityFlow S01/c002 gives H with [u v 1]^T ~ H [lat lon 1]^T for the
1920x1080 video. AICC22-Custom is the same camera resized to 1280x720. We invert H,
convert (lat, lon) to local east/north metres (equirectangular, fine over ~100 m),
then rotate so the inbound road runs along +y in the BEV.

python scripts/make_aic_config.py --calib Data/aic/S01_c002_calibration.txt
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

R_EARTH = 6378137.0


def read_aic(path):
    m = re.search(r"Homography matrix:\s*(.*)", Path(path).read_text())
    return np.array([[float(x) for x in row.split()] for row in m.group(1).split(";")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="Data/aic/S01_c002_calibration.txt")
    ap.add_argument("--out", default="configs/cameras/aicc22_s01_c002.yaml")
    a = ap.parse_args()

    H_gps = read_aic(a.calib)
    S = np.diag([1920 / 1280, 1080 / 720, 1.0])          # 720p pixel -> 1080p pixel
    Hi = np.linalg.inv(H_gps) @ S                         # 720p pixel -> (lat, lon)
    p = lambda H, pts: cv2.perspectiveTransform(np.asarray(pts, float).reshape(-1, 1, 2), H).reshape(-1, 2)

    lat0, lon0 = p(Hi, [[640, 600]])[0]
    kx, ky = np.cos(np.radians(lat0)) * np.pi / 180 * R_EARTH, np.pi / 180 * R_EARTH
    E = np.array([[0, kx, -kx * lon0], [ky, 0, -ky * lat0], [0, 0, 1]])   # (lat, lon) -> (east, north)
    H_en = E @ Hi

    # road axis from the white lane line next to the "ONLY" arrows (far -> near)
    a_far, a_near = p(H_en, [[672, 178], [690, 250]])
    d = (a_near - a_far) / np.linalg.norm(a_near - a_far)
    # rotate so the travel direction towards the camera becomes +y, then shift
    Rm = np.array([[d[1], -d[0]], [d[0], d[1]]])
    T = np.eye(3); T[:2, :2] = Rm
    H = T @ H_en
    zone = np.array([[450, 140], [920, 140], [1260, 718], [40, 718]], float)
    zw = p(H, zone)
    off = np.array([-zw[:, 0].min() + 2, -zw[:, 1].min() + 2])
    T2 = np.eye(3); T2[:2, 2] = off
    H = T2 @ H
    H /= H[2, 2]
    zw = p(H, zone)

    stop_bar = p(H, [[655, 252], [820, 265]])
    cfg = {
        "camera_id": "aicc22_s01_c002",
        "source": "AICC22-Custom (Zenodo 18195750) = AI City Challenge CityFlow S01/c002 resized to 1280x720",
        "reference": "Calibration from AI City Challenge calibration.txt (GPS homography, reported reprojection error 11.4 px at 1080p)",
        "image_size": [1280, 720],
        "fps": 10.0,
        "calibration": {"H": H.round(10).tolist()},
        "bev": {"px_per_m": 10, "x_range": [0.0, float(np.ceil(zw[:, 0].max() + 2))],
                "y_range": [0.0, float(np.ceil(zw[:, 1].max() + 2))]},
        "measure_zone_uv": zone.tolist(),
        # horizontal line through the stop bar of the inbound lane, across the whole zone
        "tripwires": [{"id": "stop_bar", "p1": [round(float(zw[:, 0].min()), 2), round(float(stop_bar[:, 1].mean()), 2)],
                       "p2": [round(float(zw[:, 0].max()), 2), round(float(stop_bar[:, 1].mean()), 2)],
                       "direction": [0.0, 1.0]}],
    }
    Path(a.out).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
