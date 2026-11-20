"""Speed evaluation on the Luvizon et al. (UTFPR) dataset, subset01.

Ground truth: inductive-loop speed per vehicle (vehicles.xml). Geometry: the
rectifying homography shipped with the dataset (dataset/matrix.txt in
github.com/dluvizon/vehicle-dsm, middle lane). The rectified plane is metric up
to one unknown factor (metres per rectified unit). Two ways to get it:

  * lane     : assume the middle lane is 3.5 m wide (no ground truth used)
  * fit      : fit the factor on video01 vehicles, then test on video02 only

For comparison the old pipeline geometry (default trapezoid, 3 x 3.5 m x 35 m)
is evaluated on the same vehicles.

python scripts/eval_luvizon.py --tracks-dir ../data_local/tracks --xml-dir ../data_local/luvizon
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bevtraffic.analytics import AnalyticsConfig, add_speed, project  # noqa: E402
from bevtraffic.calibration import Calibration, apply_h  # noqa: E402
from bevtraffic.evaluation import load_luvizon_xml, match_luvizon, speed_error_table  # noqa: E402

CAM = ROOT / "configs" / "cameras" / "luvizon_subset01.yaml"
OLD_TRAPEZOID = [[0.36, 0.28], [0.64, 0.28], [0.92, 0.96], [0.08, 0.96]]


def load_tracks(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def calib_with_scale(base: dict, s: float) -> Calibration:
    cfg = yaml.safe_load(CAM.read_text(encoding="utf-8"))
    H_rect = np.asarray(cfg["calibration"]["H_rect"], float)
    cal = Calibration.from_yaml(CAM)
    cal.H = np.diag([s, s, 1.0]) @ H_rect
    return cal


def old_pipeline_calib(w=1920, h=1080) -> Calibration:
    import cv2
    src = np.array([[x * w, y * h] for x, y in OLD_TRAPEZOID])
    dst = np.array([[2, 2], [12.5, 2], [12.5, 37], [2, 37]], float)
    H, _ = cv2.findHomography(src, dst, 0)
    return Calibration(camera_id="old_default", image_size=(w, h), H=H, zone_uv=src.astype(np.float32))


def measure(records, cal, gt):
    acfg = AnalyticsConfig()
    pts = add_speed(project(records, cal, acfg), acfg)
    return match_luvizon(gt, pts)


def lane_width_units(H_rect, lanes_xml_pts) -> float:
    """Width of the middle lane in rectified units, averaged along the lane."""
    (tl, tr, bl, br) = [np.array(p, float) for p in lanes_xml_pts]
    ws = []
    for a in np.linspace(0.1, 0.9, 9):
        l = tl + a * (bl - tl); r = tr + a * (br - tr)
        L, R = apply_h(H_rect, [l, r])
        ws.append(abs(R[0] - L[0]))
    return float(np.mean(ws))


def write_fitted_config(cfg: dict, H_rect: np.ndarray, s: float):
    """Store H = diag(s, s, 1) H_rect in the camera config and size the BEV to the zone."""
    H = np.diag([s, s, 1.0]) @ H_rect
    zw = apply_h(H, cfg["measure_zone_uv"])
    x0, y0 = np.floor(zw.min(0)) - 1
    x1, y1 = np.ceil(zw.max(0)) + 1
    cfg["calibration"]["H"] = H.round(10).tolist()
    cfg["calibration"]["scale_m_per_unit"] = round(s, 8)
    cfg["bev"] = {"px_per_m": 20, "x_range": [float(x0), float(x1)], "y_range": [float(y0), float(y1)]}
    CAM.write_text(yaml.safe_dump(cfg, sort_keys=False, width=120), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", default="outputs/tracks")
    ap.add_argument("--xml-dir", default="Data/luvizon")
    ap.add_argument("--weights-tag", default="yolo26_vn_finetuned_bytetrack_vn")
    ap.add_argument("--out", default="outputs/luvizon")
    ap.add_argument("--scale-from", default=None,
                    help="reuse the factor of an earlier run (scale.json) instead of fitting it again")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(CAM.read_text(encoding="utf-8"))
    H_rect = np.asarray(cfg["calibration"]["H_rect"], float)
    w_units = lane_width_units(H_rect, cfg["calibration"]["middle_lane_uv"])
    s_lane = 3.5 / w_units

    videos = {}
    for v in ("v01", "v02"):
        tp = Path(a.tracks_dir) / f"tracks_luvizon_{v}_{a.weights_tag}.jsonl"
        xp = Path(a.xml_dir) / f"{v}_vehicles.xml"
        if tp.exists() and xp.exists():
            videos[v] = (load_tracks(tp), load_luvizon_xml(xp))
    if "v01" not in videos:
        sys.exit("need tracks for video01")

    # 1) measure video01 with the lane-width scale, then correct the factor with the
    #    median GT/pred ratio (speed is linear in s, so this is the same as fitting s)
    m1 = measure(videos["v01"][0], calib_with_scale(cfg, s_lane), videos["v01"][1])
    ok = m1[m1["pred_kmh"].notna() & (m1["n_points"] >= 5)]
    ratio = s_lane * ok["gt_kmh"] / ok["pred_kmh"]
    s_fit = float(np.median(ratio))
    if a.scale_from:
        s_fit = json.loads(Path(a.scale_from).read_text())["s_fit_m_per_unit"]

    rows, per_vehicle = [], []
    for vid, (recs, gt) in videos.items():
        split = "fit (train)" if vid == "v01" else "test"
        for name, cal in (("old pipeline (default trapezoid)", old_pipeline_calib()),
                          (f"dataset H + lane width 3.5 m (s={s_lane:.5f})", calib_with_scale(cfg, s_lane)),
                          (f"dataset H + factor fitted on v01 (s={s_fit:.5f})", calib_with_scale(cfg, s_fit))):
            m = measure(recs, cal, gt)
            r = {"video": vid, "split": split, "geometry": name, **speed_error_table(m)}
            rows.append(r)
            per_vehicle.append(m.assign(video=vid, geometry=name))
    res = pd.DataFrame(rows).assign(tracks=a.weights_tag)
    tag = a.weights_tag.replace("_bytetrack_vn", "")
    pd.concat(per_vehicle).to_csv(out / f"per_vehicle_{tag}.csv", index=False)
    res.round(3).to_csv(out / f"speed_eval_{tag}.csv", index=False)
    meta = {"tracks": a.weights_tag, "scale_from": a.scale_from, "lane_width_units": w_units, "s_lane_m_per_unit": s_lane, "s_fit_m_per_unit": s_fit,
            "s_fit_iqr": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))], "n_fit": int(len(ok)),
            "implied_lane_width_m": s_fit * w_units}
    if not a.scale_from:
        (out / "scale.json").write_text(json.dumps(meta, indent=2))
        write_fitted_config(cfg, H_rect, s_fit)
    pd.set_option("display.width", 220)
    print(json.dumps(meta, indent=2))
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
