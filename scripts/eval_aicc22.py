"""AICC22-Custom: speed/count evaluation with an independent calibration.

Reference geometry: AI City Challenge GPS calibration of S01/c002
(configs/cameras/aicc22_s01_c002.yaml, built by make_aic_config.py).
Reference speeds: the human-annotated GT boxes projected with that geometry.

Compares
  1. geometry error : GT boxes through the old hand-guessed trapezoid (3 lanes x 3.66 m x 24 m)
  2. detector/tracker error : predicted tracks through the reference geometry
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bevtraffic.analytics import analyse  # noqa: E402
from bevtraffic.calibration import Calibration  # noqa: E402
from bevtraffic.evaluation import (compare_vehicle_speeds, error_summary, estimate_frame_offset, gt_to_records,  # noqa: E402
                                   load_mot_txt, match_tracks_iou, per_vehicle_speed, records_by_frame)

DATA = ROOT / "Data" / "VnTraffic" / "AICC22-Custom"
CAM = ROOT / "configs" / "cameras" / "aicc22_s01_c002.yaml"


def old_calibration(ref: Calibration) -> Calibration:
    src = np.array([[535, 165], [825, 165], [1120, 700], [100, 700]], float)
    dst = np.array([[2.5, 2.0], [13.48, 2.0], [13.48, 26.0], [2.5, 26.0]])
    H, _ = cv2.findHomography(src, dst, 0)
    return Calibration(camera_id="aicc22_old_trapezoid", image_size=(1280, 720), H=H, zone_uv=ref.zone_uv,
                       x_range=(0, 16), y_range=(0, 28))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks-dir", default="../data_local/tracks")
    ap.add_argument("--out", default="outputs/aicc22")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cal = Calibration.from_yaml(CAM)
    old = old_calibration(cal)
    fps = 10.0
    preds = {}
    for tag, name in (("finetuned", "yolo26_vn_finetuned"), ("coco", "yolo26s")):
        p = Path(a.tracks_dir) / f"tracks_aicc22_{name}_bytetrack_vn.jsonl"
        if p.exists():
            preds[tag] = [json.loads(l) for l in open(p, encoding="utf-8")]

    # frame offset of the GT file (it starts at frame 3 and ends at 200 for a 200-frame video)
    raw = load_mot_txt(DATA / "AICC22-Custom_GroundTruth.txt", 0)
    off = estimate_frame_offset(raw, records_by_frame(preds["finetuned"]))
    off.to_csv(out / "gt_frame_offset.csv", index=False)
    k = int(off.loc[off["mean_best_iou"].idxmax(), "offset"])
    gt = load_mot_txt(DATA / "AICC22-Custom_GroundTruth.txt", k)
    gt_recs = gt_to_records(gt, fps, "aicc22_gt")

    res_ref = analyse(gt_recs, cal)
    ref = per_vehicle_speed(res_ref["points"])
    rows, detail = [], []

    res_old = analyse(gt_recs, old)
    d = compare_vehicle_speeds(ref, per_vehicle_speed(res_old["points"]))
    rows.append({"what": "GT boxes, old trapezoid geometry", **error_summary(d)}); detail.append(d.assign(what="old_geometry"))

    for tag, recs in preds.items():
        r = analyse(recs, cal)
        mp = match_tracks_iou(gt, records_by_frame(recs))
        d = compare_vehicle_speeds(ref, per_vehicle_speed(r["points"]), mp)
        rows.append({"what": f"{tag} tracks, AIC geometry", **error_summary(d),
                     "counted": r["summary"]["counted"], "tracks": r["summary"]["tracks"]})
        detail.append(d.assign(what=tag))
        r_old = analyse(recs, old)
        rows.append({"what": f"{tag} tracks, old geometry (old pipeline)",
                     **error_summary(compare_vehicle_speeds(ref, per_vehicle_speed(r_old["points"]), mp))})

    summ = pd.DataFrame(rows)
    summ.round(2).to_csv(out / "speed_eval.csv", index=False)
    pd.concat(detail).round(2).to_csv(out / "per_vehicle.csv")
    info = {"gt_frame_offset": k, "reference_vehicles": int(len(ref)),
            "reference_speed_median_kmh": float(ref.median()), "gt_counted_stop_bar": res_ref["summary"]["counted"],
            "reference_moving_mean_kmh": res_ref["summary"]["speed_moving_mean_kmh"],
            "old_geometry_moving_mean_kmh": res_old["summary"]["speed_moving_mean_kmh"]}
    (out / "summary.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 200)
    print(off.to_string(index=False)); print(json.dumps(info, indent=2)); print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
