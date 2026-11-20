"""ByteTrack vs BoT-SORT on VNTraffic (Zenodo 18195750), MOTA / IDF1 / HOTA~.

The GT frame offset is estimated from the data (best mean IoU), not assumed.
Result with the old loader (frame - 1) is kept for reference.

python scripts/eval_mot.py --weights weights/yolo26_vn_finetuned.pt
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bevtraffic.evaluation import estimate_frame_offset, load_mot_txt, mot_metrics, records_by_frame  # noqa: E402
from bevtraffic.tracking import track_video  # noqa: E402

VID = ROOT / "Data/VnTraffic/VNTraffic/VNTraffic_Original-video.mp4"
GT = ROOT / "Data/VnTraffic/VNTraffic/VNTraffic_GroundTruth.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo26_vn_finetuned.pt")
    ap.add_argument("--tracks-dir", default="outputs/tracks")
    ap.add_argument("--out", default="outputs/mot")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tdir = Path(a.tracks_dir); tdir.mkdir(parents=True, exist_ok=True)

    rows, offsets = [], None
    for tracker in ("bytetrack_vn", "botsort_vn"):
        tp = tdir / f"tracks_vntraffic_{Path(a.weights).stem}_{tracker}.jsonl"
        if tp.exists():
            recs = [json.loads(l) for l in open(tp, encoding="utf-8")]
            info = {}
        else:
            recs, info = track_video(a.weights, VID, "vntraffic", tracker=tracker, device=a.device,
                                     tracker_dir=out / "trackers")
            with open(tp, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        pred = records_by_frame(recs)
        if offsets is None:
            offsets = estimate_frame_offset(load_mot_txt(GT, 0), pred)
            offsets.to_csv(out / "gt_frame_offset.csv", index=False)
            k = int(offsets.loc[offsets["mean_best_iou"].idxmax(), "offset"])
        for label, off in ((f"GT offset {k:+d} (estimated)", k), ("GT offset -1 (old loader)", -1)):
            r = mot_metrics(load_mot_txt(GT, off), pred, name=tracker)
            r.update({"gt_reading": label, "proc_fps_cpu": info.get("fps_proc")})
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "mot_vntraffic.csv", index=False)
    print(offsets.to_string(index=False))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
