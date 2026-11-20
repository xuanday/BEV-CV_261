"""Run detector + tracker on a video and save tracks_<camera>.jsonl.

python scripts/track.py --video Data/luvizon/video01.mp4 --camera luvizon_v01 \
    --weights weights/yolo26_vn_finetuned.pt --fps 30.15 --stride 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bevtraffic.tracking import track_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--weights", default="weights/yolo26_vn_finetuned.pt")
    ap.add_argument("--coco", action="store_true", help="weights are a COCO model (keep motorcycle/car/bus/truck)")
    ap.add_argument("--tracker", default="bytetrack_vn")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/tracks")
    a = ap.parse_args()

    recs, info = track_video(a.weights, a.video, a.camera, tracker=a.tracker, coco=a.coco, fps=a.fps,
                             vid_stride=a.stride, max_frames=a.max_frames, imgsz=a.imgsz, device=a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = Path(a.weights).stem
    p = out / f"tracks_{a.camera}_{tag}_{a.tracker}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"out": str(p), "records": len(recs), **info}))


if __name__ == "__main__":
    main()
