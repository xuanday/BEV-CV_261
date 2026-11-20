"""BEV analytics for a tracks_*.jsonl file with a camera config.

python scripts/analyse_tracks.py --tracks outputs/tracks/tracks_danang_cam_01.jsonl \
    --camera configs/cameras/danang_lf_zmcao38w.yaml --video Data/danang/LF_ZmCAo38w_C001.mp4
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bevtraffic.analytics import AnalyticsConfig, analyse  # noqa: E402
from bevtraffic.calibration import Calibration  # noqa: E402
from bevtraffic.viz import draw_bev_grid, render_dashboard  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--video", default=None, help="raw video, only needed to render the dashboard")
    ap.add_argument("--background", default=None, help="image used under the congestion map")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="outputs/analytics")
    a = ap.parse_args()

    cal = Calibration.from_yaml(a.camera)
    recs = [json.loads(l) for l in open(a.tracks, encoding="utf-8")]
    res = analyse(recs, cal, AnalyticsConfig())
    name = a.name or Path(a.tracks).stem.replace("tracks_", "")
    out = Path(a.out) / name; out.mkdir(parents=True, exist_ok=True)
    for k in ("tracks", "crossings", "violations", "headway"):
        res[k].to_csv(out / f"{k}.csv", index=False)
    s = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in res["summary"].items()}
    (out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    if a.background:
        bev = cal.warp_bev(cv2.imread(a.background))
        c = res["congestion"]
        if c.max() > 0:
            heat = cv2.applyColorMap(np.uint8(255 * c / c.max()), cv2.COLORMAP_JET)
            m = c > 0.02 * c.max()
            bev[m] = cv2.addWeighted(bev, 0.45, heat, 0.55, 0)[m]
        cv2.imwrite(str(out / "bev_congestion.png"), draw_bev_grid(bev, cal))
    if a.video:
        n = render_dashboard(a.video, res, cal, out / "dashboard.mp4")
        s["rendered_frames"] = n
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
