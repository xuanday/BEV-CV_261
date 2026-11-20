"""Download Luvizon et al. subset01 video01/video02 (+ vehicles.xml) and remux to mp4.

The raw .h264 streams carry no timing, so they are remuxed at 30.15 fps (dataset
description). Needs ~1.7 GB. ffmpeg comes from the imageio-ffmpeg package.

python scripts/get_luvizon.py --out Data/luvizon
"""
import argparse
import subprocess
import urllib.request
from pathlib import Path

# Google Drive file ids (folder "Part 1" linked from github.com/dluvizon/vehicle-dsm)
FILES = {
    "video01": {"xml": "1cx43sVS-eDDEW-A7ZcgukKQdSe-i9MKb", "h264": "16Jd00kzXN7pxGQUEkbZ6kl5ghY4iIbN7"},
    "video02": {"xml": "1JFCJwNHLxmEK20MXFRfnkr1s7c1BsW0o", "h264": "1DkciljwD2YCrgxw8F9IqPF1GVpaf96ce"},
}
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"


def fetch(file_id, dst):
    if dst.exists() and dst.stat().st_size > 0:
        return
    print("downloading", dst.name)
    urllib.request.urlretrieve(URL.format(file_id), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Data/luvizon")
    ap.add_argument("--fps", default="30.15")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    for name, ids in FILES.items():
        tag = name.replace("video", "v")
        fetch(ids["xml"], out / f"{tag}_vehicles.xml")
        raw = out / f"{name}.h264"
        fetch(ids["h264"], raw)
        mp4 = out / f"{name}.mp4"
        if not mp4.exists():
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y", "-r", a.fps, "-i", str(raw), "-c", "copy", str(mp4)], check=True)
        print("ok", mp4)


if __name__ == "__main__":
    main()
