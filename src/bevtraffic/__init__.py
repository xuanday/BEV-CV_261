"""Traffic monitoring on a bird's-eye-view (BEV) ground plane.

Modules
-------
calibration : image <-> ground-plane homography from measured points, BEV warp
preprocess  : temporal-median background, CLAHE
lanes       : lane-marking extraction (HLS + Canny + HoughLinesP) and vanishing point
tracking    : YOLO + ByteTrack/BoT-SORT -> track records (tracks_stream.jsonl schema)
analytics   : speed, counting, lane rules, time headway, congestion map on the BEV plane
evaluation  : MOT metrics, per-vehicle speed evaluation (Luvizon), GT loaders
viz         : drawing helpers and video rendering
"""

__version__ = "0.2.0"
