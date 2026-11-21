# BEV-CV_261 — Giám sát giao thông trên bản đồ Bird's-Eye View

Bài tập lớn 3 (tổng hợp) môn Thị giác máy tính, HK 261.
Hệ thống nhận video camera giao thông cố định, phát hiện và theo dấu phương tiện (YOLO26 + ByteTrack/BoT-SORT),
chiếu điểm chạm đất của xe xuống mặt đường bằng homography (BEV, đơn vị mét) rồi tính tốc độ, đếm xe theo vạch ảo,
cảnh báo khoảng cách thời gian (THW), luật làn và bản đồ ùn tắc.

```
video ─► YOLO26 (fine-tune VN) ─► ByteTrack / BoT-SORT ─► tracks_*.jsonl (Data/contracts/schema.json)
                                                                 │
ảnh nền (median) ─► CLAHE ─► HLS + Canny + Hough ─► điểm tụ ─┐   │
camera config (điểm đo / GPS / điểm tụ) ─► homography H ─────┴─► BEV analytics ─► tốc độ, đếm, THW, làn, heatmap
                                                                              └─► đánh giá với GT (Luvizon, AICC22, VNTraffic)
```

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/bevtraffic/` | Package chính: `calibration`, `preprocess`, `lanes`, `tracking`, `analytics`, `evaluation`, `viz` |
| `scripts/` | Chạy từng bước: `track.py`, `analyse_tracks.py`, `eval_luvizon.py`, `eval_aicc22.py`, `eval_mot.py`, `make_*_config.py`, `get_luvizon.py` |
| `configs/cameras/` | Hiệu chuẩn từng camera (homography ảnh → mét, vùng đo, làn, vạch đếm) |
| `A/A_Perception_MOT.ipynb` | Người A: chuẩn hoá nhãn, chia tập theo cụm cảnh, train YOLO26 hai giai đoạn (Colab L4), ByteTrack vs BoT-SORT |
| `B/B_Geometry_BEV_Analytics.ipynb` | Người B: nền tĩnh, Hough, hiệu chuẩn, phân tích BEV và đánh giá trên AICC22 (chạy local) |
| `Pipeline_Evaluation.ipynb` | Tổng hợp mọi kết quả đánh giá |
| `weights/yolo26_vn_finetuned.pt` | Trọng số YOLO26s sau 2 giai đoạn (4 lớp: motorbike, car, bus, truck) |
| `outputs/` | Kết quả: tracks, bảng đánh giá, ảnh, video dashboard |
| `Data/` | AICC22-Custom, VNTraffic (Zenodo 18195750), calibration AI City S01/c002, data contract |
| `legacy/` | Notebook tích hợp cũ (bản đã sinh file zip nộp lần trước), giữ để đối chiếu |
| `baocao.md` | Bản thảo báo cáo |

## Cài đặt

```bash
python -m venv .venv && .venv/Scripts/activate      # Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Chạy được trên CPU (YOLO26s 640 px khoảng 3–4 frame/s); GPU nhanh hơn nhiều.

## Dữ liệu

| Bộ | Dùng để | Cách lấy |
|---|---|---|
| Luvizon et al. (UTFPR) subset01 video01/02 | **Đánh giá tốc độ** với GT vòng từ (km/h từng xe) | `python scripts/get_luvizon.py` (~1.7 GB, Google Drive của tác giả) |
| AICC22-Custom (= AI City S01/c002, 1280×720, 10 fps) | Hiệu chuẩn độc lập (GPS), tốc độ theo khung GT | có sẵn trong `Data/VnTraffic/AICC22-Custom` |
| VNTraffic Hà Nội (510 frame, 30 fps) | MOTA / IDF1 / HOTA | có sẵn trong `Data/VnTraffic/VNTraffic` |
| CCTV Đà Nẵng `nhthau/CCTV-giao-thong-da-nang` | Cảnh thực VN, phân tích không gian | HuggingFace (cần đăng nhập, dataset gated); tracks đã chạy nằm trong `outputs/tracks/` |
| Vietnamese Vehicles (Kaggle `duongtran1909`) | Huấn luyện detector | xem notebook A |

## Chạy lại

```bash
# 1) hiệu chuẩn (đã có sẵn trong configs/, chạy lại nếu muốn)
python scripts/make_aic_config.py          # AICC22: từ calibration.txt GPS của AI City
python scripts/make_danang_config.py       # Đà Nẵng: hai điểm tụ + bề rộng xe Fortuner

# 2) tracking
python scripts/track.py --video Data/luvizon/video01.mp4 --camera luvizon_v01 --fps 30.15 --stride 2
python scripts/track.py --video Data/luvizon/video02.mp4 --camera luvizon_v02 --fps 30.15 --stride 2

# 3) đánh giá
python scripts/eval_luvizon.py             # tốc độ từng xe, fit hệ số trên video01, test trên video02
python scripts/eval_aicc22.py              # AICC22: sai số hình học vs sai số detector
python scripts/eval_mot.py                 # VNTraffic: ByteTrack vs BoT-SORT

# 4) phân tích một camera bất kỳ
python scripts/analyse_tracks.py --tracks outputs/tracks/tracks_danang_cam_01.jsonl \
       --camera configs/cameras/danang_lf_zmcao38w.yaml --background outputs/calib/danang_bg.jpg
```

## Kết quả chính

**Tốc độ – Luvizon video02 (test, GT vòng từ, 280 xe):**

| Cách làm | Xe đo được | MAE (km/h) | RMSE | Trong [−3, +2] km/h |
|---|---|---|---|---|
| Pipeline cũ (hình thang mặc định) | 48 | 106.9 | 116.6 | 0 % |
| H bộ dữ liệu + giả sử làn 3.5 m, YOLO26 fine-tune | 177 | 5.29 | 5.68 | 13.6 % |
| H bộ dữ liệu + hệ số fit trên video01, YOLO26 fine-tune | 177 | **1.19** | 1.92 | **88.7 %** |
| như trên, YOLO26s COCO | 214 | 1.22 | 1.92 | 88.8 % |

**Tách sai số trên AICC22** (tham chiếu: khung GT + hiệu chuẩn GPS AI City): hình học cũ gây sai 10.2 km/h/xe (bias -10.2),
detector + tracker gây 2.9 km/h/xe.

**Tracking – VNTraffic:** BoT-SORT MOTA 0.595, IDF1 0.708, HOTA~ 0.551 (ByteTrack thấp hơn một chút).
Notebook A đọc GT lệch một frame (frame − 1) nên báo MOTA thấp hơn khoảng 0.06; số ở trên tính bằng `scripts/eval_mot.py`.

**Detector – test set ảnh VN (1.922 ảnh, cụm cảnh chưa thấy):** mAP50 0.791, mAP50-95 0.463.

Chi tiết và biểu đồ: `Pipeline_Evaluation.ipynb`, `B/B_Geometry_BEV_Analytics.ipynb`, thư mục `outputs/`.

## Hiệu chuẩn

| Camera | Cách làm | Kiểm tra |
|---|---|---|
| Luvizon subset01 | Homography chỉnh thẳng mặt đường của bộ dữ liệu (`matrix.txt`, làn giữa) + 1 hệ số mét/đơn vị fit trên video01 | Test trên video02 (không dùng khi fit); bề rộng làn suy ra 3.13 m |
| AICC22 (S01/c002) | `calibration.txt` của AI City (GPS → pixel), đổi sang mét east/north | Vạch làn song song trên BEV; tấm bê tông, mũi tên "ONLY" đúng tỷ lệ |
| Đà Nẵng | Điểm tụ dọc đường (vạch tim + bó vỉa) và ngang đường (2 hàng vạch ngang) → tiêu cự, góc camera; chiều cao camera từ bề rộng Fortuner 1.855 m | Gần đúng: f ≈ 5000 px (HFOV ≈ 22°), cao ≈ 4.9 m; chưa có điểm đo kiểm chứng |

Bản trước dùng một hình thang chọn tay và gán kích thước "3 làn × 3.5 m × 35 m" (hoặc 24 m) cho mọi camera.
Trên AICC22, hình thang đó thực tế dài khoảng 66 m, trên Luvizon làm tốc độ sai khoảng 3 lần.

## Hạn chế
- Detector fine-tune trên ảnh Việt Nam bỏ sót nhiều ô tô ở cảnh Đà Nẵng và gộp box ở cảnh Luvizon (xe rất gần camera); đây là nguồn sai số lớn nhất còn lại.
- Hiệu chuẩn Đà Nẵng dựa trên giả định (điểm ảnh vuông, tâm ảnh ở giữa, bề rộng một mẫu xe), nên tốc độ ở camera này chỉ là ước lượng.
- Homography giả định mặt đường phẳng; footpoint là trung điểm cạnh đáy bbox nên lệch với xe quay ngang.
