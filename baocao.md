# BÁO CÁO BÀI TẬP LỚN 3 — THỊ GIÁC MÁY TÍNH
## Đề tài: Giám sát giao thông từ camera cố định trên bản đồ Bird's-Eye View (YOLO26 + MOT + Homography)

Mã nguồn: https://github.com/xuanday/BEV-CV_261 (package `src/bevtraffic`, script trong `scripts/`, notebook A, B và `Pipeline_Evaluation.ipynb`).

---

## Đối chiếu yêu cầu đề bài

| Yêu cầu | Nội dung trong bài | Mục |
|---|---|---|
| 3.1 Thu thập & tiền xử lý | 5 nguồn video (Luvizon – Brazil, AICC22 – Mỹ, VNTraffic – Hà Nội, CCTV Đà Nẵng, ảnh Kaggle VN). Ảnh nền bằng trung vị theo thời gian, CLAHE trên kênh L (HLS), lọc nhãn < 8 px, chia tập theo cụm cảnh | 1, 3.1 |
| 3.2 Hình học cảnh | HLS mask → Canny → HoughLinesP; điểm tụ; homography ảnh → mặt đường (đơn vị mét) bằng 3 cách: homography của bộ dữ liệu + hệ số tỉ lệ, hiệu chuẩn GPS, hai điểm tụ + kích thước đã biết | 3 |
| 3.3 Nội dung / đối tượng | YOLO26s fine-tune 2 giai đoạn (4 lớp), ByteTrack và BoT-SORT, điểm chạm đất | 4 |
| 3.4 Trực quan hoá | Ảnh hiệu chuẩn có lưới mét, video dashboard camera + BEV (box, ID, km/h, vết quỹ đạo, bản đồ ùn tắc), biểu đồ tốc độ | 3, 5, 6 |
| Mở rộng | So sánh: hình học cũ / mới, model COCO / fine-tune, ByteTrack / BoT-SORT, 4 cách ước lượng tốc độ. Đánh giá định lượng với GT tốc độ thật (vòng từ) và GT tracking | 6 |

## Phân công

| Thành viên | Phần việc | File chính |
|---|---|---|
| [Họ tên – MSSV] (A) | Dữ liệu ảnh, chuẩn hoá nhãn, chia tập theo cụm cảnh, huấn luyện YOLO26 hai giai đoạn, ByteTrack/BoT-SORT, đánh giá MOT | `A/A_Perception_MOT.ipynb`, `src/bevtraffic/tracking.py` |
| [Họ tên – MSSV] (B) | Ảnh nền, Hough, homography và BEV, 5 mô-đun phân tích không gian, dashboard | `B/B_Geometry_BEV_Analytics.ipynb`, `src/bevtraffic/analytics.py`, `viz.py` |
| [Họ tên – MSSV] (C) | Tích hợp, hiệu chuẩn từng camera, đánh giá tốc độ với GT (Luvizon, AICC22), sửa lỗi đánh giá, package và script | `src/bevtraffic/calibration.py`, `evaluation.py`, `scripts/`, `Pipeline_Evaluation.ipynb` |
| [Họ tên – MSSV] (D) | (nếu có) | |

---

## 1. Bài toán và dữ liệu

Camera giám sát chỉ cho toạ độ pixel. Do phối cảnh, 1 m trên mặt đường ở gần camera chiếm hàng chục pixel, ở xa chỉ vài pixel,
nên tốc độ hay khoảng cách tính bằng pixel không có ý nghĩa vật lý. Hệ thống phát hiện, theo dấu xe rồi chiếu điểm chạm đất xuống
mặt phẳng đường bằng homography để đo bằng mét và giây.

| Bộ dữ liệu | Quy mô | Ground truth | Vai trò |
|---|---|---|---|
| Vietnamese Vehicles (Kaggle `duongtran1909`) | 14.897 ảnh sau làm sạch (ngày/đêm), 4 lớp | bbox | Huấn luyện detector |
| Luvizon et al. (UTFPR), subset01 video01–02 | 1920×1080, 30.15 fps; 6.918 + 14.837 frame; 110 + 280 xe có tốc độ | Tốc độ từng xe đo bằng vòng từ | **Đánh giá tốc độ** |
| AICC22-Custom (Zenodo 18195750) = AI City S01/c002 | 1280×720, 10 fps, 200 frame | bbox + ID; hiệu chuẩn GPS của AI City | Hiệu chuẩn độc lập, tách sai số |
| VNTraffic (Zenodo 18195750), Quang Trung – Lê Trọng Tấn | 1920×1080, 30 fps, 510 frame, 85 ID | bbox + ID | MOTA / IDF1 / HOTA |
| CCTV Đà Nẵng (`nhthau/CCTV-giao-thong-da-nang`) | 3 đoạn C001–C003 × 2 phút, 1920×1080, 30 fps | không có | Cảnh thực Việt Nam |

Ghi chú: các bộ Roboflow trong notebook A không được tải khi chạy (thiếu API key), nên detector chỉ học trên dữ liệu Kaggle.

## 2. Kiến trúc

```
video → YOLO26 → ByteTrack/BoT-SORT → tracks_*.jsonl (hợp đồng dữ liệu, 10 trường)
ảnh nền → CLAHE → Hough → điểm tụ ┐
camera config (điểm đo / GPS / điểm tụ) → H (ảnh → mét) ┴→ chiếu điểm chạm đất → tốc độ, đếm, THW, luật làn, ùn tắc → đánh giá
```

Hai phân hệ trao đổi qua `Data/contracts/schema.json` (mỗi dòng: camera, frame, thời điểm, fps, track_id, lớp, độ tin cậy, bbox,
điểm chạm đất). Mọi thuật toán nằm trong package `src/bevtraffic`; notebook và script chỉ gọi lại.

## 3. Tiền xử lý và hình học cảnh

### 3.1 Ảnh nền và tăng tương phản
Lấy N frame cách đều (60–100), tính trung vị theo thời gian từng điểm ảnh. Xe chạy qua chiếm điểm ảnh ít hơn một nửa số mẫu nên bị loại;
xe đỗ cả đoạn video vẫn nằm lại trong nền. CLAHE (clip 2.5, lưới 8×8) làm trên kênh L của HLS để giữ màu vạch vàng/trắng.

### 3.2 Vạch kẻ đường và điểm tụ
Mặt nạ trắng (L ≥ 175, S ≤ 110; mặt đường bê tông sáng cần L ≥ 215) hợp với mặt nạ vàng (H 12–38), Canny (50, 150), cắt theo vùng đường,
`HoughLinesP` (ngưỡng 25, dài ≥ 25–30 px). Các đoạn chạy dọc đường cho điểm tụ theo bình phương tối thiểu có trọng số độ dài.
Trên BEV, một homography đúng làm các vạch dọc đường song song với trục đường; độ lệch góc của các đoạn Hough trên BEV được dùng làm phép kiểm tra.

### 3.3 Homography ảnh → mặt đường
Điểm ảnh (u, v) trên mặt đường và toạ độ (X, Y) tính bằng mét liên hệ qua ma trận H 3×3 (8 bậc tự do):
X = (h11 u + h12 v + h13)/(h31 u + h32 v + h33), Y = (h21 u + h22 v + h23)/(h31 u + h32 v + h33).
Tỉ lệ ảnh BEV là 10–20 px/m. Mỗi camera hiệu chuẩn theo cách phù hợp với thông tin sẵn có:

| Camera | Cách hiệu chuẩn | Kiểm tra |
|---|---|---|
| Luvizon | Homography chỉnh thẳng mặt đường đi kèm bộ dữ liệu (`matrix.txt`, làn giữa), chỉ còn thiếu một hệ số mét/đơn vị. Hệ số lấy theo trung vị tỉ số GT/dự đoán trên **video01**, cố định rồi kiểm tra trên **video02** | Bề rộng làn suy ra 3.13 m (hợp lý với đường đô thị); sai số trên video02 ở mục 6.1 |
| AICC22 | `calibration.txt` của AI City (GPS → pixel, sai số chiếu lại 11.4 px ở 1080p): đảo ma trận, đổi (vĩ độ, kinh độ) sang mét east/north, xoay cho đường vào nằm dọc trục y | Vạch làn song song trên BEV (lệch −1.2° ± 6.8°), tấm bê tông và mũi tên "ONLY" đúng tỉ lệ |
| Đà Nẵng | Hai điểm tụ: dọc đường (vạch tim + bó vỉa) và ngang đường (hai hàng vạch ngang). Giả sử điểm ảnh vuông và tâm ảnh ở giữa: (v1 − c)·(v2 − c) + f² = 0 cho f ≈ 5037 px (HFOV ≈ 21.6°). Từ f, v1, v2 dựng ma trận xoay; chiều cao camera ≈ 4.9 m suy ra từ bề rộng Toyota Fortuner đỗ bên đường (1.855 m) | Gần đúng; khi f lệch ±25 %, quãng đường dọc thay đổi khoảng ±20 % |

**So với bản trước.** Bản trước dùng một hình thang chọn tay và gán cứng kích thước (3 làn × 3.5 m × 35 m, hoặc × 3.66 m × 24 m với AICC22).
Theo hiệu chuẩn GPS, hình thang AICC22 thực ra dài khoảng 66–67 m chứ không phải 24 m, nên tốc độ dọc đường bị đo thấp khoảng 2.7 lần.
Ngoài ra, notebook B cũ sinh 8 "điểm hiệu chuẩn" bằng cách chiếu ngược từ chính hình thang đó rồi cộng nhiễu Gauss 0.55 px;
RMSE 0.872 px báo cáo trước đây chỉ phản ánh lượng nhiễu tự thêm (σ = 0 cho RMSE = 0). Bước `cv2.undistort` với K, D tự đặt cũng đã bỏ
vì không có hiệu chuẩn camera thật.

Độ phân giải của phép đo giảm nhanh theo khoảng cách: với AICC22, 1 px theo chiều dọc ảnh ứng với 0.03 m ở gần (v = 700) nhưng 0.55 m ở xa (v = 160).
Vùng đo được giới hạn để bỏ phần gần đường chân trời.

## 4. Phát hiện và theo dấu phương tiện

### 4.1 Dữ liệu và chia tập
Nhãn được ánh xạ về 4 lớp (motorbike, car, bus, truck), bỏ box < 8 px, lọc ảnh gần trùng bằng SSIM (≥ 0.97). Ảnh được gom theo cụm cảnh
(tiền tố tên file, ví dụ `cam_18`), cả cụm đi vào cùng một tập, nên các frame liên tiếp của một camera không lọt sang tập khác.
Kết quả: train 11.946, val 1.029, test 1.922 ảnh. Khi chia, miền ban đêm có quá ít cụm nên phép làm tròn cho 0 cụm val
(tập "nighttime_val" rỗng, chưa đánh giá được ban đêm). Cần đảm bảo mỗi miền có ít nhất một cụm val rồi train lại (cần GPU).

### 4.2 YOLO26s hai giai đoạn (Colab, NVIDIA L4)
Giai đoạn 1: 50 epoch từ `yolo26s.pt`, AdamW lr0 = 0.005, cosine, batch 40, mosaic 0.8 (tắt 10 epoch cuối), giữ đầu NMS (`nms=True`, `max_det=500`).
Giai đoạn 2: 25 epoch từ best giai đoạn 1, lr0 = 0.001, trên 6.823 ảnh ban đêm và ảnh có bus/truck.

| Tập đánh giá | Ảnh | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Val stage 1 — best giai đoạn 1 | 1.029 | 0.831 | 0.755 | 0.846 | 0.608 |
| Val stage 1 — sau giai đoạn 2 | 1.029 | 0.753 | 0.802 | 0.828 | 0.604 |
| Val stage 2 (ảnh có bus/truck) | 54 | 0.849 | 0.858 | 0.904 | 0.696 |
| **Test (cụm cảnh chưa thấy)** | 1.922 | 0.781 | 0.721 | **0.791** | **0.463** |

Trên cùng tập val, giai đoạn 2 đổi precision lấy recall và không cải thiện mAP. Tập val cũng dùng để chọn `best.pt`, nên con số đáng tin nhất là tập test.

### 4.3 Theo dấu và điểm chạm đất
ByteTrack (ngưỡng cao 0.45, thấp 0.10, buffer 30) và BoT-SORT (thêm bù chuyển động camera sparseOptFlow, buffer 45).
Điểm đại diện của xe là trung điểm cạnh đáy bbox (u = (x1 + x2)/2, v = y2). Điểm này nằm gần mặt đường, còn tâm box thì ở độ cao h/2 nên chiếu sai về phía xa.
Box chạm mép ảnh (cạnh đáy bị cắt) và box chiếm > 40 % khung hình (box gộp nhiều xe) bị loại khỏi phép đo.

## 5. Phân tích không gian trên BEV

1. **Tốc độ.** Hồi quy tuyến tính vị trí theo thời gian trên cửa sổ 0.8 s (dùng thời điểm thật của từng frame, chịu được frame bị mất), sau đó làm mượt EMA α = 0.3.
   Bước nhảy > 250 km/h (thường do đổi ID) bị bỏ. Trên AICC22, độ lệch chuẩn trong một track: 1 bước 8.2 km/h → hồi quy 5.5 → hồi quy + EMA 4.9 → Kalman 6.1.
   Với đánh giá từng xe, tốc độ của track là độ dốc hồi quy trên toàn đoạn đo.
2. **Đếm xe.** Đoạn quỹ đạo giữa hai điểm liên tiếp cắt vạch ảo (kiểm tra bằng tích có hướng) và đi đúng chiều quy định; mỗi xe đếm một lần cho mỗi vạch.
3. **Luật làn.** Điểm trong đa giác làn (thu nhỏ 0.3 m), giữ liên tục ≥ 0.8 s mới tính vi phạm (lớp xe không được phép, hoặc đi ngược chiều làn). Lớp của track lấy theo đa số phiếu có trọng số độ tin cậy để tránh đổi lớp giữa chừng.
4. **Khoảng cách thời gian THW.** Trong cùng làn, xếp xe theo trục làn; cặp xe sau/trước lệch ngang < 1.5 m; khoảng trống = khoảng cách tâm − nửa tổng chiều dài hai xe; THW = khoảng trống / tốc độ xe sau, cảnh báo khi < 1.8 s và xe sau > 20 km/h. Mỗi cặp xe ghi một sự kiện (THW nhỏ nhất, số frame).
5. **Bản đồ ùn tắc.** Cộng thời gian dừng (s) của xe chậm hơn 10 km/h vào lưới BEV, làm mờ Gauss σ = 1 m.

## 6. Kết quả

### 6.1 Tốc độ trên Luvizon (GT vòng từ)
Ghép xe GT với track có box chứa tâm biển số tại frame `iframe`, đo tốc độ trên các điểm chạm đất hợp lệ trong 2 s sau đó.

| Video | Model | Hình học | Xe GT | Đo được | MAE (km/h) | RMSE | Bias | MAPE | Trong [−3, +2] |
|---|---|---|---|---|---|---|---|---|---|
| v01 (fit) | YOLO26 fine-tune | hình thang mặc định (bản cũ) | 110 | 17 (15 %) | 98.53 | 104.67 | +98.53 | 290.6 % | 0.0 % |
| v01 (fit) | YOLO26 fine-tune | H bộ dữ liệu + giả sử làn 3.5 m | 110 | 65 (59 %) | 5.04 | 5.56 | +4.81 | 10.9 % | 20.0 % |
| v01 (fit) | YOLO26 fine-tune | H bộ dữ liệu + hệ số fit trên video01 | 110 | 65 (59 %) | 1.01 | 1.44 | -0.26 | 3.3 % | 92.3 % |
| v02 (test) | YOLO26 fine-tune | hình thang mặc định (bản cũ) | 280 | 48 (17 %) | 106.89 | 116.56 | +106.89 | 279.0 % | 0.0 % |
| v02 (test) | YOLO26 fine-tune | H bộ dữ liệu + giả sử làn 3.5 m | 280 | 177 (63 %) | 5.29 | 5.68 | +4.71 | 11.7 % | 13.6 % |
| v02 (test) | YOLO26 fine-tune | H bộ dữ liệu + hệ số fit trên video01 | 280 | 177 (63 %) | 1.19 | 1.92 | -0.45 | 4.1 % | 88.7 % |
| v01 (fit) | YOLO26s COCO | hình thang mặc định (bản cũ) | 110 | 26 (24 %) | 110.51 | 116.74 | +110.51 | 321.4 % | 0.0 % |
| v01 (fit) | YOLO26s COCO | H bộ dữ liệu + giả sử làn 3.5 m | 110 | 94 (85 %) | 5.03 | 5.55 | +4.84 | 10.8 % | 18.1 % |
| v01 (fit) | YOLO26s COCO | H bộ dữ liệu + hệ số fit trên video01 | 110 | 94 (85 %) | 1.12 | 1.49 | -0.28 | 3.4 % | 89.4 % |
| v02 (test) | YOLO26s COCO | hình thang mặc định (bản cũ) | 280 | 47 (17 %) | 100.64 | 111.15 | +100.64 | 289.1 % | 0.0 % |
| v02 (test) | YOLO26s COCO | H bộ dữ liệu + giả sử làn 3.5 m | 280 | 214 (76 %) | 5.64 | 6.11 | +5.21 | 12.1 % | 12.2 % |
| v02 (test) | YOLO26s COCO | H bộ dữ liệu + hệ số fit trên video01 | 280 | 214 (76 %) | 1.22 | 1.92 | -0.10 | 3.8 % | 88.8 % |

Hình học cũ (hình thang mặc định) cho tốc độ gấp khoảng 3 lần thực tế. Nếu chỉ giả sử làn rộng 3.5 m thì vẫn lệch có hệ thống khoảng +5 km/h.
Với hệ số lấy từ video01, sai số trên video02 (không dùng khi hiệu chuẩn) là MAE 1.19 km/h, RMSE 1.92 km/h, 88.7 % số xe trong khoảng [−3, +2] km/h với model fine-tune (177/280 xe đo được); model COCO đo được 214/280 xe với MAE 1.22 km/h (88.8 % trong [−3, +2]).
Tỉ lệ xe đo được bị giới hạn bởi detector: ở cảnh này xe rất gần camera, YOLO26 fine-tune hay gộp box hoặc gán xe con thành bus/truck.

### 6.2 AICC22: tách sai số hình học và sai số detector

| Trường hợp | Xe | MAE (km/h) | RMSE | Bias | MAPE |
|---|---|---|---|---|---|
| Khung GT + hình thang cũ (chỉ sai hình học) | 13/13 | 10.15 | 14.83 | −10.15 | 32.3 % |
| YOLO26 fine-tune + ByteTrack, hiệu chuẩn AI City (chỉ sai detector) | 11/13 | 2.87 | 4.80 | +2.13 | 17.6 % |
| YOLO26 fine-tune + ByteTrack, hình thang cũ (pipeline cũ) | 11/13 | 11.14 | 16.04 | −10.57 | 32.3 % |
| YOLO26s COCO + ByteTrack, hiệu chuẩn AI City | 13/13 | 2.21 | 5.73 | +1.27 | 18.7 % |

Tốc độ tham chiếu là khung GT chiếu bằng hiệu chuẩn GPS (trung vị 32.8 km/h). Bản cũ báo "MAE 0.15 km/h" vì so dự đoán với chính khung GT
chiếu qua **cùng** homography sai, nên lỗi hình học triệt tiêu. Offset frame của GT AICC22 là −1 (ước lượng bằng IoU).

### 6.3 Tracking trên VNTraffic

| Tracker | Đọc GT | MOTA | IDF1 | HOTA~ | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|
| ByteTrack | offset 0 (ước lượng) | 0.577 | 0.697 | 0.530 | 23 | 1723 | 5040 |
| ByteTrack | frame − 1 (cách cũ) | 0.512 | 0.664 | 0.511 | 24 | 2239 | 5556 |
| BoT-SORT | offset 0 (ước lượng) | 0.595 | 0.708 | 0.551 | 21 | 1634 | 4844 |
| BoT-SORT | frame − 1 (cách cũ) | 0.529 | 0.673 | 0.528 | 21 | 2164 | 5374 |

GT VNTraffic đánh số frame từ 0 (IoU khớp nhất ở offset 0). Notebook A đọc GT với frame − 1, lệch một frame nên MOTA thấp hơn khoảng 0.06;
hàng "offset 0" tính bằng `scripts/eval_mot.py` trên cùng file tracks.
Cả hai tracker chưa đạt mục tiêu IDF1 ≥ 0.75 và HOTA ≥ 0.60 nhóm tự đặt; FN chiếm phần lớn lỗi, tức detector bỏ sót xe máy trong cụm đông.

### 6.4 CCTV Đà Nẵng

Chạy lại tracking trên video gốc (trọn 2 phút mỗi đoạn, lấy 1 frame / 2, ảnh vào 960 px, BoT-SORT) với hai detector,
cùng hiệu chuẩn và cùng vạch đếm. Hàng "Colab" là tracks của bản nộp trước (fine-tune 640 px, ByteTrack, 30 s đầu).

| Detector | Đoạn | Thời lượng | Track | Đi > 5 m | Đếm qua vạch | Xe/phút | Theo lớp | Tốc độ TB (km/h) | Cặp THW < 1.8 s | Vi phạm làn |
|---|---|---|---|---|---|---|---|---|---|---|
| Colab (fine-tune 640) | C001 | 30 s | 104 | 6 | 4 | 8.0 | 4 xe máy | 10.1 | 6 | 2 |
| Colab (fine-tune 640) | C002 | 30 s | 52 | 1 | 1 | 2.0 | 1 bus | 9.0 | 0 | 0 |
| Colab (fine-tune 640) | C003 | 30 s | 144 | 8 | 9 | 18.0 | 5 truck, 2 bus, 1 car, 1 xe máy | 13.3 | 12 | 1 |
| fine-tune 960 | C001 | 2 phút | 57 | 10 | 8 | 4.0 | 6 xe máy, 1 bus, 1 truck | 10.5 | 5 | 2 |
| fine-tune 960 | C002 | 2 phút | 48 | 5 | 4 | 2.0 | 1 xe máy, 1 bus, 1 car, 1 truck | 10.2 | 1 | 0 |
| fine-tune 960 | C003 | 2 phút | 74 | 11 | 5 | 2.5 | 2 truck, 2 xe máy, 1 car | 13.6 | 2 | 0 |
| **YOLO26s COCO 960** | C001 | 2 phút | 210 | 62 | 21 | 10.5 | 16 xe máy, 4 car, 1 truck | 10.1 | 17 | 2 |
| **YOLO26s COCO 960** | C002 | 2 phút | 199 | 59 | 21 | 10.5 | 17 xe máy, 4 car | 10.7 | 20 | 3 |
| **YOLO26s COCO 960** | C003 | 2 phút | 282 | 65 | 32 | 16.0 | 22 xe máy, 10 car | 10.4 | 24 | 5 |

Model fine-tune gán phần lớn ô tô ở camera này thành "truck"/"bus" và gần như không theo được xe ở xa (0–1 track đi qua đoạn 10–15 m),
tăng ảnh vào lên 960 px không cải thiện. Model COCO cho số track đi > 5 m gấp 6–12 lần và nhận đúng ô tô.
Dời vạch đếm từ 10 m đến 34 m dọc đường, số track cắt qua gần như không đổi (17–22 mỗi đoạn với COCO), nên vị trí vạch không phải nguyên nhân.
Tốc độ trung bình khoảng 10 km/h, khớp với biển giới hạn 5 km/h và trạm thu phí ở cuối đường. Không có GT nên chưa biết tỉ lệ bỏ sót còn lại.
Video minh hoạ: `outputs/videos/danang_c003_dashboard.mp4` (60 s đầu của C003, COCO 960).

### 6.5 So với bản nộp trước

| Hạng mục | Bản trước | Bản này |
|---|---|---|
| Hiệu chuẩn | Hình thang đoán tay, kích thước gán cứng | Homography từ dữ liệu đo (bộ dữ liệu, GPS, điểm tụ) |
| RMSE hiệu chuẩn | 0.872 px trên điểm tự sinh | Kiểm tra trên dữ liệu độc lập (video02, GPS) |
| GT tốc độ Luvizon | Trung bình cả video (3.8–8.3 phút) so với 350 frame đầu | Ghép từng xe theo frame |
| Sai số tốc độ Luvizon | 69–72 km/h (video01) | xem 6.1 |
| "Human GT" AICC22/VNTraffic | GT chiếu qua cùng H, MAE ≈ 0.1 km/h | Tham chiếu độc lập (GPS) |
| MOT VNTraffic | GT lệch 1 frame | Đánh giá lại bằng `scripts/eval_mot.py`, offset ước lượng từ dữ liệu |
| THW | Đếm theo (frame × cặp) | Một sự kiện cho mỗi cặp xe |
| Đếm xe | Không xét hướng | Theo hướng, mỗi xe một lần |
| Video | 450 / 900 frame | AICC22 đủ 200 frame; Luvizon 30 s, Đà Nẵng 60 s minh hoạ; tracks đầy đủ trong outputs/tracks |

## 7. Ca thất bại, hạn chế, hướng phát triển

1. **Detector lệch miền.** Model học trên ảnh Việt Nam (nhiều xe máy) gộp box và gán sai lớp ở cảnh Luvizon (xe con rất gần camera) và bỏ sót ô tô ở Đà Nẵng.
   Model COCO cho kết quả tương đương hoặc tốt hơn trên ô tô (6.1, 6.2), và ở Đà Nẵng theo được nhiều xe hơn 6–12 lần (6.4). Hướng sửa: thêm ảnh CCTV góc cao vào tập train, hoặc kết hợp hai model.
2. **Hiệu chuẩn gần đúng.** Đà Nẵng dựa trên giả định điểm ảnh vuông, tâm ảnh ở giữa và kích thước một mẫu xe; điểm tụ ngang đường nằm rất xa nên f nhạy với sai số đo.
   Cần đo vài khoảng cách thực tế hoặc dùng ảnh vệ tinh để có điểm kiểm tra.
3. **Mặt đường phẳng và điểm chạm đất.** Homography giả định đường phẳng. Với xe quay ngang hoặc xe dài, trung điểm cạnh đáy bbox không trùng tâm vệt bánh xe;
   có thể dùng YOLO-OBB hoặc segmentation để lấy đường tiếp xúc chính xác hơn.
4. **Dữ liệu đánh giá ngắn.** AICC22 chỉ có 200 frame (13 xe chạy), VNTraffic 510 frame; kết quả mang tính minh hoạ.
5. **Chưa đánh giá detector ban đêm** (tập val đêm rỗng do làm tròn số cụm khi chia tập; cần sửa cách chia và train lại).
