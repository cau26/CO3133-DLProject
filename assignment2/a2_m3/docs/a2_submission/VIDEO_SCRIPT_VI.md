# Kịch bản Thuyết trình Video — Assignment 2 (Group G-M10)

**Tiêu đề YouTube bắt buộc:** `CO3133-Semester-261 - Group G-M10 - Assignment 2`  
**Thời lượng mục tiêu:** 10–12 phút (trình chiếu kèm [`output/presentation/G-M10_A2_Slides.pdf`](../../output/presentation/G-M10_A2_Slides.pdf))

---

## Slide 1–2: Giới thiệu & Bài toán (0:00 – 1:30)
- Chào thầy Lê Thành Sách và các bạn, mình là **Nguyễn Hữu Cầu**, đại diện nhóm **G-M10** (cùng bạn **Trần Gia Lâm** - Leader) trình bày đồ án **Assignment 2 — Object Detection trên PASCAL VOC 2007**.
- Mục tiêu bài toán: với mỗi ảnh RGB kích thước bất kỳ, mô hình cần dự đoán đồng thời tọa độ hộp giới hạn $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, nhãn lớp trong 20 danh mục PASCAL VOC, và điểm tự tin.

## Slide 3–4: 4 Thách thức từ EDA & Khóa Split Chống Rò rỉ (1:30 – 3:30)
- Từ EDA trên `5,011` ảnh `trainval` (`15,662` boxes), nhóm phát hiện 4 thách thức cốt lõi:
  1. Có `3,054` vật thể `difficult=1` bị mờ/che khuất (loại khỏi mục tiêu train và `ignore` khi chấm theo chuẩn VOC).
  2. Mất cân bằng lớp **$22.02:1$** giữa `person` (`4,690` boxes) và `bus` (`213` boxes), trong khi `mAP@0.5` tính trung bình đều cho 20 lớp.
  3. **`20.60%` vật thể có kích thước nhỏ (`<2%` diện tích ảnh)** — tập trung ở `bottle`, `pottedplant`, `bird`, `boat`.
  4. Phát hiện cặp ảnh gần trùng `000863`/`008364` (`dHash=3`) nên đã gộp về chung tập `train`, khóa bộ chia **`4,014` Train / `997` Val** (0 rò rỉ), và kiểm tra trùng lặp với **`4,952` ảnh Official Test** ghi nhận **0 trùng ID** và **0 trùng SHA-256**.

## Slide 5–6: Pipeline, 3 Cấu hình Model & Controlled Study (3:30 – 5:30)
- Nhóm tự cài đặt bằng PyTorch/Torchvision thuần túy:
  - **Baseline `ssdlite320`:** Single-stage, MobileNetV3-Large (`4.03M` tham số), co ảnh về $320 \times 320$, nhanh nhưng dễ mất chi tiết vật nhỏ.
  - **Pretrained `Faster R-CNN ResNet-50-FPN` (`COCO_V1`, `41.40M` tham số):** Giữ độ phân giải lớn (`640–1000 px`), tháp đa tầng FPN (`P2–P6`), 2 nhịp (RPN quét vùng nghi ngờ + RoIAlign tinh chỉnh nhãn & khung).
- **Controlled Study (`fasterrcnn_frozen` vs. `fasterrcnn_finetune`):** Giữ cố định split, `seed=42`, trọng số khởi tạo đầu `FastRCNNPredictor(21)`, `batch_size=2`, `SGD lr=0.005` không warmup; chỉ thay đổi duy nhất việc đóng băng thân ResNet-50 (`17.94M` trainable) so với mở khóa toàn bộ (`41.40M` trainable).

## Slide 7–8: Kết quả Validation & Đánh giá 1 lần trên Official Test Set (5:30 – 8:30)
- Trên tập Validation (`997` ảnh, `NVIDIA L4`):
  - `fasterrcnn_frozen` (ep 11) đạt **`83.58%` mAP@0.5**, vượt `ssdlite320` (`50.70%`) tới **`+32.88` điểm** và vượt `fasterrcnn_finetune` (`75.51%`) tới **`+8.07` điểm**, đồng thời tiết kiệm **`64.95%` VRAM** (`1,266 MB` vs `3,612 MB`).
- Trên **Official Test Set (`VOCtest_06-Nov-2007.tar`, 4,952 ảnh, 12,032 vật thể)** khi chấm 1 lần duy nhất:
  - `ssdlite320`: **`52.98%` mAP@0.5** (nhưng Small AP chỉ **`0.79%`**).
  - `fasterrcnn_finetune`: **`76.90%` mAP@0.5** (Small AP `32.89%`).
  - `fasterrcnn_frozen`: **`83.65%` mAP@0.5** (Small AP **`45.32%`**, Med `75.15%`, Large `81.89%`) — khớp gần như tuyệt đối với Validation (`+0.07 pp`).

## Slide 9–10: Phân tích Lỗi, Chi phí Tính toán & Kết luận (8:30 – 11:00)
- Ở ngưỡng `score >= 0.30` trên 12,032 vật thể Test: `fasterrcnn_frozen` bắt trúng **`11,143` vật thể (`92.61%` recall)**, chỉ sót `889`; trong khi `ssdlite320` bỏ sót tới `5,371` vật thể (`2,151` vật nhỏ).
- Mở khóa toàn bộ thân ResNet-50 (`fasterrcnn_finetune`) dưới `lr=0.005` không warmup làm lệch đặc trưng `COCO_V1` trong 8 epoch đầu, tăng lỗi nhầm lớp lên `1,242` ca (so với `938` ca ở bản frozen).
- Kết luận: Đóng băng thân ResNet-50 kết hợp FPN đạt **`83.65%` Official Test mAP@0.5** với chi phí VRAM giảm gần 2/3. Cảm ơn thầy và các bạn đã lắng nghe!
