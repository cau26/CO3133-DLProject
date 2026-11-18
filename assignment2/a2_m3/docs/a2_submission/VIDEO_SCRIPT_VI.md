# CO3133-Semester-261 - Group G-M10 - Assignment 2

**Trần Gia Lâm — 2352670 (Leader) · Nguyễn Hữu Cầu — 2352129 (Member)**

Bài trình bày gồm nghiên cứu phát hiện đa vật thể (Object Detection) trên tập PASCAL VOC 2007 với 3 cấu hình (`ssdlite320`, `fasterrcnn_frozen`, `fasterrcnn_finetune`), thí nghiệm có kiểm soát (Controlled Study) và đánh giá một lần trên Official Test Set (`4,952` ảnh). Trình chiếu song song cùng [`G-M10_A2_Slides.pdf`](../../output/presentation/G-M10_A2_Slides.pdf).

---

## Slide 1. Giới thiệu & Câu hỏi nghiên cứu — Lâm

Dạ, em chào thầy Lê Thành Sách và các bạn. Em là **Trần Gia Lâm**, trưởng nhóm **G-M10**. Cùng thực hiện đồ án với em là bạn **Nguyễn Hữu Cầu**. Đây là bài thuyết trình **Assignment 2** môn **Deep Learning and Its Applications (CO3133)**, học kỳ 261.

Trong Assignment 2, nhóm em chọn hướng **Object Detection trên tập dữ liệu PASCAL VOC 2007** (20 lớp vật thể). Khác với Assignment 1 chỉ phân loại một vật thể nằm giữa ảnh $28\times28$, bài toán này yêu cầu mô hình phải trả lời đồng thời cả **"nó là gì"** và **"nó nằm chính xác ở tọa độ nào"** trong các bức ảnh đời thực có nhiều vật thể chồng lấn và chênh lệch kích thước rất lớn.

Nhóm em đặt ra hai câu hỏi nghiên cứu chính:
1. So sánh giữa một mô hình **single-stage nhẹ cân (`ssdlite320`)** ép ảnh về $320\times320$ với một mô hình **two-stage đa tỷ lệ (`Faster R-CNN ResNet-50-FPN`)** khởi tạo từ trọng số `COCO_V1` thì hiệu năng thay đổi ra sao, đặc biệt trên các vật thể nhỏ?
2. Khi tinh chỉnh `Faster R-CNN` trên tập VOC 2007 với kích thước batch nhỏ (`batch_size = 2`), việc **đóng băng thân ResNet-50 (`fasterrcnn_frozen`)** so với **mở khóa toàn bộ mạng (`fasterrcnn_finetune`)** trong cùng một chế độ huấn luyện sẽ ảnh hưởng thế nào đến độ chính xác, mức độ ổn định và chi phí VRAM?

Trước hết, em mời bạn Cầu trình bày phần bài toán, EDA và quy trình khóa tập dữ liệu chống rò rỉ.

---

## Slide 2. Bài toán & Thước đo đánh giá — Cầu

Em chào thầy và các bạn, em là **Nguyễn Hữu Cầu**. Em xin trình bày phần định nghĩa bài toán, phân tích dữ liệu EDA và đánh giá trên Official Test Set.

Đầu vào của hệ thống là một ảnh RGB kích thước bất kỳ $H \times W$. Đầu ra là danh sách các dự đoán gồm hộp giới hạn $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$, nhãn lớp từ 1 đến 20 (lớp 0 dành cho nền background) và điểm tự tin trong khoảng $[0, 1]$.

Thước đo chính thức nhóm em tự cài đặt theo chuẩn **PASCAL VOC 2007 11-point interpolated mAP@0.5** tại ngưỡng $\text{IoU} \ge 0.50$. Có một quy tắc rất quan trọng của bộ dữ liệu này là cờ `difficult = 1` dành cho những vật thể quá mờ hoặc bị che khuất nặng: khi huấn luyện, nhóm em loại bỏ toàn bộ hộp `difficult = 1` khỏi mục tiêu học; còn khi chấm điểm trên Validation và Test, nếu mô hình dự đoán trúng hộp `difficult = 1` thì trọng tài sẽ bỏ qua (`ignore`) — tức là không cộng điểm nhưng cũng không phạt lỗi. Để đảm bảo bộ chấm điểm chính xác tuyệt đối, nhóm em đã viết sẵn hàm tự kiểm thử toán học `self_test_voc_evaluator()` chạy trước mọi lượt huấn luyện.

---

## Slide 3. Phân tích dữ liệu EDA & 4 Thách thức cốt lõi — Cầu

Từ việc phân tích toàn bộ `5,011` ảnh và `15,662` hộp vật thể trong tập `VOCtrainval_06-Nov-2007.tar` (`12,608` hộp `difficult=0` và `3,054` hộp `difficult=1`), nhóm em rút ra 4 thách thức lớn:

1. **Thứ nhất, lệch phân bố lớp rất mạnh ($22.02 : 1$):** Lớp `person` chiếm tới `4,690` hộp rõ ràng, trong khi các lớp hiếm như `bus` (`213`), `diningtable` (`215`), `sofa` (`248`) hay `sheep` (`257`) có số lượng ít hơn rất nhiều. Nhưng khi tính `mAP@0.5`, tất cả 20 lớp đều có trọng số ngang nhau ($1/20$), nên nếu bỏ rơi các lớp ít mẫu thì điểm tổng sẽ tụt mạnh.
2. **Thứ hai, bài toán vật thể nhỏ (Small-Object Bottleneck):** Khi đo tỷ lệ diện tích hộp so với diện tích toàn ảnh, có tới **`20.60%` (`2,597` hộp) là vật thể nhỏ** chiếm dưới $2\%$ diện tích ảnh — tập trung nhiều ở `bottle`, `pottedplant`, `bird`, `boat`. Nhóm em đặt giả thuyết ngay từ bước EDA rằng mô hình nào co ảnh xuống $320\times320$ sẽ bị mất chi tiết nghiêm trọng ở nhóm vật thể nhỏ này.
3. **Thứ ba, mật độ đa vật thể và che khuất:** Trung bình mỗi ảnh có `2.52` vật thể rõ ràng và tối đa lên tới `37` vật thể trong một khung hình.
4. **Thứ tư, nguy cơ rò rỉ dữ liệu (Data Leakage):** Qua kiểm tra perceptual hash, nhóm em phát hiện một cặp ảnh gần trùng nhau là `000863` và `008364` ($\text{dHash}=3$, $\text{aHash}=2$) đang bị chia vắt ngang giữa Validation và Train.

---

## Slide 4. Khóa Split chống rò rỉ & Kiểm tra Overlap với Official Test — Cầu

Để xử lý triệt để rò rỉ dữ liệu, trước khi chạy bất kỳ mô hình nào ở M2 và M3, nhóm em đã gộp ảnh `000863` về chung với `008364` trong tập Train. Nhờ đó, bộ chia cuối cùng được khóa cố định với **`4,014` ảnh Train** (`10,016` hộp rõ ràng) và **`997` ảnh Validation** (`2,592` hộp rõ ràng + `619` hộp `difficult=1`), đảm bảo **0 cặp trùng lặp xuyên tập** (`split_sha256 = c60294fe...`).

Ngoài ra, ở mốc M3 khi tải về **Official Test Set (`VOCtest_06-Nov-2007.tar`)** gồm **`4,952` ảnh** (`12,032` hộp rõ ràng và `2,944` hộp `difficult=1`), nhóm em chạy thêm bước kiểm tra trùng lặp giữa `5,011` ảnh trainval với `4,952` ảnh test (`test_overlap_check.json`): kết quả ghi nhận **`0` trùng ID** và **`0` trùng SHA-256 tuyệt đối**, đảm bảo đánh giá công bằng trên tập kiểm thử độc lập.

Sau đây, em mời bạn Lâm trình bày định hướng chọn kiến trúc mô hình, pipeline huấn luyện và thiết kế thí nghiệm có kiểm soát.

---

## Slide 5. Định hướng Kiến trúc, Pipeline & Controlled Study — Lâm

Cảm ơn bạn Cầu. Về mặt kiến trúc, nhóm em cài đặt hoàn toàn bằng **PyTorch và Torchvision thuần túy**, không dùng các framework tự động như Ultralytics YOLO để tự kiểm soát toàn bộ vòng lặp nạp dữ liệu, loss và đánh giá.

Ở tầng tiền xử lý, ảnh được đưa về `float32` trong đoạn $[0, 1]$, kiểm tra ràng buộc $x_{\max} > x_{\min}$ và $y_{\max} > y_{\min}$ cho từng hộp. Khi train, nhóm chỉ dùng tăng cường dữ liệu lật ngang ngẫu nhiên với xác suất $0.5$ kèm phản chiếu chính xác tọa độ hộp; khi validation và test thì giữ nguyên ảnh gốc.

Nhóm em định hướng đối chiếu **hai họ kiến trúc đại diện**:
- **Thứ nhất là Baseline `ssdlite320`:** Dùng kiến trúc một giai đoạn (single-stage) kết hợp xương sống nhẹ `MobileNetV3-Large` (`IMAGENET1K_V1`) và đầu phát hiện 21 lớp khởi tạo mới (`4.03 triệu` tham số). Mô hình này ép mọi ảnh về kích thước cố định $320\times320$ và dự đoán một lần duy nhất — ưu điểm là rất nhanh và tốn ít VRAM, nhưng nhược điểm đúng như giả thuyết EDA là dễ mất chi tiết vật nhỏ.
- **Thứ hai là mô hình cải tiến `Faster R-CNN ResNet-50-FPN`:** Khởi tạo từ trọng số `COCO_V1` (`41.40 triệu` tham số) giúp tận dụng biểu diễn giàu thông tin kể cả với các lớp ít mẫu trong VOC. Mô hình giữ độ phân giải lớn linh hoạt (cạnh ngắn `640 px`, cạnh dài tối đa `1000 px`), trích xuất đặc trưng đa tầng qua **tháp FPN (`P2`–`P6`)** và xử lý qua 2 nhịp: nhịp 1 mạng **RPN** quét các vùng nghi ngờ từ nhỏ tới lớn, nhịp 2 **RoIAlign** cắt từng vùng ra soi kỹ lại để chốt nhãn 21 lớp và tinh chỉnh tọa độ hộp.

Từ đó, nhóm thiết kế **Controlled Experiment** giữa hai phiên bản của Faster R-CNN:
- **`fasterrcnn_frozen`:** Đóng băng hoàn toàn thân ResNet-50 (`23.45M` tham số), chỉ cho phép học FPN, RPN và đầu dự đoán (`17.94M` tham số huấn luyện).
- **`fasterrcnn_finetune`:** Mở khóa toàn bộ `41.40M` tham số để fine-tune từ đầu đến cuối.
Hai phiên bản này dùng chung 100% tập dữ liệu, cùng `seed = 42`, cùng trọng số khởi tạo đầu `FastRCNNPredictor` từng bit, cùng `batch_size = 2`, optimizer `SGD (lr = 0.005)`, `MultiStepLR([8, 11])` và không dùng LR warmup. Yếu tố duy nhất thay đổi là đóng băng hay mở khóa `backbone.body`.

---

## Slide 6. Động lực Huấn luyện & Kết quả Pilot — Lâm

Trước khi chạy toàn bộ các epoch chính thức, ở M2 nhóm em đã chạy pilot 200 batch trên GPU `NVIDIA A100` để đo thời gian từng bước, peak VRAM và tín hiệu hội tụ sớm. Ngay sau 200 bước đầu tiên, `fasterrcnn_frozen` đã đạt **`65.02%` mAP**, trong khi `fasterrcnn_finetune` chỉ đạt **`35.81%`** và tốn gấp `2.69 lần` VRAM.

Quan sát đường cong huấn luyện trên phiên chạy M3 đầy đủ bằng GPU `NVIDIA L4` (`22.03 GB` VRAM):
- Với `ssdlite320` (20 epoch), loss giảm đều và mAP tăng vọt khi giảm learning rate 10 lần ở epoch 13, đạt đỉnh Validation **`50.70%`** tại epoch 17.
- Với `fasterrcnn_finetune` (12 epoch), trong 8 epoch đầu ở mức `lr = 0.005` không có warmup, đường mAP dao động mạnh và có lúc tụt xuống quanh `60–61%` do gradient lớn từ đầu dự đoán khởi tạo ngẫu nhiên làm nhiễu đặc trưng `COCO_V1` của thân ResNet-50. Chỉ sau khi giảm LR ở epoch 9, mạng mới ổn định trở lại và đạt **`75.51%`** tại epoch 12.
- Ngược lại, `fasterrcnn_frozen` giữ vững đặc trưng gốc của ResNet-50, tăng mượt mà từ trên `81%` ngay từ epoch 1 lên đỉnh **`83.58%`** tại epoch 11 — tái lập rất sát kết quả A100 ở M2 (`83.63%`).

---

## Slide 7. Kết quả Chính thức: Validation vs. Official Test Set — Lâm & Cầu

**Lâm:** Như vậy trên tập Validation, trong thí nghiệm có kiểm soát, chiến lược đóng băng thân ResNet-50 thắng mở khóa toàn bộ **`+8.07 điểm phần trăm`** (`83.58%` so với `75.51%`), đồng thời giảm **`64.95%` peak VRAM** (`1,266.2 MB` so với `3,612.2 MB`) và rút ngắn **`38.2%` thời gian huấn luyện**. Sau khi chốt các checkpoint tốt nhất (`epoch 17` cho `ssdlite320`, `epoch 11` cho `fasterrcnn_frozen`, `epoch 12` cho `fasterrcnn_finetune`), em mời bạn Cầu báo cáo kết quả chấm một lần duy nhất trên **Official VOC 2007 Test Set**.

**Cầu:** Khi đưa 3 checkpoint đã khóa sang chấm trên **`4,952` ảnh Official Test** (`12,032` vật thể rõ ràng), kết quả tổng quát hóa giữ nguyên rất ổn định:
- **`ssdlite320`:** Đạt **`52.98%` mAP@0.5** (`+2.28 pp` so với Val). Tuy nhiên đúng như giả thuyết EDA ban đầu, khi tách theo kích thước vật thể trên tập Test, AP trên vật thể lớn đạt `64.98%` nhưng AP trên **vật thể nhỏ (`<2%` diện tích) sụp đổ chỉ còn `0.79%`**.
- **`fasterrcnn_finetune`:** Đạt **`76.90%` mAP@0.5** (`+1.40 pp` so với Val), Small AP đạt `32.89%`.
- **`fasterrcnn_frozen`:** Đạt **`83.65%` mAP@0.5** (chênh đúng `+0.07 pp` so với Validation `83.58%`), dẫn đầu tuyệt đối cả 3 nhóm kích thước: Small AP **`45.32%`** (cao gấp 57 lần `ssdlite320`), Medium AP **`75.15%`**, và Large AP **`81.89%`**.

---

## Slide 8. Phân tích Lỗi (Error Taxonomy) trên Official Test Set — Cầu

Để hiểu sâu hơn mô hình sai ở đâu thay vì chỉ nhìn điểm trung bình, nhóm em phân loại toàn bộ các phát hiện có độ tự tin $\ge 0.30$ trên `12,032` vật thể của Official Test Set:
- **`fasterrcnn_frozen`** phát hiện đúng **`11,143` vật thể (tỷ lệ recall `92.61%`)**, chỉ bỏ sót `889` vật thể (trong đó có `367` vật nhỏ).
- **`ssdlite320`** chỉ bắt đúng `6,661` vật thể (`55.36%`) và **bỏ sót tới `5,371` vật thể** (riêng vật nhỏ bị sót tới `2,151` trường hợp).
- **`fasterrcnn_finetune`** bắt đúng `10,108` vật thể (`84.01%`), bỏ sót nhiều hơn gấp `2.16` lần so với bản đóng băng (`1,924` vs `889`) và đặc biệt là **lỗi nhầm lớp (`wrong_class`) tăng vọt `+32.4%`** (`1,242` ca so với `938` ca của bản frozen) — cho thấy việc làm nhiễu đặc trưng backbone lúc đầu làm suy giảm khả năng phân biệt ngữ nghĩa.

Về các cặp nhầm lớp phổ biến nhất trên tập Test, các mô hình hay nhầm giữa những đối tượng có hình dáng và bối cảnh gần nhau như nội thất trong nhà (`sofa` $\to$ `chair`: `124` ca ở `fasterrcnn_frozen`) và động vật bốn chân (`cat` $\leftrightarrow$ `dog`: `51` và `40` ca).

---

## Slide 9. Ví dụ Trực quan & Đánh đổi Chi phí Tính toán — Lâm

Nhìn vào các ảnh minh họa trên Official Test Set (khung xanh là Ground Truth, khung đỏ là dự đoán $\ge 0.35$): ở những cảnh đơn giản có vật thể lớn như người, chó hay tàu hỏa, cả `ssdlite320` và `fasterrcnn_frozen` đều nhận diện tốt. Nhưng khi bước vào cảnh đông đúc ngoài phố hoặc phòng ăn nhiều ghế và chai lọ nhỏ, `ssdlite320` bỏ sót gần hết các vật thể nhỏ ở xa, trong khi `fasterrcnn_frozen` khoanh vùng chính xác phần lớn vật thể với độ tự tin cao.

Về bài toán đánh đổi (Trade-off):
- `ssdlite320` chỉ có `4.03M` tham số, mất `462.8 MB` VRAM và đạt tốc độ suy luận **`16.22 ms/ảnh` (~`61.7 FPS`)**, phù hợp nếu triển khai trên thiết bị biên với điều kiện vật thể phải to và gần camera.
- `fasterrcnn_frozen` nặng hơn (`41.40M` tổng tham số, tốc độ `31.34 ms/ảnh` ~`31.9 FPS` trên L4, `1,266.2 MB` VRAM) nhưng đổi lại mang đến mức tăng **`+30.67 điểm phần trăm` Test mAP@0.5**, và vượt trội hoàn toàn so với fine-tune toàn bộ cả về chất lượng lẫn bộ nhớ.

---

## Slide 10. Hạn chế, Kết luận & Tính tái tạo — Lâm

Cuối cùng, nhóm em xin nêu thẳng các hạn chế của nghiên cứu hiện tại:
1. Thí nghiệm hiện mới chạy trên một random seed (`seed = 42`), chưa đo phương sai qua nhiều seed.
2. Ở nhánh `fasterrcnn_finetune`, nhóm cố ý giữ nguyên tốc độ học đồng nhất `lr = 0.005` không warmup để làm controlled experiment công bằng; thực tế nếu muốn fine-tune toàn bộ ổn định hơn với `batch_size = 2`, cần bổ sung linear LR warmup hoặc tốc độ học phân tầng nhỏ hơn cho backbone.
3. `ssdlite320` bị giới hạn bởi độ phân giải $320\times320$ và chưa áp dụng tăng cường cắt phóng đa tỷ lệ (multi-scale crop).

**Kết luận:** Bằng việc kết hợp phân tích EDA phát hiện nút thắt vật thể nhỏ, khóa bộ chia chống rò rỉ (`000863`/`008364`), dùng tháp đặc trưng đa tầng `FPN` và đóng băng thân `ResNet-50` tiền huấn luyện trên `COCO_V1`, nhóm G-M10 đạt **`83.65%` mAP@0.5 trên Official VOC 2007 Test Set**, đồng thời tiết kiệm gần hai phần ba bộ nhớ GPU khi huấn luyện. Toàn bộ mã nguồn, danh sách split, notebook `A2_M3_Colab_Run.ipynb` và các checkpoint `best.pt` trên Google Drive đã được lưu trữ đầy đủ để kiểm chứng và tái tạo kết quả.

Em xin thay mặt nhóm G-M10 cảm ơn thầy Lê Thành Sách và các bạn đã lắng nghe!
