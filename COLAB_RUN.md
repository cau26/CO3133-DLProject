# Nguồn gốc lần chạy M1 trên Google Colab

## Tài liệu đầu vào

- outputs.zip do người dùng cung cấp; báo cáo chính chỉ dùng outputs/a1_t4.
- Untitled4.ipynb được chép nguyên byte sang notebooks/A1_Colab_Run.ipynb.
- Hai ảnh examples do người dùng gửi khớp nguyên byte với hình trong outputs/a1_t4.
- Nội dung báo cáo được biên tập từ bản nháp người dùng cung cấp và kiểm tra lại bằng kết quả gốc.

## Các mốc khác nhau

- **Chạy training:** người dùng Trần Gia Lâm xác nhận đã thực hiện trên Colab Tesla T4. Metadata trước lượt Linear là 2026-09-22T16:51:30.724936+00:00; trước lượt MLP là 2026-09-22T16:52:45.367547+00:00.
- **Git lúc training:** không có checkout Git trong thư mục giải nén, nên metrics ghi git_commit = null. Trường này được giữ nguyên.
- **Commit lưu mã sau training:** 5d7430a2c5b344ab3bb8af81e7bac7e99f9ab01b. Mười file Python trong commit khớp từng SHA-256 đã ghi trong metrics T4. Commit này do trợ lý tạo để lưu trữ source, không thể hiện thời điểm hay tác giả chạy Colab.
- **Fingerprint source:** 60f66b45930a48b305abd941d400125bcbe313626fcac45a8ce60d9e23e28c5c.
- **Fingerprint split:** 0289c4940126abe338a376ea49eb866788d57cd67fa3ca17fe80139abb8c8fba.

## Bằng chứng kết quả

[comparison.csv](outputs/a1_t4/comparison.csv) tổng hợp hai mô hình. Mỗi thư mục mô hình chứa history.csv, metrics.json, best.pt, validation_predictions.csv và ba hình.

[verification/colab_review.json](verification/colab_review.json) ghi lại các phép kiểm tra đã hoàn thành bởi trợ lý: source hash, split, tính lại metrics từ 6.000 dự đoán, đối chiếu notebook và nạp checkpoint trên CPU. Mỗi checkpoint cho 0 dự đoán khác CSV; sai khác loss nhỏ hơn 1e-8.

Các kiểm tra này không được ghi thành công việc do Cầu hoặc Lâm đã trực tiếp thực hiện. Việc nhóm rà soát nội dung cuối cùng vẫn cần xác nhận.

## Các file được giữ nguyên

Toàn bộ 25 file trong outputs/a1_t4 được chép nguyên byte từ ZIP. Bản draft_report.md bên trong là bản sinh tự động ban đầu; báo cáo biên tập hiện tại là [assignment1.md](assignment1.md).

Notebook gốc được giữ cả output và ô tổng hợp từng in None để bảo toàn lịch sử. Ô sau đã đọc đúng các khóa metrics. Notebook có cảnh báo xung đột numba/numpy; log không ghi cách giải quyết triệt để cảnh báo này.

## Kiểm chứng và tái lập

Lệnh train lại trong README ghi vào thư mục output mới. Nếu chỉ đánh giá checkpoint gốc trên CUDA:

~~~bash
python run_a1.py --stage evaluate --device cuda --out outputs/a1_t4
~~~

Lệnh evaluate hiện ghi lại config và thông tin split. Vì vậy, giữ bản ZIP gốc làm bằng chứng; khi kiểm tra trên thiết bị khác, hãy dùng bản sao thư mục output để không thay metadata thiết bị của bản gốc.

Hai thư mục smoke_t4 và reference_cpu không được dùng cho kết quả chính và không nằm trong gói công bố này.
