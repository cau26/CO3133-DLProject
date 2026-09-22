# Đối chiếu M1 Draft với handbook

Nguồn: Course Project Handbook CO3133, revision 14 September 2026. Cột trạng thái phân biệt bằng chứng trong bộ file đã chuẩn bị và các bước trên hệ thống của nhóm chưa được xác nhận.

## 1. Yêu cầu tối thiểu riêng của M1

Theo mục 7.4, hạn handbook là 23/09/2026, 23:59 GMT+7; M1 chiếm 25% Assignment 1. Mục 7.1 cho phép LMS quy định khác cho từng milestone. Cần xem thông báo M1 trên LMS trước khi chốt cách nộp.

| Yêu cầu | Bằng chứng | Trạng thái |
|---|---|---|
| EDA | outputs/a1_t4/eda; Phần 1 báo cáo | Đã có số liệu và hình |
| Dataset/DataLoader | a1/data.py; batch_info.json | Đã có code và lần chạy; dùng Dataset của torchvision |
| Train-validation loop | a1/train.py; hai history.csv | Đã chạy 10 epoch/mô hình |
| Linear chạy được | linear/best.pt, metrics, notebook | Có bằng chứng và kiểm tra checkpoint |
| MLP chạy được | mlp/best.pt, metrics, notebook | Có bằng chứng và kiểm tra checkpoint |
| CNN | Mục 7.4: optional ở M1 | Chưa làm; không coi là thiếu phần tối thiểu M1 |
| LSTM/GRU và Transformer | Mục 7.4: yêu cầu M2 Final | Kế hoạch Final |

M1 chấp nhận báo cáo một phần và repo đang hoàn thiện (mục 7.7). Slides/video và bộ đầy đủ năm mô hình thuộc yêu cầu hoàn thiện Assignment 1 Final, trừ khi thông báo M1 trên LMS yêu cầu riêng.

## 2. Báo cáo, repo và công bố

| Nội dung theo handbook | Đã chuẩn bị | Việc còn lại |
|---|---|---|
| Ba phần báo cáo (mục 3) | assignment1.md; HTML và PDF | Thành viên rà soát, xác nhận nội dung |
| EDA và batch sau tiền xử lý | 3 hình EDA, thống kê lớp/batch | Kiểm tra hiển thị sau khi đưa lên Pages |
| Phương pháp và thí nghiệm | Kiến trúc, loss, loop, cấu hình, thước đo | Nhóm phải giải thích được code |
| Kết quả và phân tích | Metrics thật, curves, confusion, ví dụ, giới hạn | Xác nhận các nhận xét trong bản nộp |
| Tái lập (mục 4.2) | Source, requirements, config, commands, checkpoints | Đưa toàn bộ file cần thiết lên repo |
| Truy vết code theo commit | Commit lưu đúng source hash; COLAB_RUN.md | Push commit và giữ lịch sử khi merge |
| Trang chung và Assignment 1 | README.md, assignment1.md | Chưa công bố bản mới; kiểm tra link sau push |
| AI ở landing/page/report/log (mục 5) | Đã soạn tóm tắt và bổ sung AI_USAGE.md | Xác nhận đóng góp và rà soát thực tế |
| Link A2/A3 | Giữ các trang hiện có | Cập nhật khi có tài liệu thực tế |
| Nộp M1 (mục 7.1/7.4) | Có bản nháp và gói repo | Cần thông báo/ảnh màn hình yêu cầu LMS |

## 3. Thông tin cần nhóm cung cấp

### A. Yêu cầu nộp M1 trên LMS

Gửi ảnh màn hình hoặc chép nguyên văn thông báo/ô nộp M1: nội dung phải nộp, định dạng file hoặc link, hạn nộp, quy tắc tên file nếu có. Handbook chỉ bắt buộc PDF trên LMS đối với Final; không tự suy ra quy tắc đó cho Draft.

### B. Nhóm và đăng ký

Handbook mục 1.1 yêu cầu nhóm 3-4 người, nhưng thông tin hiện có chỉ gồm Lâm và Cầu. Gửi thông tin thành viên còn lại (họ tên, MSSV, GitHub, vai trò) nếu có; nếu nhóm được phép 2 người, cung cấp xác nhận của giảng viên. Không thêm tên giả hoặc giả định đã được ngoại lệ.

Xác nhận G-M10 đã được điền thống nhất ở GroupRegistration và link landing page đã được điền ở GroupLink. Nếu có ảnh xác nhận, chỉ cần phần của nhóm.

### C. Đóng góp, sử dụng AI và kiểm chứng

Đã xác nhận: Lâm là leader, đã thực hiện lần chạy Colab được cung cấp. Còn cần biết Cầu đã trực tiếp làm phần nào trong M1; Lâm/Cầu đã đọc, sửa và kiểm chứng phần code hoặc báo cáo nào. Phân công dự kiến trong hội thoại không được ghi thành công việc đã hoàn thành.

Nếu nhóm đã dùng thêm công cụ AI, prompt hoặc lần sửa code chưa có trong tài liệu này, gửi nội dung tương ứng. Log Claude có sẵn được giữ nguyên theo repo; nhóm cần kiểm tra lại tên công cụ/model nếu ghi chưa chính xác.

### D. Thay đổi mới ngoài các file đã gửi

Chỉ cần gửi thêm nếu source, outputs hoặc repo đã được nhóm sửa sau các file này: link commit/nhánh hoặc ZIP source mới. Nếu không sửa, bộ source hiện tại đã khớp fingerprint Colab; không cần gửi lại dataset hay chạy training lại chỉ để biên tập báo cáo.

## 4. Trạng thái công bố và nộp

Việc tạo file hoặc commit local không đồng nghĩa đã push, merge, build GitHub Pages hoặc nộp LMS. Các bước này chưa được đánh dấu hoàn thành trong bản kiểm tra hiện tại.
