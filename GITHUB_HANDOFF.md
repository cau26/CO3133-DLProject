# Bộ file chuẩn bị cho GitHub

Bản đã biên tập là assignment1.md. Mở G-M10_A1_Draft.html hoặc G-M10_A1_Draft.pdf để đọc; DRAFT1_CHECKLIST.md liệt kê những xác nhận còn cần từ nhóm.

## Các file chính

| File/thư mục | Mục đích |
|---|---|
| assignment1.md | Nội dung trang Assignment 1 và nguồn của báo cáo |
| G-M10_A1_Draft.html / G-M10_A1_Draft.pdf | Hai định dạng để đọc bản đã biên tập |
| outputs/a1_t4 | Kết quả gốc của nhóm, được giữ nguyên |
| notebooks/A1_Colab_Run.ipynb | Notebook gốc, giữ cả lịch sử chạy |
| a1, configs, run_a1.py, requirements-a1.txt | Source và cấu hình để chạy lại |
| README.md / AI_USAGE.md | Trang chung, hướng dẫn tái lập và AI disclosure |
| COLAB_RUN.md / verification | Nguồn gốc và kiểm chứng |
| DRAFT1_CHECKLIST.md | Đối chiếu handbook và các mục chờ nhóm xác nhận |

Gói không chứa dữ liệu Fashion-MNIST tải về, kết quả smoke hoặc kết quả CPU tham khảo.

## Giữ lịch sử source khi đưa lên GitHub

Gói ZIP có kèm handoff/CO3133_M1_Review.bundle, chứa lịch sử Git của nhánh local a1-m1-review. Nhánh dựa trên main ở commit 44cb576; commit đầu lưu source khớp lượt Colab, commit sau thêm báo cáo và kết quả. Tác giả các commit chuẩn bị là Codex, không được diễn giải là commit do sinh viên tự viết.

Lịch sử này giúp giữ đúng liên kết đến commit lưu source trong COLAB_RUN.md. Chỉ upload các file qua giao diện web sẽ không đưa commit đó lên GitHub.

Trong một bản clone repo sạch, có thể nhập nhánh từ file bundle bằng cách thay đường dẫn sau bằng đường dẫn thật:

~~~bash
git status
git fetch "DUONG_DAN/CO3133_M1_Review.bundle" a1-m1-review:a1-m1-review
git switch a1-m1-review
git fetch origin
git merge origin/main
~~~

Nếu có thay đổi đang làm trong bản clone hoặc đã có nhánh cùng tên, cần giữ các thay đổi đó và điều chỉnh cách nhập nhánh trước khi chạy. Không dùng lệnh xóa hay reset để bỏ công việc của nhóm.

Sau khi nhóm xác nhận các mục còn thiếu và kiểm tra thay đổi:

~~~bash
git push -u origin a1-m1-review
~~~

Tạo Pull Request từ a1-m1-review vào main. Dùng Create a merge commit để giữ lịch sử source. Nếu chọn squash/rebase, phải kiểm tra lại khả năng truy cập commit được dẫn trong báo cáo và cập nhật nguồn gốc cho phù hợp.

Các liên kết GitHub tới source, notebook, checkpoint và commit chỉ hoạt động sau khi công bố lịch sử/file tương ứng. Sau merge, kiểm tra GitHub Pages đã build và các link trên README/assignment1 hoạt động; làm bước nộp M1 theo thông báo LMS.

## Tạo lại HTML và PDF sau khi biên tập

~~~bash
python export_html.py assignment1.md --output G-M10_A1_Draft.html
python -m pip install -r requirements-docs.txt
python tools/render_report.py
~~~

Các lệnh này không huấn luyện lại và không sửa kết quả T4 gốc. Luôn kiểm tra bản PDF mới sau khi đổi nội dung hoặc hình.
