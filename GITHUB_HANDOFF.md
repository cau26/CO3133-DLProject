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

Gói ZIP có kèm handoff/CO3133_M1_Review.bundle, chứa lịch sử Git của nhánh local a1-m1-review. Nhánh dựa trên main ở commit 44cb576; commit đầu lưu source khớp lượt Colab, các commit tiếp theo thêm báo cáo, kết quả và cập nhật thông tin nhóm. Tác giả các commit chuẩn bị là Codex, không được diễn giải là commit do sinh viên tự viết.

Lịch sử này giúp giữ đúng liên kết đến commit lưu source trong COLAB_RUN.md. Chỉ upload các file qua giao diện web sẽ không đưa commit đó lên GitHub.

## Các bước trên Windows

1. Nếu tài khoản n1velo chưa có quyền ghi vào repo, nhờ Cầu thêm tài khoản này trong Settings → Collaborators và chấp nhận lời mời.
2. Giải nén CO3133_M1_Draft_Package_v2.zip. Mở thư mục handoff, có file CO3133_M1_Review.bundle.
3. Trong File Explorer, bấm thanh địa chỉ, gõ cmd rồi Enter. Cửa sổ lệnh sẽ mở tại thư mục handoff.
4. Gõ git --version. Nếu chưa có Git, cài Git từ https://git-scm.com rồi mở lại cửa sổ lệnh.
5. Chạy lần lượt các lệnh sau. Lệnh clone tạo thư mục mới CO3133-publish; nếu tên đó đã tồn tại, chọn tên thư mục mới và thay cả lệnh cd tương ứng. Không cần chép file thủ công hoặc git add/commit vì bundle đã chứa các commit chuẩn bị.

~~~bash
git clone -b a1-m1-review CO3133_M1_Review.bundle CO3133-publish
cd CO3133-publish
git remote set-url origin https://github.com/cau26/CO3133-DLProject.git
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch origin
git merge --ff-only origin/main
git status
~~~

Lệnh config giúp clone từ bundle lấy được mọi nhánh của repo, gồm main. Khi main còn ở phiên bản đã kiểm tra, merge sẽ báo Already up to date. Nếu fetch/merge báo lỗi, dừng ở lệnh đó và gửi nội dung lỗi để xử lý; không dùng force/reset. Tùy chọn --ff-only sẽ dừng nếu main có thay đổi mới cần gộp, giúp tránh tự tạo một bản merge chưa được rà soát.

6. Mở các file trong CO3133-publish để rà soát báo cáo, vai trò, AI log và kết quả. Sau đó đưa nhánh lên GitHub:

~~~bash
git push -u origin a1-m1-review
~~~

Nếu Git mở cửa sổ đăng nhập, đăng nhập tài khoản GitHub có quyền ghi vào repo. Nếu push bị từ chối, giữ nguyên file và gửi nội dung lỗi để xử lý.

7. Mở https://github.com/cau26/CO3133-DLProject, chọn Pull requests → New pull request, base: main, compare: a1-m1-review. Kiểm tra Files changed; tiêu đề gợi ý: Complete A1 M1 Draft with Colab results. Bấm Create pull request.
8. Cầu rà soát phần EDA/MLP; Lâm rà soát báo cáo và phần tích hợp. Khi nhóm đã đọc và thống nhất bản này, chọn Create a merge commit → Confirm merge để giữ lịch sử source. Nếu chọn squash/rebase, phải kiểm tra lại khả năng truy cập commit được dẫn trong báo cáo và cập nhật nguồn gốc cho phù hợp.
9. Trong Settings → Pages, nếu chưa cấu hình, chọn Deploy from a branch, branch main, thư mục / (root), Save. Nếu đã cấu hình đúng thì giữ nguyên. Đợi lượt build Pages trong Actions hoàn tất rồi mở các link dưới đây:

- https://cau26.github.io/CO3133-DLProject/
- https://cau26.github.io/CO3133-DLProject/assignment1.html
- https://cau26.github.io/CO3133-DLProject/G-M10_A1_Draft.html

Kiểm tra hình, bảng, PDF, notebook và checkpoint. File ZIP và bundle là gói chuyển giao cho nhóm; không cần đưa chúng vào repo. Source, tài liệu và outputs bên trong đã nằm trong nhánh vừa push.

Các liên kết GitHub tới source, notebook, checkpoint và commit chỉ hoạt động sau khi công bố lịch sử/file tương ứng. Sau merge, kiểm tra GitHub Pages đã build và các link trên README/assignment1 hoạt động; làm bước nộp M1 theo thông báo LMS.

## Tạo lại HTML và PDF sau khi biên tập

~~~bash
python export_html.py assignment1.md --output G-M10_A1_Draft.html
python -m pip install -r requirements-docs.txt
python tools/render_report.py
~~~

Các lệnh này không huấn luyện lại và không sửa kết quả T4 gốc. Luôn kiểm tra bản PDF mới sau khi đổi nội dung hoặc hình.
