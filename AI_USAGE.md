# AI Usage Disclosure Log
## CO3133 – Deep Learning and Its Applications, Semester-261
**Group:** G-M10

This file logs all generative AI tool usage across Assignments 1–3, following the course's academic integrity policy.

---

## Current Status


---

## Log Template (copy for each AI usage entry)

### Entry #1
- **Tool:** [Claude Sonnet 5]
- **Used by:** [Nguyễn Hữu Cầu]
- **Time / Stage:** [Week 0]
- **Purpose:** [Building Github skeleton]
- **Affected assignment section(s):** []
- **Prompt summary:** [Uploaded the course handbook PDF and asked for step-by-step guidance to create a GitHub Pages skeleton that satisfies all handbook requirements; also asked for help drafting placeholder content for the landing page, assignment pages, and AI usage log.]
- **AI contribution:** [Explained the GitHub Pages setup steps (creating a repo, enabling Pages via Settings, renaming a repo, clearing repo history); drafted Markdown templates for README.md (landing page), assignment1/2/3.md (page skeletons with the required section headers), and AI_USAGE.md (this log's format), based on the mandatory content list in the handbook.]
- **Student verification:** [Cross-checked each generated template against the handbook's required content checklist (Sections 2.1–2.3, 3, 4.1, 5.1–5.4) to confirm no required field was missing; manually filled in real group/member information (names, student IDs) instead of leaving AI-generated placeholders; verified the GitHub Pages link loaded correctly after enabling it in repository Settings.]
- **Responsible member:** [Nguyễn Hữu Cầu]

---

*Add new entries below as needed, following the same template.*


---


## Entry #2 - Hỗ trợ mã nguồn và chạy lại M1

- **Tool:** ChatGPT; mã model cụ thể không được ghi nhận.
- **Used by:** Trần Gia Lâm.
- **Time / stage:** Chuẩn bị Assignment 1 M1 Draft; lượt Colab có metadata ngày 22/09/2026 UTC.
- **Purpose:** Hiểu handbook, tổ chức công việc, tạo mã khởi đầu và hướng dẫn chạy pipeline Fashion-MNIST.
- **Affected sections/files:** a1/*.py, run_a1.py, export_html.py, configs/a1.json, requirements-a1.txt; cấu trúc báo cáo tự động.
- **Representative prompts:** “hãy hướng dẫn mình từ bước từ đầu đi, về mọi thứ”; “mình cần chia task, mình là gia lâm, là leader, còn bạn kia là cầu”; yêu cầu đọc notebook và outputs để chuẩn bị đưa lên GitHub.
- **AI contribution:** Soạn pipeline EDA, split, Dataset/DataLoader, Linear, MLP, train/validation, checkpoint, metric và hình; hướng dẫn cài đặt và sử dụng Colab/Git.
- **Group actions confirmed:** Lâm xác nhận tải starter ZIP lên Colab và chạy lại; notebook ghi lần thử 1 epoch rồi lần T4 10 epoch. Source hash trong kết quả khớp bộ mã khởi đầu; chưa thấy bằng chứng source đã được sửa trước lần chạy này.
- **Student verification confirmed:** Có bằng chứng notebook thực thi và file kết quả do người dùng gửi. Chưa xác nhận thành viên đã rà soát toàn bộ logic mã và nội dung báo cáo cuối cùng.
- **Responsible member:** Trần Gia Lâm phụ trách điều phối với vai trò leader; xác nhận rà soát cuối cùng còn chờ nhóm.
- **Verification sources:** Handbook mục 3/4.2/5/7.4; Fashion-MNIST; PyTorch Quickstart; source, notebook và outputs/a1_t4.

## Entry #3 - Rà soát và biên tập báo cáo M1

- **Tool / used by:** ChatGPT, qua trao đổi với Trần Gia Lâm; mã model cụ thể không được ghi nhận.
- **Stage:** Sau lần chạy T4, trước công bố Draft 1.
- **Purpose:** Đối chiếu yêu cầu, sửa Markdown, chèn hình, kiểm tra phân tích và chuẩn bị bộ tài liệu có thể rà soát.
- **Representative prompts:** “đây là bản draft mình viết thử, hãy đánh giá xem sao”; “bây giờ mình cần bạn làm giúp mình, và check lại yêu cầu xem mình còn thiếu cái gì ... lưu ý không được chế nhé”.
- **Affected files:** assignment1.md, README.md, COLAB_RUN.md, DRAFT1_CHECKLIST.md, AI_USAGE.md, bản HTML/PDF và sơ đồ pipeline.
- **Edits by AI:** Gom ba phần báo cáo; sửa bảng và đường dẫn hình; phân biệt train/validation/fit wall time; thêm phân tích các ảnh có chỉ số 18, 164, 169; tính số lỗi/recall theo lớp từ CSV; làm rõ giới hạn một seed, tiêu chí checkpoint và nguồn gốc Git.
- **Verification by AI:** Đối chiếu 20 epoch và JSON giữa notebook/ZIP; kiểm tra source hash và split; tính lại accuracy/macro-F1 từ 6.000 dự đoán; nạp hai checkpoint T4 trên CPU với 0 dự đoán khác CSV. Xem verification/colab_review.json.
- **Provenance:** Source được lưu vào một commit sau lượt Colab; không thay git_commit = null trong metrics gốc. Không dùng số liệu tham khảo CPU làm số liệu thực nghiệm của nhóm.
- **Student review:** Leader đã xác nhận Cầu phụ trách EDA và MLP. Chờ Lâm/Cầu xác nhận các phần thực sự đã đọc, sửa và kiểm chứng; thông tin phụ trách không được diễn giải thành việc tự viết toàn bộ mã hoặc đã hoàn thành mọi bước kiểm chứng.
- **Final review responsibility:** Lâm điều phối việc xác nhận bản nộp; việc rà soát cuối cùng chưa được ghi là hoàn thành.
- **Verification sources:** Handbook; các file gốc trong outputs/a1_t4; notebook Colab; source archive commit trong COLAB_RUN.md; tài liệu PyTorch và Fashion-MNIST.

## Thông tin còn cần nhóm xác nhận

1. Lâm/Cầu đã kiểm chứng những mục cụ thể nào và đã rà soát bản báo cáo cuối cùng hay chưa.
2. Các chỉnh sửa hoặc lần sử dụng công cụ AI khác chưa có trong log này.
3. Tên công cụ/model trong Entry #1 hiện có cần được nhóm đối chiếu; entry đó được giữ theo repo, không được trợ lý xác minh độc lập.
