# CO3133 - Deep Learning and Its Applications

## Thông tin môn học và nhóm

**Trường:** Ho Chi Minh City University of Technology, VNU-HCM

**Khoa:** Faculty of Computer Science and Engineering

**Môn học:** Deep Learning and Its Applications - CO3133, Semester-261

**Giảng viên:** Lê Thành Sách

**Nhóm:** G-M10 (2 thành viên, đã xin phép theo xác nhận của leader)

| Thành viên | MSSV | Vai trò đã xác nhận | GitHub |
|---|---|---|---|
| Trần Gia Lâm | 2352670 | Leader; chạy thí nghiệm Colab được cung cấp | [n1velo](https://github.com/n1velo) |
| Nguyễn Hữu Cầu | 2352129 | Phụ trách EDA và MLP | [cau26](https://github.com/cau26) |

**Repository:** [https://github.com/cau26/CO3133-DLProject](https://github.com/cau26/CO3133-DLProject)

## Các assignment

- [Assignment 1 - M1 Draft](assignment1.md)
- [Assignment 2](assignment2.md)
- [Assignment 3](assignment3.md)

## Assignment 1: bản nháp hiện tại

- [Báo cáo HTML](G-M10_A1_Draft.html) và [báo cáo PDF](G-M10_A1_Draft.pdf)
- [Bảng kết quả](outputs/a1_t4/comparison.csv)
- [Notebook Colab gốc](notebooks/A1_Colab_Run.ipynb)
- [Nguồn gốc và kiểm chứng](COLAB_RUN.md)
- [Yêu cầu M1 và thông tin còn cần xác nhận](DRAFT1_CHECKLIST.md)
- [Mã nguồn](https://github.com/cau26/CO3133-DLProject/tree/main/a1)

Kết quả validation T4: Linear accuracy 86,62%, macro-F1 0,8644; MLP accuracy 89,30%, macro-F1 0,8924. Cả hai chọn checkpoint epoch 9 theo validation loss. Test chưa được đánh giá.

## Cài đặt và chạy lại

Dùng Python 3.12 hoặc môi trường Colab tương thích. Bản ghi gốc dùng Python 3.13.15, torch 2.8.0+cu128 và Tesla T4; xem metadata gốc trong metrics.json. Python 3.12 là lựa chọn đã dùng để kiểm tra CPU trong môi trường trợ lý, không phải phiên bản của Colab gốc.

~~~bash
python -m pip install -r requirements-a1.txt
~~~

Trên Colab, chọn runtime có GPU và kiểm tra CUDA trước khi chạy. Notebook đính kèm là bản ghi gốc dùng starter ZIP; để tái lập từ repo, clone repo hoặc lấy source của commit đã ghi trong COLAB_RUN.md rồi chạy từ thư mục gốc.

### Chạy trên CUDA

~~~bash
python run_a1.py --stage all --device cuda --out outputs/a1_t4_rerun
python run_a1.py --stage evaluate --out outputs/a1_t4_rerun
~~~

### Chạy trên CPU

~~~bash
python run_a1.py --stage all --device cpu --out outputs/a1_cpu_rerun
python run_a1.py --stage evaluate --out outputs/a1_cpu_rerun
~~~

Dữ liệu được tự tải vào data/. Một thư mục output mới được dùng cho mỗi lượt chạy để tránh ghi đè kết quả. Khi thư mục rerun đã có kết quả, dùng tên mới cho lần tiếp theo. Lệnh evaluate đọc cấu hình trong thư mục output, nên hai lệnh evaluate ở trên giữ đúng thiết bị của lượt chạy tương ứng.

### Chỉ chạy EDA hoặc tạo lại báo cáo tự động

~~~bash
python run_a1.py --stage eda --device cpu --out outputs/eda_check
python run_a1.py --stage report --out outputs/a1_t4
~~~

Lệnh report tạo lại draft_report.md tự động trong thư mục output; bản đã biên tập nằm ở assignment1.md. Cấu hình mặc định ở configs/a1.json; cấu hình lượt T4 đã báo cáo ở outputs/a1_t4/config.json.

## Checkpoint và file kết quả

- [Linear checkpoint](outputs/a1_t4/linear/best.pt), [metrics](outputs/a1_t4/linear/metrics.json), [history](outputs/a1_t4/linear/history.csv)
- [MLP checkpoint](outputs/a1_t4/mlp/best.pt), [metrics](outputs/a1_t4/mlp/metrics.json), [history](outputs/a1_t4/mlp/history.csv)
- [Split indices](outputs/a1_t4/split_indices.npz), [data summary](outputs/a1_t4/data_summary.json)

Các file T4 gốc được giữ nguyên. Không sử dụng số liệu reference_cpu hay smoke_t4 trong bảng kết quả chính.

## AI Usage Disclosure

ChatGPT hỗ trợ tìm hiểu yêu cầu, soạn mã khởi đầu, hướng dẫn chạy, kiểm chứng kết quả và biên tập báo cáo. Trần Gia Lâm xác nhận đã chạy lại bộ mã trên Colab. Cầu phụ trách EDA và MLP theo xác nhận của leader. Việc rà soát nội dung cuối cùng còn cần nhóm xác nhận. Chi tiết công cụ, prompt, phần bị ảnh hưởng và các bước kiểm chứng nằm trong [AI_USAGE.md](AI_USAGE.md).

## Tình trạng M1

Đã có bằng chứng thực thi phần kỹ thuật tối thiểu. Bộ file cần được nhóm rà soát, công bố lên repo/Pages và nộp theo thông báo M1 trên LMS. Xem DRAFT1_CHECKLIST.md để xử lý các mục đăng ký, rà soát cuối cùng và nộp bài chưa được xác nhận.
