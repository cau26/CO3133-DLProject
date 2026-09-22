"""Generate a partial draft from actual saved metrics; never invent results."""
from pathlib import Path

from .common import load_json, save_csv


def create_report(output_dir):
    out = Path(output_dir)
    results = [load_json(out / model / "metrics.json") for model in ["linear", "mlp"]]
    if results[0]["data"]["split_sha256"] != results[1]["data"]["split_sha256"]:
        raise ValueError("The models used different splits: rerun with one shared configuration.")
    if results[0]["config"] != results[1]["config"]:
        raise ValueError("The models used different configurations: document a controlled comparison first.")
    if results[0]["source"]["source_sha256"] != results[1]["source"]["source_sha256"]:
        raise ValueError("The source files changed between model runs. Rerun both models together.")
    cfg, data = results[0]["config"], results[0]["data"]
    rows = [{"model": r["model"], "val_accuracy": r["accuracy"], "val_macro_f1": r["macro_f1"],
             "parameters": r["parameters"], "best_epoch": r["best_epoch"],
             "train_seconds": r["train_seconds"],
             "inference_ms_per_image": r["inference"]["per_image_ms_amortized"]} for r in results]
    save_csv(out / "comparison.csv", rows)
    table = ["| Model | Val accuracy | Val macro-F1 | Parameters | Best epoch | Train (s) | Inference (ms/image) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    table += [f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['parameters']:,} | "
              f"{r['best_epoch']} | {r['train_seconds']:.2f} | "
              f"{r['inference']['per_image_ms_amortized']:.6f} |" for r in results]
    comparison = "\n".join(table)
    (out / "comparison.md").write_text(comparison + "\n", encoding="utf-8")
    delta = (results[1]["accuracy"] - results[0]["accuracy"]) * 100
    parts = [
        "# Assignment 1 - M1 Draft\n",
        f"**Group:** {cfg['group']}  \n**Course:** CO3133, Semester-261  \n**Instructor:** Lê Thành Sách\n",
        "**Members:** " + "; ".join(cfg.get("members", [])) + "\n",
        "> Bản nháp được tạo từ lần chạy code thực tế. Nhóm cần đọc, kiểm chứng và bổ sung phân tích trước khi nộp. Các số dưới đây là validation; chưa phải test.\n",
        "## Part 1 - Problem and Data Description\n",
        "Bài toán: từ một ảnh xám của một sản phẩm thời trang, dự đoán một trong 10 lớp. "
        "Đơn vị dự đoán là một ảnh; đầu vào có kích thước 1 × 28 × 28, đầu ra là 10 logits. "
        "Đây là bài toán phân loại đa lớp, đơn nhãn.\n",
        "Dữ liệu: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), do Zalando Research công bố, giấy phép MIT. "
        "Ảnh và nhãn được tải dưới dạng IDX nén gzip; torchvision kiểm tra MD5 khi tải. "
        "Bản dữ liệu dùng trong lần chạy được nhận diện qua nguồn tải và checksum chuẩn của torchvision.\n",
        f"Chia tập: {data['train']:,} train, {data['validation']:,} validation, {data['test']:,} test. "
        f"Tách stratified từ 60.000 ảnh train chính thức, seed {cfg['seed']}; giữ test chính thức độc lập. "
        "Chỉ số train/validation không trùng nhau và được lưu trong `split_indices.npz`. "
        "Chưa kiểm tra trùng lặp theo nội dung ảnh; kiểm tra chỉ số không thay thế kiểm tra đó.\n",
        "![Phân bố lớp](eda/class_distribution.png)\n\n![Ảnh mẫu](eda/samples.png)\n",
        "Xem số lượng chính xác trong [class_counts.csv](eda/class_counts.csv). "
        "Các lớp cân bằng về số mẫu trong cách chia này. Ảnh có độ phân giải thấp và chỉ có một kênh; "
        "kết quả trên benchmark này chưa chứng minh khả năng tổng quát hóa sang ảnh sản phẩm thực tế.\n",
        "Tiền xử lý: `ToTensor()` chuyển uint8 [0,255] thành float32 [0,1]. "
        "Không augmentation, không chuẩn hóa theo mean/std trong cấu hình baseline này. "
        "Dùng Dataset có sẵn của torchvision, Subset để chia tập và DataLoader để tạo batch.\n",
        "![Batch sau tiền xử lý](eda/batch_preview.png)\n",
        "## Part 2 - Methodology\n",
        "Pipeline: ảnh/nhãn → ToTensor → split và DataLoader → Linear hoặc MLP → "
        "CrossEntropyLoss → Adam cập nhật trọng số → validation → chọn checkpoint → argmax và đánh giá.\n",
        f"- Linear: Flatten → Linear(784,10).\n- MLP: Flatten → Linear(784,{cfg['hidden_size']}) → ReLU → Linear({cfg['hidden_size']},10).\n",
        "Linear dùng phép biến đổi tuyến tính trực tiếp trên pixel đã làm phẳng. "
        "MLP thêm hidden layer và ReLU để học quan hệ phi tuyến. Cả hai không có cơ chế tích chập "
        "để khai thác trực tiếp cấu trúc không gian cục bộ. Không áp dụng Softmax trước CrossEntropyLoss.\n",
        f"Adam, learning rate {cfg['learning_rate']}, batch size {cfg['batch_size']}, "
        f"{cfg['epochs']} epochs, seed {cfg['seed']}, {cfg['cpu_threads']} CPU threads, "
        f"device {cfg['device']}. Không scheduler, dropout, weight decay, early stopping hoặc mixed precision. "
        "Lưu checkpoint có validation loss thấp nhất; nếu bằng nhau, giữ checkpoint xuất hiện trước.\n",
        "Thiết kế thí nghiệm: kiểm tra giả thuyết thêm hidden layer phi tuyến cải thiện phân loại. "
        "Yếu tố thay đổi là kiến trúc; dữ liệu, split, tiền xử lý, loss, optimizer, learning rate, "
        "batch size và số epoch được giữ cố định. Số tham số không được khớp bằng nhau, nên chưa thể "
        "quy toàn bộ chênh lệch cho tính phi tuyến. Cấu hình này là điểm khởi đầu, chưa được tìm kiếm siêu tham số.\n",
        "Code nhóm cần hiểu và kiểm chứng: mô hình, train/validation loop, chọn checkpoint, EDA và tổng hợp kết quả. "
        "Thư viện: PyTorch cho layers/autograd/optimizer/loss, torchvision cho dữ liệu và ToTensor, "
        "scikit-learn cho split/metrics, matplotlib cho biểu đồ. AI hỗ trợ soạn code; xem AI disclosure.\n",
        "## Part 3 - Implementation Results\n", comparison + "\n",
        f"MLP trừ Linear: {delta:+.2f} điểm phần trăm validation accuracy trong lần chạy này. "
        "Cần đọc thêm macro-F1, số tham số và thời gian để đánh giá sự đánh đổi. "
        "Đây là một seed; chưa có mean/std nhiều lần chạy hoặc kết luận về ý nghĩa thống kê.\n",
        "Training time trong bảng gồm nạp batch, forward/backward, cập nhật và thu metric; "
        "không gồm validation. Inference là thời gian forward thuần, input đã ở device, "
        f"batch {cfg['batch_size']}, warm-up 10 batch, đo 50 batch; ms/image là giá trị chia trung bình theo batch. "
        "Không diễn giải nó thành độ trễ một yêu cầu đơn ảnh hoặc thời gian toàn pipeline.\n",
    ]
    for r in results:
        name = r["model"]
        parts += [f"### {name.upper()}\n",
                  f"![Curves]({name}/curves.png)\n\n![Confusion matrix]({name}/confusion_matrix.png)\n\n![Examples]({name}/examples.png)\n",
                  "Các cặp nhầm nhiều nhất trên validation (số liệu thực tế):\n"]
        parts += [f"- {e['true_class']} → {e['predicted_class']}: {e['count']} ảnh "
                  f"({e['rate_within_true_class']:.1%} số ảnh của lớp thật).\n" for e in r["top_errors"]]
        parts += [f"Máy chạy: {r['environment']['cpu']}; device {r['environment']['device']}. "
                  f"Xem cấu hình, phiên bản, fingerprint code, thời gian và checkpoint trong [{name}/metrics.json]({name}/metrics.json).\n"]
    parts += [
        "### Phân tích cần nhóm bổ sung\n",
        "1. Đọc hai learning curves: từ epoch nào validation dừng cải thiện trong khi train tiếp tục cải thiện?\n"
        "2. Chọn ít nhất hai ảnh dự đoán sai, mô tả chi tiết nhìn thấy và đưa ra giả thuyết nguyên nhân.\n"
        "3. Giải thích accuracy, macro-F1 và trade-off tốc độ/số tham số của hai mô hình.\n"
        "4. Ghi lỗi triển khai thực sự gặp và cách sửa; nếu không có, ghi rõ.\n",
        "### Giới hạn và kế hoạch Final\n",
        "Draft mới có Linear và MLP, một seed và một cấu hình. Test chưa được đánh giá. "
        "Tiếp theo triển khai CNN tự thiết kế, LSTM hoặc GRU, Transformer; mở rộng so sánh, "
        "phân tích biểu diễn/inductive bias và kiểm tra test sau khi chốt lựa chọn bằng validation. "
        "Hoàn thiện báo cáo, slides, video và các liên kết theo handbook.\n",
        "### AI Usage Disclosure\n",
        "ChatGPT hỗ trợ tạo code, cấu trúc báo cáo và hướng dẫn. Bản báo cáo này lấy số từ metric thật; "
        "các nhận xét mang tính giả thuyết cần nhóm kiểm chứng. Nhóm phải điền người sử dụng, "
        "prompt, thời điểm, phần bị ảnh hưởng, cách kiểm chứng và người chịu trách nhiệm vào AI_USAGE.md. "
        "Chưa xác nhận kiểm chứng bởi thành viên nhóm tại thời điểm sinh tự động.\n",
        "### Tài liệu tham khảo\n",
        "- Course Project Handbook CO3133, revision 14 September 2026, mục 3, 5, 7.4 và Part II.\n"
        "- [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).\n"
        "- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).\n",
    ]
    (out / "draft_report.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"Draft and comparison saved: {out / 'draft_report.md'}", flush=True)
