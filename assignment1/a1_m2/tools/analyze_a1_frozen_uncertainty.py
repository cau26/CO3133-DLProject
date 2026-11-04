"""Post-hoc exploratory uncertainty for one already-frozen A1 test run.

Reads saved predictions only: no torch, inference, training or dataset download.
Run from the project root:
  python tools/analyze_a1_frozen_uncertainty.py

Wilson intervals concern image sampling under the stated binomial assumptions,
conditional on the fixed checkpoints. They do not measure training-seed variation.
The CNN reference was chosen after observing the run; Holm across these four
comparisons does not turn that exploratory choice into a preregistered analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from statistics import NormalDist
from typing import Mapping, Sequence

try:
    import scipy
    from scipy.stats import binomtest
except ImportError:
    scipy = None
    binomtest = None

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("linear", "mlp", "cnn", "gru", "transformer")
TEST_N = 10000
ALPHA = 0.05
Prediction = tuple[int, int, int]  # official_test_index, true, predicted
REFERENCES = {
    "wilson": "https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html",
    "mcnemar": "https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html",
    "binomial": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
    "holm": "https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html",
}


def count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided Wilson score interval, without continuity correction."""
    count(correct, "correct")
    count(total, "total")
    if not total or correct > total:
        raise ValueError("Require 0 <= correct <= total and total > 0")
    if isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (0.0 if correct == 0 else max(0.0, center - radius),
            1.0 if correct == total else min(1.0, center + radius))


def exact_mcnemar_pvalue(b: int, c: int, *, prefer_scipy: bool = True) -> float:
    """Exact two-sided test of equal directions among discordant outcomes.

    b: reference correct / other wrong; c: other correct / reference wrong.
    For Binomial(b+c, .5), two-sided probability ordering equals twice the
    smaller inclusive tail, clipped at one. No-discordance convention: p=1.
    """
    count(b, "b")
    count(c, "c")
    n = b + c
    if n == 0:
        return 1.0
    if prefer_scipy and binomtest is not None:
        return float(binomtest(b, n, p=0.5, alternative="two-sided").pvalue)
    numerator = 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, numerator / (1 << n))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down Bonferroni adjusted p-values, in original key order."""
    for value in pvalues.values():
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("p-values must be finite numbers in [0,1]")
    adjusted: dict[str, float] = {}
    previous = 0.0
    size = len(pvalues)
    for rank, (name, p) in enumerate(sorted(pvalues.items(), key=lambda pair: pair[1])):
        previous = min(1.0, max(previous, (size - rank) * p))
        adjusted[name] = previous
    return {name: adjusted[name] for name in pvalues}


def validate_predictions(rows: Sequence[Prediction], expected_n: int = TEST_N) -> None:
    count(expected_n, "expected_n")
    if not expected_n or len(rows) != expected_n:
        raise ValueError(f"Expected exactly {expected_n} predictions")
    for position, row in enumerate(rows):
        if len(row) != 3:
            raise ValueError("Each prediction must contain index, truth, prediction")
        index, true, predicted = row
        for value in row:
            count(value, "prediction index/label")
        if index != position:
            raise ValueError("Official test IDs/order must equal range(expected_n); duplicate or reordered ID")
        if true > 9 or predicted > 9:
            raise ValueError("Prediction labels must be in [0,9]")


def read_predictions(path: Path, expected_n: int = TEST_N) -> list[Prediction]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["official_test_index", "true", "predicted"]:
            raise ValueError("Unexpected official-test prediction CSV columns")
        rows = [(int(row["official_test_index"]), int(row["true"]), int(row["predicted"])) for row in reader]
    validate_predictions(rows, expected_n)
    return rows


def paired_counts(reference: Sequence[Prediction], other: Sequence[Prediction],
                  expected_n: int = TEST_N) -> dict[str, int | float]:
    validate_predictions(reference, expected_n)
    validate_predictions(other, expected_n)
    result = dict(both_correct=0, reference_correct_other_wrong=0,
                  other_correct_reference_wrong=0, both_wrong=0)
    for left, right in zip(reference, other):
        if left[:2] != right[:2]:
            raise ValueError("Paired prediction indices/truth labels are not aligned")
        a, b = left[1] == left[2], right[1] == right[2]
        key = ("both_correct" if a and b else "reference_correct_other_wrong" if a else
               "other_correct_reference_wrong" if b else "both_wrong")
        result[key] += 1
    difference = result["reference_correct_other_wrong"] - result["other_correct_reference_wrong"]
    return {**result, "n": expected_n, "correctness_discordant":
            result["reference_correct_other_wrong"] + result["other_correct_reference_wrong"],
            "reference_minus_other_correct_count": difference,
            "reference_minus_other_accuracy": difference / expected_n,
            "reference_minus_other_accuracy_percentage_points": 100 * difference / expected_n}


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    root = root.resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic links are not supported in evidence directories")
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError("Evidence file escapes run directory")
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def safe_child(root: Path, relative: str) -> Path:
    item = PurePosixPath(relative)
    if (not relative or "\\" in relative or ":" in relative or item.is_absolute()
            or any(part in (".", "..") for part in item.parts)):
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    path = (root / Path(*item.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest entry escapes evidence directory")
    return path


def ensure_output_paths(root: Path, outputs: Sequence[Path]) -> None:
    resolved = [Path(path).resolve() for path in outputs]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Analysis output paths must differ")
    if any(path == root.resolve() or path.is_relative_to(root.resolve()) for path in resolved):
        raise ValueError("Analysis outputs must be outside the immutable input directory")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_hash_map(root: Path, entries: dict[str, str]) -> None:
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Missing hash manifest")
    for relative, expected in entries.items():
        if file_hash(safe_child(root, relative)) != expected:
            raise ValueError(f"Evidence hash mismatch: {relative}")


def analyze(root: Path) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Input run directory does not exist")
    before = snapshot(root)
    artifact = read_json(root / "artifact_manifest.json")
    if set(artifact) != set(before) - {"artifact_manifest.json"}:
        raise ValueError("Export manifest must cover every input file except itself")
    verify_hash_map(root, artifact)
    run = read_json(root / "run_manifest.json")
    freeze = read_json(root / "freeze_manifest.json")
    content = {key: value for key, value in freeze.items() if key != "manifest_sha256"}
    self_hash = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
    if freeze.get("manifest_sha256") != self_hash:
        raise ValueError("Freeze self-hash mismatch")
    verify_hash_map(root, freeze["files_sha256"])
    started = read_json(root / "final_test/TEST_STARTED.json")
    complete = read_json(root / "final_test/RESULTS_COMPLETE.json")
    if (set(freeze["models"]) != set(MODELS) or len(complete["models"]) != 5
            or set(complete["models"]) != set(MODELS)):
        raise ValueError("Expected all five frozen, completed models")
    if any(record["run_id"] != run["run_id"] for record in (freeze, started, complete)):
        raise ValueError("Run identity mismatch")
    frozen_hash = file_hash(root / "freeze_manifest.json")
    if any(record["freeze_manifest_sha256"] != frozen_hash for record in (started, complete)):
        raise ValueError("Official test is not bound to this freeze")

    predictions = {}
    scores = {}
    for name in MODELS:
        rows = read_predictions(root / "final_test" / name / "test_predictions.csv")
        if sorted(Counter(row[1] for row in rows).values()) != [1000] * 10:
            raise ValueError("Expected 1,000 truth labels in each of ten test classes")
        metrics = read_json(root / "final_test" / name / "metrics.json")
        if (metrics["model"] != name or metrics["n_eval"] != TEST_N
                or metrics["evaluation_split"] != "official_test" or metrics["test_evaluated"] is not True
                or metrics["run_id"] != run["run_id"] or metrics["freeze_manifest_sha256"] != frozen_hash
                or metrics["checkpoint_sha256"] != freeze["models"][name]["checkpoint_sha256"]
                or metrics["source"]["source_sha256"] != run["source_sha256"]
                or metrics["official_test_dataset_sha256"] != started["official_test_dataset_sha256"]):
            raise ValueError(f"{name}: inconsistent fixed test/checkpoint binding")
        if file_hash(root / name / "best.pt") != metrics["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint bytes differ")
        correct = sum(t == p for _, t, p in rows)
        if not math.isclose(correct / TEST_N, metrics["accuracy"], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"{name}: saved accuracy differs from predictions")
        low, high = wilson_interval(correct, TEST_N)
        predictions[name] = rows
        scores[name] = {"n": TEST_N, "correct": correct, "accuracy": correct / TEST_N,
                        "wilson_95_ci": {"lower": low, "upper": high},
                        "checkpoint_sha256": metrics["checkpoint_sha256"]}

    comparisons = {}
    for name in MODELS:
        if name == "cnn":
            continue
        counts = paired_counts(predictions["cnn"], predictions[name])
        p = exact_mcnemar_pvalue(counts["reference_correct_other_wrong"],
                                 counts["other_correct_reference_wrong"])
        comparisons[name] = {**counts, "exact_two_sided_mcnemar_p": p}
    adjusted = holm_adjust({name: value["exact_two_sided_mcnemar_p"]
                            for name, value in comparisons.items()})
    for name, value in comparisons.items():
        value.update(holm_adjusted_p=adjusted[name],
                     reject_equal_discordant_directions_at_alpha_0_05=adjusted[name] <= ALPHA)
    after = snapshot(root)
    if before != after:
        raise ValueError("Input evidence changed during analysis; no result is valid")
    return {"schema_version": 1, "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "post-hoc exploratory fixed-frozen-run comparison",
            "analysis_tool_sha256": file_hash(Path(__file__)),
            "input_directory": str(root), "run_id": run["run_id"],
            "source_commit": run["source"]["git_commit"],
            "training_seed": read_json(root / "config.json")["seed"],
            "new_training_or_inference": False, "alpha": ALPHA,
            "runtime": {"python": platform.python_version(),
                        "scipy": scipy.__version__ if scipy is not None else None,
                        "exact_test_backend": "scipy.stats.binomtest" if binomtest is not None else
                        "stdlib exact integer binomial tail"},
            "methods": {"interval": "Two-sided 95% Wilson score; no continuity correction; marginal per model",
                        "test": "Exact two-sided McNemar: Binomial(b+c,.5) on discordant correctness pairs",
                        "multiplicity": "Holm step-down Bonferroni across exactly four CNN-versus-other comparisons",
                        "reference_selection": "CNN was identified after observing this run; exploratory, not preregistered"},
            "models": scores, "cnn_paired_comparisons": comparisons,
            "integrity": {"status": "PASS", "all_inputs_unchanged": True,
                          "files_hashed_before_and_after": len(before),
                          "export_manifest_entries_verified": len(artifact),
                          "freeze_entries_verified": len(freeze["files_sha256"]),
                          "freeze_manifest_sha256": frozen_hash,
                          "input_files_sha256": before,
                          "ids_truth_labels_and_10000_rows_aligned": True},
            "limitations_vi": [
                "Đây là phân tích thăm dò hậu nghiệm của các dự đoán đã có; không huấn luyện hoặc suy luận lại trên official test.",
                "Wilson mô tả bất định do lấy mẫu ảnh theo mô hình nhị thức, với checkpoint cố định. Bộ test hữu hạn đang quan sát có accuracy đếm được chính xác; khoảng chỉ có ý nghĩa suy luận nếu chấp nhận giả định lấy mẫu ảnh thích hợp.",
                "Ảnh cần được xem là độc lập theo mô hình phân tích. Fashion-MNIST có số ảnh cố định theo lớp; khác biệt xác suất đúng giữa lớp, ảnh gần trùng hoặc cụm sản phẩm có thể làm mô hình nhị thức đơn giản không phù hợp.",
                "Các khoảng Wilson là khoảng riêng lẻ 95%, không phải khoảng đồng thời cho năm model và không phải khoảng tin cậy của hiệu accuracy giữa hai model. Không so sánh model bằng việc chỉ nhìn khoảng có giao nhau hay không.",
                "McNemar dùng cặp trên cùng ảnh; giả thuyết không là hai hướng bất đồng có cùng xác suất. p-value không phải xác suất CNN tốt hơn hay xác suất giả thuyết không đúng.",
                "Holm chỉ hiệu chỉnh bốn phép so sánh được nêu. Việc chọn CNN sau khi thấy kết quả và các phân tích khác không được sửa hết bởi hiệu chỉnh này; kết quả vẫn mang tính thăm dò.",
                "Không bác bỏ giả thuyết không không chứng minh hai model tương đương; bác bỏ có điều kiện cũng không chứng minh ưu thế phổ quát hay mức cải thiện thực tế quan trọng.",
                "Chỉ có một lần huấn luyện với seed42 cho mỗi model trong run này. Không có mean±SD nhiều seed hoặc khoảng 95% cho độ biến thiên giữa các lần huấn luyện.",
                "Không dùng các kết quả hậu nghiệm này để đổi checkpoint, cấu hình, seed hay lựa chọn thử nghiệm rồi kiểm tra lại official test. Mọi nghiên cứu tiếp theo phải dựa trên validation và một quy trình test độc lập phù hợp."],
            "references": REFERENCES}


def render_markdown(result: dict) -> str:
    lines = ["# Bất định của kết quả test đã freeze", "",
             "**Phân tích thăm dò hậu nghiệm trên một run cố định.** Chỉ đọc CSV dự đoán đã lưu; không huấn luyện hoặc suy luận lại.",
             f"Run `{result['run_id']}`, seed huấn luyện `{result['training_seed']}`, source `{result['source_commit']}`.", "",
             "## Accuracy và khoảng Wilson 95%", "",
             "Các khoảng dưới đây có điều kiện theo checkpoint cố định và mô hình lấy mẫu ảnh; không đo biến thiên do seed huấn luyện. Đây là các khoảng riêng lẻ, không được dùng độ giao nhau để quyết định khác biệt cặp.",
             f"Phương pháp: [Wilson score interval]({REFERENCES['wilson']}).", "",
             "| Model | Đúng / 10.000 | Accuracy | Wilson 95% (theo điểm %) |",
             "|---|---:|---:|---:|"]
    for name, value in result["models"].items():
        ci = value["wilson_95_ci"]
        lines.append(f"| {name} | {value['correct']:,} | {value['accuracy']:.2%} | {100*ci['lower']:.4f}–{100*ci['upper']:.4f} |")
    lines += ["", "## So sánh cặp CNN với bốn model", "",
              "b: CNN đúng/model kia sai; c: model kia đúng/CNN sai. Hiệu accuracy = (b−c)/10.000. Kiểm định hai phía chỉ dùng b+c ảnh bất đồng về đúng/sai.",
              f"Dùng [McNemar exact]({REFERENCES['mcnemar']}) qua [binomial test]({REFERENCES['binomial']}); [Holm]({REFERENCES['holm']}) hiệu chỉnh đúng bốn phép so sánh, ngưỡng 0,05.", "",
              "| Model đối chiếu | b | c | Hiệu CNN (điểm %) | p exact hai phía | p sau Holm | Diễn giải ở 0,05 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name, value in result["cnn_paired_comparisons"].items():
        decision = ("Bác bỏ H0 có điều kiện" if value["reject_equal_discordant_directions_at_alpha_0_05"]
                    else "Chưa đủ bằng chứng phân biệt")
        lines.append(f"| {name} | {value['reference_correct_other_wrong']} | {value['other_correct_reference_wrong']} | {value['reference_minus_other_accuracy_percentage_points']:+.2f} | {value['exact_two_sided_mcnemar_p']:.9g} | {value['holm_adjusted_p']:.9g} | {decision} |")
    rejected = [name for name, value in result["cnn_paired_comparisons"].items()
                if value["reject_equal_discordant_directions_at_alpha_0_05"]]
    unresolved = [name for name in result["cnn_paired_comparisons"] if name not in rejected]
    lines += ["", "Trong bốn đối chiếu này, kết quả với " + ", ".join(rejected)
              + " có p sau Holm nhỏ hơn 0,05, nên bác bỏ H0 có điều kiện theo các giả định đã nêu; kết quả với " + ", ".join(unresolved)
              + " còn chưa phân biệt được ở mức 0,05. Điều này không chứng minh tương đương hoặc ưu thế trên mọi tập dữ liệu.",
              "CNN được chọn làm đối chiếu sau khi đã thấy điểm của run. Vì vậy bảng này là mô tả thăm dò, không phải kiểm định xác nhận đã đăng ký trước. Không dùng bảng để chỉnh mô hình rồi chạy official test lại.",
              "", "## Giả định và giới hạn", ""]
    lines += ["- " + item for item in result["limitations_vi"]]
    integrity = result["integrity"]
    lines += ["", "## Toàn vẹn bằng chứng", "",
              f"Đã so SHA-256 trước/sau của {integrity['files_hashed_before_and_after']} file đầu vào: không đổi. Kiểm tra {integrity['export_manifest_entries_verified']} mục manifest export và {integrity['freeze_entries_verified']} file freeze. Mỗi model có đúng 10.000 ID theo thứ tự 0–9.999, nhãn 0–9, 1.000 ảnh/lớp; ID/nhãn thật khớp giữa cả năm model. Accuracy tính lại khớp số đã lưu.",
              f"Backend exact: `{result['runtime']['exact_test_backend']}`; SciPy `{result['runtime']['scipy']}`; Python `{result['runtime']['python']}`.",
              "Đầu ra phân tích nằm ngoài thư mục run. Các loss, checkpoint và kết quả gốc không bị sửa.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/a1_m2_final_returned_867c74469240")
    parser.add_argument("--json", type=Path, default=ROOT / "verification/A1_FROZEN_UNCERTAINTY.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "verification/A1_FROZEN_UNCERTAINTY.md")
    parser.add_argument("--overwrite", action="store_true", help="Replace derived analysis files only, outside the input run")
    args = parser.parse_args(argv)
    ensure_output_paths(args.out, [args.json, args.markdown])
    for path in (args.json, args.markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Analysis output already exists: {path}; use --overwrite for derived outputs")
    result = analyze(args.out)
    for path in (args.json, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["integrity"]["status"], "json": str(args.json),
                      "markdown": str(args.markdown), "new_training_or_inference": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
